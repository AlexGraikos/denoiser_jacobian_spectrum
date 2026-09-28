import os
import torch
from torch.utils.data import DataLoader
from dataset import H5Dataset
from sit import SiT_models
from copy import deepcopy
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.utils import tqdm
import argparse

## Helpers ##
@torch.no_grad()
def update_ema(ema_model, model, decay):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 
## ####### ##

def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accumulation_steps,
    )
    device = accelerator.device

    ## Data ##
    # Create results directory
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.save_dir, args.experiment_name), exist_ok=True)

    # ImageNet VAE features dataset
    trainset = H5Dataset(os.path.join(args.data_dir, 'dataset.h5'))
    trainloader = DataLoader(
        trainset,
        batch_size=int(args.batch_size // accelerator.num_processes),
        shuffle=True,
        drop_last=True,
        num_workers=max(1,int(args.num_workers // accelerator.num_processes)),
    )
    if accelerator.is_main_process:
        print("Dataset has", len(trainset), "samples")
    ## #### ##

    ## Model ##
    # Denoiser
    block_kwargs = {"fused_attn": True, "qk_norm": False}
    latent_size = 256 // 8
    model = SiT_models[args.model_name](
        input_size=latent_size,
        in_channels=4,
        out_channels=4, # Sigma is not learned
        num_classes=1000,
        use_cfg=True,
        **block_kwargs,
    )
    model.train().to(device)
    # Disable automatic condition dropping from SiT
    model.y_embedder.dropout_prob = 0.0

    if accelerator.is_main_process:
        print(f"Training {args.model_name}")
        print(f"# model parameters {sum([p.numel() for p in model.parameters()]):,d}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    opt.zero_grad(set_to_none=True)

    # EMA copy of model
    ema = deepcopy(model).eval().to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    ## ##### ##

    ## Train ##
    if args.resume:
        model.load_state_dict(
            torch.load(os.path.join(args.save_dir, args.experiment_name, f"model_{args.resume:06d}.pt"), map_location=device),
            strict=True
        )
        ema.load_state_dict(
            torch.load(os.path.join(args.save_dir, args.experiment_name, f"model_ema_{args.resume:06d}.pt"), map_location=device),
            strict=True
        )
        if args.use_jacobian:
            ## Debug -- fails when loading without g/r values ##
            try:
                g_vals = torch.load(os.path.join(args.save_dir, args.experiment_name, f"g_vals_{args.resume:06d}.pt"), map_location=device)
                r_vals = torch.load(os.path.join(args.save_dir, args.experiment_name, f"r_vals_{args.resume:06d}.pt"), map_location=device)
            except Exception as e:
                g_vals = torch.ones((1000,), device=device)
                r_vals = torch.ones((1000,), device=device)
            ## ########### ##

        global_idx = args.resume
        model.train()
        model.y_embedder.dropout_prob = 0.0
        ema.eval()
        print("Resuming from", os.path.join(args.save_dir, args.experiment_name, f"model_{args.resume:06d}.pt"))
    else:
        update_ema(ema, model, decay=0)
        global_idx = 0
        ema.eval()

        if args.use_jacobian:
            g_vals = torch.ones((1000,), device=device)
            r_vals = torch.ones((1000,), device=device)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    model, opt, trainloader = accelerator.prepare(
        model, opt, trainloader
    )

    progress_bar = tqdm(
        range(0, args.train_iterations),
        initial=global_idx,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # EMA of losses
    log_ema_decay = 0.99
    ema_velocity_loss = None
    ema_jac_loss = None

    # Gain target
    k_vals = torch.ones_like(g_vals)

    while global_idx < args.train_iterations:
        for batch in trainloader:
            moments, y = batch
            batch_size = y.shape[0]

            # Sample VAE latent
            z = sample_posterior(moments.to(device), latents_scale, latents_bias)

            # Condition drop
            y = y.long().to(device)
            drop = torch.rand(y.shape, device=device) < 0.1
            y = torch.where(drop, torch.full_like(y, 1000), y)

            timesteps = torch.rand(batch_size, device=device).view(batch_size,1,1,1)
            eps = torch.randn_like(z)
            zt = (1-timesteps)*eps + timesteps*z
            # velocity = (z - eps) # for v-pred
            velocity = (z-zt)/(1-timesteps).clamp_min(5e-2)

            with accelerator.accumulate(model):
                # Velocity prediction loss
                z0_pred = model(zt, timesteps.view(batch_size), y)
                velocity_pred = (z0_pred-zt)/(1-timesteps).clamp_min(5e-2)
                velocity_loss = ((velocity_pred - velocity)**2).mean()
                total_loss = velocity_loss

                # Jacobian loss
                if args.use_jacobian:
                    # Use the residual as the direction
                    eigendir = z0_pred.detach()-z

                    ## Threshold ##
                    lvl = 0.95
                    th = torch.quantile(eigendir.abs().mean(1).view(batch_size,-1), lvl, dim=1).view(batch_size,1,1)
                    mask = (eigendir.abs().mean(1) >= th).float().unsqueeze(1)
                    eigendir_masked = mask*eigendir

                    # Use masked eigendir
                    eigendir = eigendir_masked
                    # ## ######### ##

                    t_indices = (timesteps.view(batch_size)*1000).long()
                    k = k_vals[t_indices].view(batch_size,1,1,1)

                    # Velocity regularization
                    zt_eigen = zt - eigendir/k
                    z0_pred_eigen = model(zt_eigen, timesteps.view(batch_size), y)

                    velocity_eigen = (z-zt_eigen)/(1-timesteps).clamp_min(5e-2)
                    velocity_pred_eigen = (z0_pred_eigen-zt_eigen)/(1-timesteps).clamp_min(5e-2)
                    velocity_loss_eigen = ((velocity_pred_eigen - velocity_eigen)**2).mean()

                    total_loss = total_loss + args.jacobian_w*velocity_loss_eigen

                    # Update statistics
                    with torch.no_grad():
                        # gain = ||Jr||/||r||
                        delta = (eigendir/k).flatten(1).norm(dim=1)
                        gain = (z0_pred_eigen - z0_pred).flatten(1).norm(dim=1)/delta 

                        # R(r) = (r/||r|)^T Jr/||r||
                        rnorm = (eigendir/k)/delta.view(batch_size,1,1,1)
                        Jdnorm = -(z0_pred_eigen - z0_pred)/delta.view(batch_size,1,1,1)
                        rayleigh = (rnorm*Jdnorm).flatten(1).sum(dim=1)

                        # EMA
                        alpha = 0.99
                        g_vals[t_indices] = alpha*g_vals[t_indices] + (1-alpha)*gain
                        r_vals[t_indices] = alpha*r_vals[t_indices] + (1-alpha)*rayleigh

                        # Fix numerical issues 
                        g_vals[850:] = 1.0
                        r_vals[850:] = 1.0
                        # Mean pool from all processes
                        g_vals = accelerator.reduce(g_vals, reduction="mean")
                        r_vals = accelerator.reduce(r_vals, reduction="mean")

                        # Set target
                        k_vals = (args.amp*r_vals).clamp_min(1e-2)


                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)


            if accelerator.sync_gradients:
                update_ema(ema, model, decay=args.ema_decay)
                global_idx += 1
                progress_bar.update(1)

                v = accelerator.gather(velocity_loss.detach()).mean().item()
                ema_velocity_loss = log_ema_decay*ema_velocity_loss + (1-log_ema_decay)*v if ema_velocity_loss is not None else v
                postfix = {"velocity_loss": ema_velocity_loss}
                if args.use_jacobian:
                    vve = accelerator.gather(velocity_loss_eigen.detach()).mean().item()
                    ema_jac_loss = log_ema_decay*ema_jac_loss + (1-log_ema_decay)*vve if ema_jac_loss is not None else vve
                    postfix["jac_loss"] = ema_jac_loss

                progress_bar.set_postfix(postfix)

                ## Debug ##
                if global_idx % 1000 == 0:
                    accelerator.wait_for_everyone()
                    if args.use_jacobian:
                        g_log = accelerator.reduce(g_vals.clone(), reduction="mean")
                        accelerator.save(
                            g_log.cpu(),
                            os.path.join(args.save_dir, args.experiment_name, f"g_vals_curr.pt")
                        )

                        r_log = accelerator.reduce(r_vals.clone(), reduction="mean")
                        accelerator.save(
                            r_log.cpu(),
                            os.path.join(args.save_dir, args.experiment_name, f"r_vals_curr.pt")
                        )
                ## ##### ##

                # Save model/ema
                if global_idx % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    if args.use_jacobian:
                        g_log = accelerator.reduce(g_vals.clone(), reduction="mean")
                        r_log = accelerator.reduce(r_vals.clone(), reduction="mean")

                    if accelerator.is_main_process:
                        accelerator.save(
                            accelerator.get_state_dict(model),
                            os.path.join(args.save_dir, args.experiment_name, f"model_{global_idx:06d}.pt")
                        )
                        accelerator.save(
                            ema.state_dict(),
                            os.path.join(args.save_dir, args.experiment_name, f"model_ema_{global_idx:06d}.pt")
                        )
                        if args.use_jacobian:
                            accelerator.save(
                                g_log.cpu(),
                                os.path.join(args.save_dir, args.experiment_name, f"g_vals_{global_idx:06d}.pt")
                            )
                            accelerator.save(
                                r_log.cpu(),
                                os.path.join(args.save_dir, args.experiment_name, f"r_vals_{global_idx:06d}.pt")
                            )

                    accelerator.wait_for_everyone()

            if global_idx >= args.train_iterations:
                break

    # Save final model/ema
    accelerator.wait_for_everyone()
    if args.use_jacobian:
        g_log = accelerator.reduce(g_vals.clone(), reduction="mean")

    if accelerator.is_main_process:
        accelerator.save(
            accelerator.get_state_dict(model),
            os.path.join(args.save_dir, args.experiment_name, f"model_{global_idx:06d}.pt")
        )
        accelerator.save(
            ema.state_dict(),
            os.path.join(args.save_dir, args.experiment_name, f"model_ema_{global_idx:06d}.pt")
        )
        if args.use_jacobian:
            accelerator.save(
                g_log.cpu(),
                os.path.join(args.save_dir, args.experiment_name, f"g_vals_{global_idx:06d}.pt")
            )
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--experiment_name", type=str, default='baseline')
    parser.add_argument("--model_name", type=str, default='SiT-S/2')
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--train_iterations", type=int, default=400_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_jacobian", action="store_true", default=False)
    parser.add_argument("--jacobian_w", type=float, default=1.0)
    parser.add_argument("--amp", type=float, default=5.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--mixed_precision", type=str, default='bf16')
    args = parser.parse_args()

    main(args)