# On the Spectral Properties of Generative Denoiser Jacobians

[[arXiv]](https://arxiv.org/abs/2609.36210)

Code for "On the Spectral Properties of Generative Denoiser Jacobians"

## 2D Mixture-of-Gaussians experiment

The notebook `toy.ipynb` implements the mixture-of-Gaussians experiment shown below:

<img width="4280" height="1355" alt="toy" src="https://github.com/user-attachments/assets/53af531c-1abc-47b2-885a-ca2ec30ab255" />

Using different Jacobian regularization objectives, we can control how the denoiser fits the target distribution:
- **(a)** Baseline -- no regularization
- **(b)** "Square" regularization that limits variation along the orthogonal axes.
- **(c)** Residual regularization that increases the eigenvalues of the learned Jacobian.

We also include an example of _direct_ regularization, where the goal is to maximize all eigenvalues isotropically. This experiment fails as it overshoots.

## ImageNet experiments

In `imagenet/` we include sample code to implement our ImageNet experiments.

First, `compute_eigcets.ipynb` downloads pre-trained [SiT](https://github.com/willisma/SiT) models (SiT-XL+REPA, SiT-XL, SiT-S) and implements the algorithm to find the top-n eigenvectors.
The example image we provide `dog.png` comes from the ImageNet validation set. The eigenvectors around it should look like this:

<img width="8971" height="2609" alt="eigenvectors" src="https://github.com/user-attachments/assets/b70093f2-66ca-45af-aae8-498590fbda70" />

Then, `optimize.py` implements the training algorithm that regularizes the model to increase eigenvalues. We pre-process the ImageNet training set as in [REPA](https://github.com/sihyun-yu/REPA), cropping to 256x256 and pre-extracting VAE features.

## TODO:
- [ ] Include eigenvalue computation script to recreate the eigenvalue statistics figures.
- [ ] Provide trained SiT-S/B models with the proposed regularization.
- [ ] Include the random perturbation regularizer.


## Bibtex

```
@article{graikos2026spectral,
  title={On the spectral properties of generative denoiser Jacobians}, 
  author={Graikos, Alexandros and Jojic, Nebojsa, and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2609.36210},
  year={2026},
}
```
