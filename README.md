# Poisson Latent Neural Differential Equations (PLNDE)

#### Code for [Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations](https://timkimd.github.io/papers/Kim_ICML2021_PLNDE.pdf)

## Getting Started
This repository contains demo code for the Poisson Latent Neural Differential Equations (PLNDE) model introduced in this [work](https://timkimd.github.io/papers/Kim_ICML2021_PLNDE.pdf). PLNDE is a low-dimensional nonlinear model that infers latent neural population dynamics using neural ordinary differential equations (neural ODEs), with sensory inputs and Poisson spike train outputs. After installing the relevant Julia packages below, run `plnde_spiral.jl` to train PLNDE on the nonlinear spiral dynamics synthetic dataset in Fig. 2 of our paper above. If you have any questions, please feel free to reach out to tdkim@princeton.edu.

### Requirements
- Julia v1.6.0
- DifferentialEquations v6.16.0
- DiffEqFlux v1.35.1
- DiffEqSensitivity v6.43.1
- Flux v0.12.1
- Optim v1.3.0
- StatsBase v0.33.5
- Distributions v0.24.15
- BSON v0.3.3
- Random

## Citation

```bibtex
@article{kim2021plnde,
 title     = {Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations},
 author    = {Kim, Timothy Doyeon and Luo, Thomas Zhihao and Pillow, Jonathan W. and Brody, Carlos D.}, 
 journal   = {Proceedings of the 38th International Conference on Machine Learning},
 year      = {2021}
}
```
