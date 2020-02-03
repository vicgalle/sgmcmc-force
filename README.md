# sgmcmc-force

This notebook contains the samplers from the papers ["Stochastic Gradient MCMC with Repulsive Forces"](https://arxiv.org/abs/1812.00071) and "Accelerating Stochastic Gradient Markov Chain Monte Carlo with Momentum and Repulsive Forces" (to appear soon).

All the samplers are implemented in [jax](https://github.com/google/jax/). You can open the notebooks and then run it in Colab.


### SG-MCMC samplers

The implemented samplers/optimizers are located in ```Samplers_jax.ipynb```. The implemented ones are:

* SVGD (Stein Variational Gradient Descend)

* SGLD+R (Stochastic Gradient Langevin Dynamics plus Repulsion)

* SGD (Stochastic Gradient Descent)

* SGLD (Stochastic Gradient Langevin Dynamics)

* SGDm (SGD plus Momentum)

* SGDm+R (SGD plus Momentum and Repulsion)


All the samplers are vectorized and use ```jit``` for increased performance, with an emphasis on simplicity. The notebook constains a standard Gaussian as the target distribution.

### Gaussian example

See the notebook ```Gaussian_example_jax.ipynb``` for a comparison between SVGD and SGLD+R.


### Citation

If you find this code useful for your research, please consider citing

```
@inproceedings{gallego2018sgmcmc,
  author = {Victor Gallego and David Rios Insua},
  title = {Stochastic Gradient MCMC with Repulsive Forces},
  booktitle = {Bayesian Deep Learning Workshop, Neural Information and Processing Systems (NIPS)},
  year = {2018},
}
```
