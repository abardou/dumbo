# Decentralized Message-passing Bayesian Optimization (DuMBO)

## TL;DR

DuMBO is a decentralized, high-dimensional Bayesian optimization (BO) algorithm. It is one of the first BO algorithms able to optimize high-dimensional black-boxes (i) in a reasonable amount of time, (ii) without making unrealistic assumptions on the objective function and (iii) with a state-of-the-art cumulative regret bound.

DuMBO is authored by [Anthony Bardou](https://abardou.github.io), [Patrick Thiran](https://people.epfl.ch/patrick.thiran) and [Thomas Begin](https://perso.ens-lyon.fr/thomas.begin/). It was supported by the [EDIC Doctoral Program](https://www.epfl.ch/education/phd/edic-computer-and-communication-sciences/) of [EPFL](https://www.epfl.ch/en/) and [LabEx MiLyon](https://milyon.universite-lyon.fr/). The code in this repository is based on [BoTorch](https://botorch.org/) [1].

DuMBO will be introduced to the community at the 12th [International Conference on Learning Representations](https://iclr.cc) (Vienna, May 2024).

## Contents

* [Citing this Work](#citing-this-work)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [References](#references)

## Citing this Work

If DuMBO is useful in your research or industrial problems, please cite [the following paper](https://arxiv.org/abs/2305.19838) [2]:

```
Bardou, A., Thiran, P., & Begin, T. (2023). Relaxing the Additivity Constraints in Decentralized No-Regret High-Dimensional Bayesian Optimization. arXiv preprint arXiv:2305.19838.
```

The corresponding BibTeX entry is

```
@article{bardou2023relaxing,
  title={Relaxing the Additivity Constraints in Decentralized No-Regret High-Dimensional Bayesian Optimization},
  author={Bardou, Anthony and Thiran, Patrick and Begin, Thomas},
  journal={arXiv preprint arXiv:2305.19838},
  year={2023}
}
```

The information above will be updated once our paper appears in ICLR'24 proceedings.

## Installation

To use DuMBO, just download the code from this repository and unzip it.

All the required packages are listed in `requirements.txt`. To install them, open your favorite command line and run
```
pip install -r requirements.txt
```

## Quick Start

In this section, we provide all the relevant details to use DuMBO. Note that a minimal working example is provided in `main.py` to illustrate its ease-of-use.

To use DuMBO, just instantiate a `dumbo.optimize.DuMBOOptimizer` object using the following constructor call

```python
from dumbo.optimize import DuMBOOptimizer
import gpytorch

dumbo_optimizer = DuMBOOptimizer(
  intervals, # Objective function domain
  X=None, y=None, # Training dataset (if any)
  n_init_points=2, # Initial uniform sampling of observations
  dmax=None, # MFS
  n_samples_per_iteration=5, # Number of sampled additive decompositions
  precision=0.05, max_it=10, # ADMM stopping criteria
  base_kernel_class=gpytorch.kernels.MaternKernel, # Kernel class for the factors
  base_kernel_args=[2.5] # Arguments for instantiating the kernel class
  n_cores=None, # Number of available cores for parallel computation
)
```

Its arguments are detailed below. Assuming that you want to optimize a black-box $f : \mathcal{D} \subset \mathbb{R}^d \to \mathbb{R}$:

* `intervals` is an array of shape $(d,2)$ providing the infimum and supremum for every dimension of $\mathcal{D}$.
* `X` is an array of shape $(n,d)$ gathering the $n$ queries already observed in the input space (if any).
* `y` is an array of shape $(n,)$ gathering the $n$ queries already observed in the output space (if any).
* `n_init_points` indicates how many observations must be uniformly sampled in $\mathcal{D}$ before using ADMM. Cannot be lower than 2 for normalization purposes. If $n$ is larger than `n_init_points`, ADMM is used immediately.
* `dmax` indicates the Maximal Factor Size (MFS) for the inferred additive decompositions. If `None` is provided, `dmax` is set to $d$.
* `n_samples_per_iteration` indicates the number of additive decompositions inferred at each iteration of DuMBO.
* `precision` is the first stopping criterion for ADMM. The smaller it is, the more iterations are required for ADMM to stop.
* `max_it` is the second stopping criterion for ADMM. It bounds the number of ADMM iterations.
* `base_kernel_class` is the class of the covariance function for each factor of the inferred additive decompositions. It must derive from `gpytorch.kernels.Kernel`.
* `base_kernel_args` is a list of arguments useful for the instantiation of `base_kernel_class`.
* `n_cores` is the number of CPUs available for parallel computation.


Once instantiated, DuMBO can be used in a simple optimization loop:

```python
for _ in range(100):
  # Get the next query in the input space
  next_query = dumbo_optimizer.next_query()
  # Observe the objective function value at that point
  objective_value = my_objective_function(next_query)
  # Add the new observation to the dataset
  dumbo_optimizer.tell(next_query, objective_value)
```

## References

The references listed here are cited with their unique id number within brackets (e.g. [1]) directly in the code or in the README. You can find them formatted with the APA style below:

[1] Balandat, M., Karrer, B., Jiang, D., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. Advances in neural information processing systems, 33, 21524-21538.

[2] Bardou, A., Thiran, P., & Begin, T. (2023). Relaxing the Additivity Constraints in Decentralized No-Regret High-Dimensional Bayesian Optimization. arXiv preprint arXiv:2305.19838.

[3] Gardner, J., Guo, C., Weinberger, K., Garnett, R., & Grosse, R. (2017, April). Discovering and exploiting additive structure for Bayesian optimization. In Artificial Intelligence and Statistics (pp. 1311-1319). PMLR.

[4] Gabay, D., & Mercier, B. (1976). A dual algorithm for the solution of nonlinear variational problems via finite element approximation. Computers & mathematics with applications, 2(1), 17-40.