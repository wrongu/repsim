(Metric) Representational Similarity Analysis in PyTorch
========================================================

![](https://github.com/wrongu/repsim/actions/workflows/tests.yml/badge.svg)

This repository provides the `repsim` package for comparing representational similarity in PyTorch.

See [rsatoolbox](https://github.com/rsagroup/rsatoolbox) for a more mature and fully-featured toolbox, or 
[netrep](https://github.com/ahwillia/netrep) for a sklearn-like interface for shape metrics. In contrast, this repository
- does everything in PyTorch, so the outputs are in principle differentiable.
- provides kernel-based methods such as CKA and Stress.
- emphasizes metric properties of computing "distance" between representations, inspired by [Williams et al. (2021)](http://arxiv.org/abs/2110.14739) and [Shahbazi et al. (2021)](https://doi.org/10.1016/j.neuroimage.2021.118271).
- In particular, we implemented closed-form 'shortest paths' or geodesics between representations for each metric.

## Entry point

If `x` and `y` are matrices of data (`torch.Tensor`s specifically), each with `m` rows (where `x[i,:]` and `y[i,:]` 
correspond to the same input), then compare representations in each using
```python
import repsim
x = ... # some (m, n_x) matrix of neural data
y = ... # some (m, n_y) matrix of neural data
dist = repsim.compare(x, y, method='angular_cka')
print("The representational distance between x and y is", dist)
```

For more fine-grained control, don't use `repsim.compare`, but explicitly instantiate a metric instead. Each metric
requires first mapping from neural data to some other space. For example, the `AngularCKA` metric converts `(m,n_x)`
size neural data into a `(m,m)` size Gram matrix, then computes distances between Gram matrices. The 
`metric.neural_data_to_point` function accomplishes this. So, we could do something like the following:
```python
from repsim.metrics import AngularCKA
from repsim.kernels import SquaredExponential
# By default, AngularCKA uses a Linear kernel for the Gram matrix, but we can override that here
metric = AngularCKA(m=x.shape[0], kernel=SquaredExponential())
dist = metric.length(metric.neural_data_to_point(x), metric.neural_data_to_point(y))
```

## Terminology

- Here, a **neural representation** refers to a `m` by `n` matrix containing the activity of `n` neurons in
response to `m` inputs.
- **Similarity** and **distance** are essentially inverses. Similarity is high when distance is low, and vice versa.
Both are (with few exceptions) non-negative quantities.
- **Pairwise similarity** refers to a `m` by `m` matrix of similarity scores among all pairs of input-items for a given 
neural representation. Likewise, **pairwise distance** is `m` by `m` but contains distances rather than similarities.
- **Similarity** is a scalar score that is large when two neural representations are similar to each
other. **Distance** is likewise a scalar that is large when two representations are dissimilar.
- When talking about **metrics**, we mean methods for computing distances between neural representations that satisfy four key properties:
  1. Identity, or $d(x,x) = 0$. (Really, we have $d(x,y)=0$ for all "equivalent" $x$ and $y$, e.g. we might want a distance that is invariant to scale)
  2. Symmetry, or $d(x,y) = d(y,x)$. (Note that in the future we may want to support asymmetry)
  3. Triangle Inequality, or $d(x,z) ≤ d(x,y) + d(y,z)$
  4. [Length](https://en.wikipedia.org/wiki/Intrinsic_metric). Intuitively, this means that $d(x,y)$ can be broken up into the sum of segment lengths of a shortest path connecting $x$ to $y$.

## Design

* The `repsim/geometry/` module contains fairly generic code for handling geometry, like computing geodesics and angles
in arbitrary spaces. The key interfaces are defined in `repsim/geometry/length_space.py`.
* The `repsim/metrics/` module is where the primary classes for neural representational distance are defined. They
inherit from (a subclass of) `repsim.geometry.LengthSpace`, so each metric has nice geometric properties.
* The `repsim/kernels/` module, as its name implies, contains classes for computing kernel-ified inner products
