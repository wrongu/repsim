(Metric) Representational Similarity Analysis in PyTorch
========================================================

This repository provides the `repsim` package for comparing representational similarity in PyTorch.

See [rsatoolbox](https://github.com/rsagroup/rsatoolbox) for a more mature and fully-featured toolbox. In contrast, this
repository
- does everything in PyTorch, so the outputs are in principle differentiable.
- provides kernel-based methods such as CKA.
- provides metric RSA methods of [Williams et al. (2021)](http://arxiv.org/abs/2110.14739) and [Shahbazi et al. (2021)](https://doi.org/10.1016/j.neuroimage.2021.118271).

## Entry point

If `x` and `y` are matrices of data (`torch.Tensor`s specifically), each with `n` rows (where `x[i,:]` and `y[i,:]` 
correspond to the same input), then compare representations in each using
```python
import repsim
dist = repsim.compare(x, y, method='stress')
print("The representational distance between x and y is", dist)
```

## Terminology

- Here, a **neural representation** refers to a `n` by `d` matrix containing the activity of `d` neurons in
response to `n` inputs.
- **Similarity** and **distance** are essentially inverses. Similarity is high when distance is low, and vice versa.
Both are (with few exceptions) non-negative quantities.
- **Pairwise similarity** refers to a `n` by `n` matrix of similarity scores among all pairs of input-items for a given 
neural representation. Likewise, **pairwise distance** is `n` by `n` but contains distances rather than similarities.
- **Representational similarity** is a scalar score that is large when two neural representations are similar to each
other. **Representational distance** is likewise a scalar that is large when two representations are dissimilar.

## Design

There are two core operations of any measure of representational similarity:

1. Computing **pairwise similarity** (or **pairwise distance**). The result is a `n` by `n` Representational Similarity
Matrix (RSM) (or Representational Distance Matrix (RDM)). 
2. Comparing two RSMs (or RDMs) to each other to get a scalar **representational similarity** (or distance) score. 

Step 1 is handled by functions in the `repsim.pairwise` module. See the `repsim.pairwise.compare()` function to get started.

Step 2 is handled by the top-level `repsim.compare()` function.

In some special circumstances, we can take computational shortcuts bypassing Step 1, so most users will not explicitly
call anything inside `repsim.pairwise`.

Applying the kernel trick is central to some of the representational similarity measures we use. The `repsim.kernels`
module contains the kernel logic. By default, `repsim` makes all pairwise comparisons using a Linear kernel - in other
words using the usual definition of the inner-product. This can be overridden specifying a `kernel` keyword argument to
`repsim.pairwise.compare`, or a `kernel_x` and `kernel_y` argument to `repsim.compare`. For example:
```python
import repsim
import repsim.kernels
import repsim.pairwise
from repsim.util import CompareType
import torch

n, d = 10, 3
x = torch.randn(n, d)

# Get pairwise Euclidean distances
rdm_linear = repsim.pairwise.compare(x, type=CompareType.DISTANCE)

# Get pairwise distances in feature space using a laplace kernel with an automatic length-scale
k = repsim.kernels.Laplace(length_scale='auto')
rdm_laplace = repsim.pairwise.compare(x, kernel=k, type=CompareType.DISTANCE)

# Get pairwise similarity using squared exponential kernel with a custom length-scale
k = repsim.kernels.SquaredExponential(length_scale=0.2)
rsm_sqexp_short = repsim.pairwise.compare(x, kernel=k, type=CompareType.INNER_PRODUCT)
```

## Testing and Generating Documentation

To test, run `pytest` from the root directory.

```shell
pytest
```

To generate documentation, run `docshund` from the root directory.

(`pip install docshund`)

```shell
docshund ./repsim
```

This will generate API reference documentation in the `docs/` directory. Note that this does not require importing the package, and can be done without installing dependencies.