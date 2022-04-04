### *Function* `compare(
    x: torch.Tensor,
    *,
    type: CompareType = CompareType.INNER_PRODUCT,
    kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor ()`


Compute n by n pairwise distance (or similarity) between all pairs of rows of x.

### Arguments
> - **x** (`None`: `None`): n by d matrix of data.
> - **type** (`None`: `None`): a CompareType enum value - one of (INNER_PRODUCT, ANGLE, DISTANCE,
        SQUARE_DISTANCE)
> - **kernel** (`None`: `None`): a kernels.Kernel instance, or None. Defaults to None, which
        falls back on a Linear kernel

### Returns
    n by n matrix of pairwise comparisons (similarity, distance, or squared
        distance, depending on 'type')

