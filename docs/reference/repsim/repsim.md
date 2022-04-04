### *Function* `compare(
    x: torch.Tensor,
    y: torch.Tensor,
    method: Union[RepresentationMetricSpace, str] = "stress",
    **kwargs,
) -> torch.Tensor ()`


Compute n by n pairwise distance (or similarity) between all pairs of rows of x.

### Arguments
> - **x** (`torch.Tensor`: `None`): n by d matrix of data.
> - **y** (`torch.Tensor`: `None`): n by d matrix of data.
> - **str])** (`None`: `None`): a RepresentationMetricSpace
        instance, or a string indicating which metric to use. Defaults to "stress".
> - **kwargs** (`None`: `None`): keyword arguments to pass to the metric.

### Returns
    n by n matrix of pairwise comparisons (similarity, distance, or squared
        distance, depending on 'type')

