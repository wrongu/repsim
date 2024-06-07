import torch
from repsim.util import pdist2, upper_triangle


def median_euclidean(x: torch.Tensor, y=None) -> float:
    if y is None:
        # Compare rows of x to other rows of x
        return torch.sqrt(torch.median(upper_triangle(pdist2(x, x)))).item()
    else:
        # Compare x to y
        return torch.sqrt(torch.median(pdist2(x, y))).item()


def mean_euclidean(x: torch.Tensor, y=None) -> float:
    if y is None:
        # Compare rows of x to other rows of x
        return torch.mean(torch.sqrt(upper_triangle(pdist2(x, x)))).item()
    else:
        # Compare x to y
        return torch.mean(torch.sqrt(pdist2(x, y))).item()


def auto_length_scale(x: torch.Tensor, expression: str) -> float:
    """Evaluate an expression like 'auto' or 'median/2'; these expressions are some simple formula
    applied to the median or mean pairwise distance between points.

    Example valid expressions:
    - 'auto' : this is equivalent to 'median' (backwards compatibility)
    - 'median' : get median pairwise euclidean distance between rows of x
    - 'median/2' : get half the median pairwise distance
    - 'mean' : get mean pairwise euclidean distance
    - 'mean*3' : get mean pairwise euclidean distance and multiply by three
    - etc

    :param x: torch.Tensor where each row is a data point in #columns-dimensional Euclidean space
    :param expression: a string expression that will be eval()ed
    :return: scalar float value for length scale
    """
    if expression == "auto":
        return median_euclidean(x)

    if "median" in expression:
        expression_locals = {"median": median_euclidean(x)}
    elif "mean" in expression:
        expression_locals = {"mean": mean_euclidean(x)}
    else:
        raise ValueError(f"Cannot parse length_scale formula '{expression}'")

    return eval(expression, {}, expression_locals)


__all__ = [
    "median_euclidean",
    "auto_length_scale",
]
