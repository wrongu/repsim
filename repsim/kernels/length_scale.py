import torch
from repsim.util import pdist2, upper_triangle


def median_euclidean(x: torch.Tensor, y=None, fraction=1.0):
    if y is None:
        # Compare rows of x to other rows of x
        return fraction * torch.sqrt(torch.median(upper_triangle(pdist2(x, x))))
    else:
        # Compare x to y
        return fraction * torch.sqrt(torch.median(pdist2(x, y)))
