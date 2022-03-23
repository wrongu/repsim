import torch
from repsim import kernels
from repsim.types import CompareType
from typing import Union


def compare(x: torch.Tensor, *, type: CompareType = CompareType.SIMILARITY, kernel: Union[None, kernels.Kernel] = None) -> torch.Tensor:
    """Compute n by n pairwise distance (or similarity) between all pairs of rows of x.

    :param x: n by d matrix of data.
    :param type: an enum value - one of (CompareType.SIMILARITY, CompareType.DISTANCE, CompareType.SQUARE_DISTANCE)
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls backon a SquaredExponential with automatic length scale.
    :return: n by n matrix of pairwise comparisons (similarity, distance, or squared distance, depending on 'type')
    """

    if kernel is None:
        # Default kernel is Squared Exponential with automatic length scale
        kernel = kernels.SquaredExponential(length_scale='auto')

    sim = kernel(x)

    if type == CompareType.SIMILARITY:
        return sim
    elif type == CompareType.SQUARE_DISTANCE:
        self_sim = torch.diag(sim)
        # Using that (x_i - x_j)*(x_i - x_j) = x_i*x_i + x_j*x_j - 2*x_i*x_j
        return self_sim[:, None] + self_sim[None, :] - 2*sim
    elif type == CompareType.DISTANCE:
        self_sim = torch.diag(sim)
        # See above
        return torch.sqrt(self_sim[:, None] + self_sim[None, :] - 2*sim)


__all__ = ['compare']
