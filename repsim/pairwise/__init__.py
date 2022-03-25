import torch
from repsim import kernels
from repsim.types import CompareType
from typing import Union


def compare(x: torch.Tensor, *, type: CompareType = CompareType.INNER_PRODUCT, kernel: Union[None, kernels.Kernel] = None) -> torch.Tensor:
    """Compute n by n pairwise distance (or similarity) between all pairs of rows of x.

    :param x: n by d matrix of data.
    :param type: a CompareType enum value - one of (INNER_PRODUCT, ANGLE, DISTANCE, SQUARE_DISTANCE)
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on a Linear kernel
    :return: n by n matrix of pairwise comparisons (similarity, distance, or squared distance, depending on 'type')
    """

    if kernel is None:
        # Default kernel is Linear kernel; in other words, all inner products are natively in x-space
        kernel = kernels.Linear()

    inner_product = kernel(x)

    if type == CompareType.INNER_PRODUCT:
        return inner_product
    elif type == CompareType.ANGLE:
        diag = torch.diag(inner_product)
        norm_inner_product = inner_product / torch.sqrt(diag[:, None]) / torch.sqrt(diag[None, :])
        return torch.arccos(norm_inner_product)
    elif type == CompareType.SQUARE_DISTANCE:
        self_sim = torch.diag(inner_product)
        # Using that (x_i - x_j)*(x_i - x_j) = x_i*x_i + x_j*x_j - 2*x_i*x_j
        return self_sim[:, None] + self_sim[None, :] - 2*inner_product
    elif type == CompareType.DISTANCE:
        self_sim = torch.diag(inner_product)
        # See above
        return torch.sqrt(self_sim[:, None] + self_sim[None, :] - 2*inner_product)


__all__ = ['compare']
