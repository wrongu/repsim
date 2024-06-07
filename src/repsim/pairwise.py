import torch
from repsim import kernels
from repsim.util import CompareType
from typing import Union


def inner_product(
    x: torch.Tensor, *, kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute (n x n) matrix of inner-products between rows of (n x ?) input 'x'. AKA the Gram
    matrix.

    :param x: n by ? matrix of data.
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on
        kernels.DEFAULT_KERNEL
    :return: n by n matrix of pairwise inner products
    """
    if kernel is None:
        kernel = kernels.DEFAULT_KERNEL
    return kernel(x, x)


def cosine(
    x: torch.Tensor, *, kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute (n x n) matrix of cosine distances (cosine of angles between vectors).

    :param x: n by ? matrix of data.
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on
        kernels.DEFAULT_KERNEL
    :return: n by n matrix of cos(theta) where theta is the angle between vectors xi and xj
    """
    inner = inner_product(x, kernel=kernel)
    vector_lengths = torch.sqrt(torch.diag(inner))
    product_of_vector_lengths = vector_lengths[:, None] * vector_lengths[None, :]
    return inner / product_of_vector_lengths


def angle(
    x: torch.Tensor, *, kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute (n x n) matrix of angles between vectors.

    :param x: n by ? matrix of data.
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on
        kernels.DEFAULT_KERNEL
    :return: n by n matrix of angles theta between vectors xi and xj
    """
    cos_theta = cosine(x, kernel=kernel)
    # Include clipping because sometimes we get things like cosine(x,x)=1.000000000001,
    # the arccos() of which is NaN
    return torch.arccos(torch.clip(cos_theta, min=-1.0, max=+1.0))


def squared_euclidean(
    x: torch.Tensor, *, kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute (n x n) matrix of squared Euclidean distances between rows of (n x ?) input 'x'.

    :param x: n by ? matrix of data.
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on
        kernels.DEFAULT_KERNEL
    :return: n by n matrix of pairwise squared Euclidean distances
    """
    inner = inner_product(x, kernel=kernel)
    # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    xx = torch.diag(inner)
    # Clip to zero, otherwise we may get values like -0.000000001 due to numerical imprecision,
    # which become nan inside sqrt() call in euclidean()
    return torch.clip(xx[:, None] + xx[None, :] - 2 * inner, min=0.0, max=None)


def euclidean(
    x: torch.Tensor, *, kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute (n x n) matrix of Euclidean distances between rows of (n x ?) input 'x'.

    :param x: n by ? matrix of data.
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on
        kernels.DEFAULT_KERNEL
    :return: n by n matrix of pairwise Euclidean distances
    """
    return torch.sqrt(squared_euclidean(x, kernel=kernel))


TYPE_TO_METHOD = {
    CompareType.INNER_PRODUCT: inner_product,
    CompareType.ANGLE: angle,
    CompareType.COSINE: cosine,
    CompareType.SQUARE_DISTANCE: squared_euclidean,
    CompareType.DISTANCE: euclidean,
}


def compare(
    x: torch.Tensor,
    *,
    type: CompareType = CompareType.INNER_PRODUCT,
    kernel: Union[None, kernels.Kernel] = None
) -> torch.Tensor:
    """Compute n by n pairwise distance (or similarity) between all pairs of rows of x.

    :param x: n by d matrix of data.
    :param type: a CompareType enum value - one of (INNER_PRODUCT, ANGLE, DISTANCE, SQUARE_DISTANCE)
    :param kernel: a kernels.Kernel instance, or None. Defaults to None, which falls back on a
        Linear kernel
    :return: n by n matrix of pairwise comparisons (similarity, distance, or squared distance,
        depending on 'type')
    """

    method = TYPE_TO_METHOD[type]
    return method(x, kernel=kernel)


__all__ = [
    "inner_product",
    "cosine",
    "angle",
    "squared_euclidean",
    "euclidean",
    "compare",
]
