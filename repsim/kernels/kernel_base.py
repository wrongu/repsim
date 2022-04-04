import torch
from typing import Union, Iterable


def center(k: torch.Tensor) -> torch.Tensor:
    """
    Center features of a kernel by pre- and post-multiplying by the centering
    matrix H.

    In other words, if k_ij is dot(x_i, x_j), the result will be

        dot(x_i - mu_x, x_j - mu_x)

    Arguments:
        k (torch.Tensor): a (n,n) Gram matrix of inner products between xs

    Returns:
        torch.Tensor: the centered matrix

    """
    n = k.size()[0]
    if k.size() != (n, n):
        raise ValueError(
            f"Expected k to be nxn square matrix, but it has size {k.size()}"
        )
    H = (
        torch.eye(n, device=k.device, dtype=k.dtype)
        - torch.ones((n, n), device=k.device, dtype=k.dtype) / n
    )
    return H @ k @ H


class Kernel(object):
    """
    Base class for kernels.

    """

    def __call__(
        self, x: torch.Tensor, y: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            y = x

        if x.size()[0] != y.size()[0]:
            raise ValueError("Mismatch in first dimension of x and y")

        return self._call_impl(x, y)

    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Kernel._call_impl must be implemented by a subclass")

    def __add__(self, other):
        return SumKernel(self, [other])


class SumKernel(Kernel):
    """
    Sum of kernels.

    """

    def __init__(self, kernels: Iterable[Kernel], weights=None):
        """
        Create a new SumKernel.

        Arguments:
            kernels (Iterable[Kernel]): the kernels to sum
            weights (Iterable[float]): the weights to use for each kernel

        """

        super(SumKernel, self).__init__()
        self.kernels = list(kernels)
        self.weights = (
            torch.tensor(weights)
            if weights is not None
            else torch.ones(len(self.kernels))
        )

    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tot = self.weights[0] * self.kernels[0](x, y)
        for w, k in zip(self.weights[1:], self.kernels[1:]):
            tot += k(x, y) * w
        return tot
