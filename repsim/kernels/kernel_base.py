from typing import Union, Iterable
import tensorly as tl


def center(k: tl.tensor) -> tl.tensor:
    """Center features of a kernel by pre- and post-multiplying by the centering matrix H.

    In other words, if k_ij is dot(x_i, x_j), the result will be dot(x_i - mu_x, x_j - mu_x).

    :param k: a n by n Gram matrix of inner products between xs
    :return: a n by n centered matrix
    """
    n = k.size()[0]
    if k.size() != (n, n):
        raise ValueError(
            f"Expected k to be nxn square matrix, but it has size {k.size()}"
        )
    H = (
        tl.eye(n, device=k.device, dtype=k.dtype)
        - tl.ones((n, n), device=k.device, dtype=k.dtype) / n
    )
    return H @ k @ H


class Kernel(object):
    def __call__(self, x: tl.tensor, y: Union[None, tl.tensor] = None) -> tl.tensor:
        if y is None:
            y = x

        if x.size()[0] != y.size()[0]:
            raise ValueError("Mismatch in first dimension of x and y")

        return self._call_impl(x, y)

    def _call_impl(self, x: tl.tensor, y: tl.tensor) -> tl.tensor:
        raise NotImplementedError("Kernel._call_impl must be implemented by a subclass")


class SumKernel(Kernel):
    def __init__(self, kernels: Iterable[Kernel], weights=None):
        super(SumKernel, self).__init__()
        self.kernels = list(kernels)
        self.weights = (
            tl.tensor(weights) if weights is not None else tl.ones(len(self.kernels))
        )

    def _call_impl(self, x: tl.tensor, y: tl.tensor) -> tl.tensor:
        tot = self.weights[0] * self.kernels[0](x, y)
        for w, k in zip(self.weights[1:], self.kernels[1:]):
            tot += k(x, y) * w
        return tot
