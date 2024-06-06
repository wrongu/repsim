import torch
from typing import Union, Iterable
import abc


class Kernel(abc.ABC):
    def __call__(
        self, x: torch.Tensor, y: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            y = x

        return self._call_impl(x, y)

    @abc.abstractmethod
    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Class-specific implementation of __call__(x, y). If x is size (mx, ?) and y is size (my,
        ?), then output will be size (mx, my), containing pairwise kernel evaluations (inner
        products in the RKHS) for each pair of x and y.

        :param x: tensor of size (mx, ?)
        :param y: tensor of size (my, ?)
        :return: pairwise kernel evaluations of size (mx, my)
        """

    @abc.abstractmethod
    def string_id(self):
        """Get a string identifier for this kernel."""

    @abc.abstractmethod
    def effective_dim(self, x) -> float:
        """Get 'effective' dimensionality of the feature space. This may be data-dependent, or it
        might ignore 'x'.

        :param x: example data point
        :return: dimensionality of phi(x), the feature embedding of x
        """


class SumKernel(Kernel):
    def __init__(self, kernels: Iterable[Kernel], weights=None):
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

    def string_id(self):
        return f"SumKernel[{'+'.join(k.string_id() for k in self.kernels)}]"

    def effective_dim(self, x) -> float:
        return max([k.effective_dim(x) for k in self.kernels])
