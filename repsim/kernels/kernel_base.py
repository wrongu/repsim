import torch
from typing import Union, Iterable
import abc


class Kernel(abc.ABC):
    def __call__(
        self, x: torch.Tensor, y: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            y = x

        if x.size()[0] != y.size()[0]:
            raise ValueError("Mismatch in first dimension of x and y")

        return self._call_impl(x, y)

    @abc.abstractmethod
    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def string_id(self):
        pass

    @abc.abstractmethod
    def effective_dim(self, x) -> float:
        pass


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
