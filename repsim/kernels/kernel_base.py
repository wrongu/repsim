import torch
from typing import Union, Iterable


class Kernel(object):
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

    def string_id(self):
        raise NotImplementedError("Kernel.name must be implemented by a subclass")

    def effective_dim(self, x) -> float:
        raise NotImplementedError("Kernel.effective_dim must be implemented by a subclass")


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
