import torch
from .kernel_base import Kernel
from .length_scale import median_euclidean
from repsim.util import pdist2


class SquaredExponential(Kernel):
    def __init__(self, length_scale="auto"):
        super(SquaredExponential, self).__init__()
        self._scale = length_scale

    def set_scale(self, lengh_scale):
        self._scale = lengh_scale

    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if type(self._scale) is str and self._scale == "auto":
            sc = median_euclidean(x)
        else:
            sc = self._scale

        return torch.exp(-pdist2(x, y) / sc**2)

    def string_id(self):
        if type(self._scale) is str:
            scale_str = self._scale
        else:
            scale_str = f"{self._scale:.3f}"
        return f"SqExp[{scale_str}]"

    def effective_dim(self, x) -> float:
        return float("inf")



class Laplace(Kernel):
    def __init__(self, length_scale="auto"):
        super(Laplace, self).__init__()
        self._scale = length_scale

    def set_scale(self, lengh_scale):
        self._scale = lengh_scale

    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if type(self._scale) is str and self._scale == "auto":
            sc = median_euclidean(x)
        else:
            sc = self._scale

        return torch.exp(-torch.sqrt(pdist2(x, y)) / sc)

    def string_id(self):
        if type(self._scale) is str:
            scale_str = self._scale
        else:
            scale_str = f"{self._scale:.3f}"
        return f"Laplace[{scale_str}]"

    def effective_dim(self, x) -> float:
        return float("inf")

