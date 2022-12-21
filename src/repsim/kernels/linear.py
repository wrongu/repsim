import torch
from repsim.util import prod
from .kernel_base import Kernel


class Linear(Kernel):
    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("n...,m...->nm", x, y)

    def string_id(self):
        return "Linear"

    def effective_dim(self, x):
        return prod(x.shape[1:])
