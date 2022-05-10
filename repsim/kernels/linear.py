import torch
from .kernel_base import Kernel


class Linear(Kernel):
    def _call_impl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("n...,m...->nm", x, y)
