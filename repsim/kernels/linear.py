import tensorly as tl
from .kernel_base import Kernel


class Linear(Kernel):
    def _call_impl(self, x: tl.tensor, y: tl.tensor) -> tl.tensor:
        return tl.einsum("nd,md->nm", x, y)
