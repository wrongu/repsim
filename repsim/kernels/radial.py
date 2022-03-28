import tensorly as tl
from .kernel_base import Kernel
from .length_scale import median_euclidean
from repsim.util import pdist2


class SquaredExponential(Kernel):
    def __init__(self, length_scale="auto"):
        super(SquaredExponential, self).__init__()
        self._scale = length_scale

    def set_scale(self, lengh_scale):
        self._scale = lengh_scale

    def _call_impl(self, x: tl.tensor, y: tl.tensor) -> tl.tensor:
        if type(self._scale) is str and self._scale == "auto":
            sc = median_euclidean(x)
        else:
            sc = self._scale

        return tl.exp(-pdist2(x, y) / sc**2)


class Laplace(Kernel):
    def __init__(self, length_scale="auto"):
        super(Laplace, self).__init__()
        self._scale = length_scale

    def set_scale(self, lengh_scale):
        self._scale = lengh_scale

    def _call_impl(self, x: tl.tensor, y: tl.tensor) -> tl.tensor:
        if type(self._scale) is str and self._scale == "auto":
            sc = median_euclidean(x)
        else:
            sc = self._scale

        return tl.exp(-tl.sqrt(pdist2(x, y)) / sc)
