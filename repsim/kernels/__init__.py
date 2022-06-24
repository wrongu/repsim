from .kernel_base import Kernel
from .radial import SquaredExponential, Laplace
from .linear import Linear
from .kernel_methods import center, hsic, cka

DEFAULT_KERNEL = Linear()
