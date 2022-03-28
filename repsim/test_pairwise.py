import tensorly as tl
from repsim import pairwise
from repsim.kernels import Linear, SquaredExponential, Laplace
from repsim.util import CompareType, upper_triangle
import numpy as np


def test_pairwise_compare_random_data():
    x = tl.randn(10, 3)
    kernels = [None, Linear(), SquaredExponential(), Laplace()]
    for k in kernels:
        for cmp in CompareType:
            xx = pairwise.compare(x, type=cmp, kernel=k)
            assert not np.any(
                np.isnan(xx)
            ), f"NaN value in pairwise comparison using {cmp}!"
            assert np.allclose(xx, xx.T), f"Asymmetric pairwise comparison using {cmp}"


def test_upper_triangle():
    A = tl.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert tl.all(upper_triangle(A) == tl.tensor([1, 2, 5]))
    assert tl.all(upper_triangle(A, offset=0) == tl.tensor([0, 1, 2, 4, 5, 8]))
    assert tl.all(upper_triangle(A, offset=2) == tl.tensor([2]))
