import torch
from repsim import pairwise
from repsim.kernels import Linear, SquaredExponential, Laplace
from repsim.util import CompareType, upper_triangle


def test_pairwise_compare_random_data():
    x = torch.randn(10, 3)
    kernels = [None, Linear(), SquaredExponential(), Laplace()]
    for k in kernels:
        for cmp in CompareType:
            xx = pairwise.compare(x, type=cmp, kernel=k)
            assert not torch.any(
                torch.isnan(xx)
            ), f"NaN value in pairwise comparison using {cmp}!"
            assert torch.allclose(
                xx, xx.T
            ), f"Asymmetric pairwise comparison using {cmp}"


def test_upper_triangle():
    A = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert torch.all(upper_triangle(A) == torch.tensor([1, 2, 5]))
    assert torch.all(upper_triangle(A, offset=0) == torch.tensor([0, 1, 2, 4, 5, 8]))
    assert torch.all(upper_triangle(A, offset=2) == torch.tensor([2]))
