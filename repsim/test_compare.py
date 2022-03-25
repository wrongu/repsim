import torch
from repsim import compare
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim.compare_impl import AffineInvariantRiemannian
import pytest


def test_compare_random_data():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    methods = [
        "stress",
        "generalized_shape_metric",
        "spearman",
        "pearson",
        "riemannian",
    ]
    kernels = [Linear(), Laplace(), SquaredExponential()]
    for k in kernels:
        for meth in methods:
            val_xy = compare(x, y, method=meth, kernel_x=k, kernel_y=k)
            val_yx = compare(y, x, method=meth, kernel_x=k, kernel_y=k)
            assert not torch.isnan(val_yx) and not torch.isnan(
                val_xy
            ), f"NaN value in compare() using method {meth} and kernel {k}"
            assert torch.isclose(
                val_yx, val_xy, rtol=1e-3
            ), f"Asymmetry in comparison using method {meth} and kernel {k}: {val_yx.item()} vs {val_xy.item()}"


def test_riemmannian_rank_deficient():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    unregularized = AffineInvariantRiemannian(shrinkage=0.0)
    # We expect the unregularized method to fail when x,y have more rows than columns.
    # (Note that in @test_compare_random_data above, specifying method="riemannian" defaults to a regularized version)
    with pytest.raises(ValueError):
        compare(x, y, method=unregularized, kernel_x=Linear(), kernel_y=Linear())
