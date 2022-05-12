import torch
from repsim import pairwise
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim import compare, Stress, AngularCKA, AffineInvariantRiemannian, CompareType
import pytest


def test_compare_random_data():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    methods = [
        "stress",
        "scale_invariant_stress",
        "angular_cka",
        "riemannian",
    ]
    kernels = [Linear(), Laplace(), SquaredExponential()]
    for k in kernels:
        for meth in methods:
            val_xy = compare(x, y, method=meth, kernel=k)
            val_yx = compare(y, x, method=meth, kernel=k)
            assert not torch.isnan(val_yx) and not torch.isnan(
                val_xy
            ), f"NaN value in compare() using method {meth} and kernel {k}"
            assert torch.isclose(
                val_yx, val_xy, rtol=1e-3
            ), f"Asymmetry in comparison using method {meth} and kernel {k}: {val_yx.item()} vs {val_xy.item()}"


def test_compare_rdms_directly():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    kernel = SquaredExponential()
    k_x, k_y = kernel(x), kernel(y)
    for method in [Stress(n=10), AngularCKA(n=10), AffineInvariantRiemannian(n=10)]:
            val_xy = method.length(k_x, k_y)
            val_yx = method.length(k_y, k_x)
            assert not torch.isnan(val_yx) and not torch.isnan(
                val_xy
            ), f"NaN value in compare() using method {method}"
            assert torch.isclose(
                val_yx, val_xy, rtol=1e-3
            ), f"Asymmetry in comparison using method {method}: {val_yx.item()} vs {val_xy.item()}"


def test_riemmannian_rank_deficient():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    unregularized = AffineInvariantRiemannian(n=10, shrinkage=0.0)
    # We expect the unregularized method to fail when x,y have more rows than columns.
    # (Note that in @test_compare_random_data above, specifying method="riemannian" defaults to a regularized version)
    with pytest.raises(ValueError):
        compare(x, y, method=unregularized, kernel_x=Linear(), kernel_y=Linear())


def test_scale_invariance():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    d_x, d_y = pairwise.compare(x, type=CompareType.DISTANCE), pairwise.compare(y, type=CompareType.DISTANCE)

    base_distance = compare(d_x, d_y, method="scale_invariant_stress")
    scale_x_distance = compare(d_x*2, d_y, method="scale_invariant_stress")
    scale_y_distance = compare(d_x, d_y/2, method="scale_invariant_stress")

    assert torch.isclose(base_distance, scale_x_distance), "Failed scale invariance"
    assert torch.isclose(base_distance, scale_y_distance), "Failed scale invariance"
    assert torch.isclose(scale_x_distance, scale_y_distance), "Failed scale invariance"

    base_distance = compare(d_x, d_y, method="stress")
    scale_x_distance = compare(d_x*2, d_y, method="stress")
    scale_y_distance = compare(d_x, d_y/2, method="stress")

    assert not torch.isclose(base_distance, scale_x_distance), "Original stress should not be scale invariant"
    assert not torch.isclose(base_distance, scale_y_distance), "Original stress should not be scale invariant"
    assert not torch.isclose(scale_x_distance, scale_y_distance), "Original stress should not be scale invariant"
