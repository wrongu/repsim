import torch
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim import compare, Stress, AngularCKA, AffineInvariantRiemannian, ScaleInvariantStress
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


def test_cka_scale_invariant():
    _test_scale_invariant_helper(10, 4, 5, AngularCKA(10))


def test_stress_scale_variant():
    _test_scale_variant_helper(10, 4, 5, Stress(10))


def test_scaleinvariantstress_scale_invariant():
    _test_scale_invariant_helper(10, 4, 5, ScaleInvariantStress(10))


def test_riemannian_scale_invariant():
    _test_scale_invariant_helper(10, 4, 5, AffineInvariantRiemannian(n=10))


def _test_scale_invariant_helper(m, nx, ny, metric):
    x, y = torch.randn(m, nx), torch.randn(m, ny)
    d_x = metric.to_rdm(x)
    d_x_scaled = metric.to_rdm(x * torch.rand(size=(1,)) * 10)
    d_y = metric.to_rdm(y)

    base_distance = metric.length(d_x, d_y)
    x_scale_x_distance = metric.length(d_x, d_x_scaled)
    scale_x_distance = metric.length(d_x_scaled, d_y)

    rtol, atol = 1e-4, 1e-5
    assert torch.isclose(base_distance, scale_x_distance, rtol=rtol, atol=atol), "Failed scale invariance"
    assert torch.isclose(base_distance, scale_x_distance, rtol=rtol, atol=atol), "Failed scale invariance"
    assert torch.isclose(x_scale_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale invariance"


def _test_scale_variant_helper(m, nx, ny, metric):
    x, y = torch.randn(m, nx), torch.randn(m, ny)
    d_x = metric.to_rdm(x)
    d_x_scaled = metric.to_rdm(x * torch.rand(size=(1,)) * 10)
    d_y = metric.to_rdm(y)

    base_distance = metric.length(d_x, d_y)
    x_scale_x_distance = metric.length(d_x, d_x_scaled)
    scale_x_distance = metric.length(d_x_scaled, d_y)

    rtol, atol = 1e-4, 1e-5
    assert not torch.isclose(base_distance, scale_x_distance, rtol=rtol, atol=atol), "Failed scale variance"
    assert not torch.isclose(base_distance, scale_x_distance, rtol=rtol, atol=atol), "Failed scale variance"
    assert not torch.isclose(x_scale_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale variance"
