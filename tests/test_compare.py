import torch
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim import (
    compare,
    Stress,
    AngularCKA,
    AffineInvariantRiemannian,
    EuclideanShapeMetric,
    AngularShapeMetric,
)
from tests.constants import size_m, size_n, rtol, atol
import pytest


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("stress", {"kernel": Linear()}),
        ("stress", {"kernel": SquaredExponential()}),
        ("stress", {"kernel": SquaredExponential(length_scale="median/2")}),
        ("stress", {"kernel": Laplace()}),
        ("stress", {"kernel": Laplace(length_scale="median/2")}),
        ("angular_cka", {"kernel": Linear()}),
        ("angular_cka", {"kernel": SquaredExponential()}),
        ("angular_cka", {"kernel": Laplace()}),
        ("angular_cka", {"kernel": Linear(), "use_unbiased_hsic": False}),
        ("angular_cka", {"kernel": SquaredExponential(), "use_unbiased_hsic": False}),
        ("angular_cka", {"kernel": Laplace(), "use_unbiased_hsic": False}),
        ("euclidean_shape_metric", {"p": size_n - 1, "alpha": 0.0}),
        ("euclidean_shape_metric", {"p": size_n - 1, "alpha": 0.5}),
        ("euclidean_shape_metric", {"p": size_n - 1, "alpha": 1.0}),
        ("euclidean_shape_metric", {"p": size_n + 1, "alpha": 0.0}),
        ("euclidean_shape_metric", {"p": size_n + 1, "alpha": 0.5}),
        ("euclidean_shape_metric", {"p": size_n + 1, "alpha": 1.0}),
        ("angular_shape_metric", {"p": size_n - 1, "alpha": 0.0}),
        ("angular_shape_metric", {"p": size_n - 1, "alpha": 0.5}),
        ("angular_shape_metric", {"p": size_n - 1, "alpha": 1.0}),
        ("angular_shape_metric", {"p": size_n + 1, "alpha": 0.0}),
        ("angular_shape_metric", {"p": size_n + 1, "alpha": 0.5}),
        ("angular_shape_metric", {"p": size_n + 1, "alpha": 1.0}),
        ("affine_invariant_riemannian", {"kernel": Linear()}),
        ("affine_invariant_riemannian", {"kernel": SquaredExponential()}),
        ("affine_invariant_riemannian", {"kernel": Laplace()}),
        ("affine_invariant_riemannian", {"kernel": Linear()}),
        ("affine_invariant_riemannian", {"mode": "cov", "p": 15}),
        ("affine_invariant_riemannian", {"mode": "cov", "p": 15}),
        ("affine_invariant_riemannian", {"mode": "cov", "p": 15}),
    ],
)
def test_compare_random_data(name, kwargs, data_x, data_y, data_labels):
    def _test_compare_helper(x, y):
        val_xy = compare(x, y, method=name, **kwargs)
        val_yx = compare(y, x, method=name, **kwargs)
        assert not torch.isnan(val_yx) and not torch.isnan(
            val_xy
        ), f"NaN value in compare() using method {name}"
        assert torch.isclose(
            val_yx, val_xy, rtol=rtol, atol=atol
        ), f"Asymmetry in comparison using method {name}: {val_yx.item()} vs {val_xy.item()}"

    # randn to randn comparison
    _test_compare_helper(data_x, data_y)

    # randn to one-hot-label comparison
    _test_compare_helper(data_x, data_labels)


def test_compare_points(metric, data_x, data_y):
    pt_x, pt_y = metric.neural_data_to_point(data_x), metric.neural_data_to_point(
        data_y
    )
    val_xy = metric.length(pt_x, pt_y)
    val_yx = metric.length(pt_y, pt_x)
    assert not torch.isnan(val_yx) and not torch.isnan(
        val_xy
    ), f"NaN value in compare() using method {metric.string_id()}"
    assert torch.isclose(val_yx, val_xy, rtol=rtol, atol=atol), (
        f"Asymmetry in comparison using method {metric.string_id()}: "
        f"{val_yx.item()} vs {val_xy.item()}"
    )


def test_riemmannian_rank_deficient_inf_distance(data_x, data_y):
    assert (
        size_n < size_m
    ), "Test constants changed?! Rank deficiency test assumes n < m"
    unregularized = AffineInvariantRiemannian(m=size_m, eps=0.0)
    # We expect the unregularized method to fail when x,y have more rows than columns. (Note that
    # in @test_compare_random_data above, specifying method="riemannian" defaults to a
    # regularized version)
    assert torch.isinf(
        compare(
            data_x, data_y, method=unregularized, kernel_x=Linear(), kernel_y=Linear()
        )
    )
