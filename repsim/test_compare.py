import numpy as np
import torch
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim import compare, Stress, AngularCKA, AffineInvariantRiemannian, EuclideanShapeMetric, AngularShapeMetric


def _test_compare_random_data_helper(name, kwargs, m, nx, ny):
    x, y = torch.randn(m, nx), torch.randn(m, ny)
    val_xy = compare(x, y, method=name, **kwargs)
    val_yx = compare(y, x, method=name, **kwargs)
    assert not torch.isnan(val_yx) and not torch.isnan(val_xy), \
        f"NaN value in compare() using method {name}"
    assert torch.isclose(val_yx, val_xy, rtol=1e-3), \
        f"Asymmetry in comparison using method {name}: {val_yx.item()} vs {val_xy.item()}"


def _test_compare_one_hot_helper(name, kwargs, m, nx, ny):
    x, y = torch.zeros(m, nx), torch.zeros(m, ny)
    x[torch.arange(m), torch.randint(0, nx, (m,))] = 1.
    y[torch.arange(m), torch.randint(0, ny, (m,))] = 1.
    val_xy = compare(x, y, method=name, **kwargs)
    val_yx = compare(y, x, method=name, **kwargs)
    assert not torch.isnan(val_yx) and not torch.isnan(val_xy), \
        f"NaN value in compare() using method {name}"
    assert torch.isclose(val_yx, val_xy, rtol=1e-3), \
        f"Asymmetry in comparison using method {name}: {val_yx.item()} vs {val_xy.item()}"


def test_stress_random_data():
    _test_compare_random_data_helper("stress", {"kernel": Linear()}, 100, 10, 20)
    _test_compare_random_data_helper("stress", {"kernel": SquaredExponential()}, 100, 10, 20)
    _test_compare_random_data_helper("stress", {"kernel": Laplace()}, 100, 10, 20)


def test_stress_one_hot_data():
    _test_compare_one_hot_helper("stress", {"kernel": Linear()}, 100, 10, 10)
    _test_compare_one_hot_helper("stress", {"kernel": SquaredExponential()}, 100, 10, 10)
    _test_compare_one_hot_helper("stress", {"kernel": Laplace()}, 100, 10, 10)


def test_angular_cka_random_data():
    _test_compare_random_data_helper("angular_cka", {"kernel": Linear()}, 100, 10, 20)
    _test_compare_random_data_helper("angular_cka", {"kernel": SquaredExponential()}, 100, 10, 20)
    _test_compare_random_data_helper("angular_cka", {"kernel": Laplace()}, 100, 10, 20)


def test_angular_cka_one_hot_data():
    _test_compare_one_hot_helper("angular_cka", {"kernel": Linear()}, 100, 10, 10)
    _test_compare_one_hot_helper("angular_cka", {"kernel": SquaredExponential()}, 100, 10, 10)
    _test_compare_one_hot_helper("angular_cka", {"kernel": Laplace()}, 100, 10, 10)


def test_riemannian_random_data():
    _test_compare_random_data_helper("affine_invariant_riemannian", {"kernel": Linear()}, 100, 10, 20)
    _test_compare_random_data_helper("affine_invariant_riemannian", {"kernel": SquaredExponential()}, 100, 10, 20)
    _test_compare_random_data_helper("affine_invariant_riemannian", {"kernel": Laplace()}, 100, 10, 20)
    # Include a case where nx and ny > m with a Linear kernel
    _test_compare_random_data_helper("affine_invariant_riemannian", {"kernel": Linear()}, 10, 20, 20)


def test_riemannian_one_hot_data():
    _test_compare_one_hot_helper("affine_invariant_riemannian", {"kernel": Linear()}, 100, 10, 10)
    _test_compare_one_hot_helper("affine_invariant_riemannian", {"kernel": SquaredExponential()}, 100, 10, 10)
    _test_compare_one_hot_helper("affine_invariant_riemannian", {"kernel": Laplace()}, 100, 10, 10)
    # Include a case where nx and ny > m with a Linear kernel
    _test_compare_one_hot_helper("affine_invariant_riemannian", {"kernel": Linear()}, 10, 20, 20)


def test_euclidean_shape_metric_random_data():
    # Test each combination of (small/big p) x alpha x
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 2, "alpha": 0.0}, 100, 10, 20)
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 2, "alpha": 0.5}, 100, 10, 20)
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 2, "alpha": 1.0}, 100, 10, 20)
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 5, "alpha": 0.0}, 100, 10, 20)
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 5, "alpha": 0.5}, 100, 10, 20)
    _test_compare_random_data_helper("euclidean_shape_metric", {"p": 5, "alpha": 1.0}, 100, 10, 20)


def test_euclidean_shape_metric_one_hot_data():
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 2, "alpha": 0.0}, 100, 10, 20)
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 2, "alpha": 0.5}, 100, 10, 20)
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 2, "alpha": 1.0}, 100, 10, 20)
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 5, "alpha": 0.0}, 100, 10, 20)
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 5, "alpha": 0.5}, 100, 10, 20)
    _test_compare_one_hot_helper("euclidean_shape_metric", {"p": 5, "alpha": 1.0}, 100, 10, 20)


def test_angular_shape_metric_random_data():
    # Test each combination of (small/big p) x alpha x
    _test_compare_random_data_helper("angular_shape_metric", {"p": 2, "alpha": 0.0}, 100, 10, 20)
    _test_compare_random_data_helper("angular_shape_metric", {"p": 2, "alpha": 0.5}, 100, 10, 20)
    _test_compare_random_data_helper("angular_shape_metric", {"p": 2, "alpha": 1.0}, 100, 10, 20)
    _test_compare_random_data_helper("angular_shape_metric", {"p": 5, "alpha": 0.0}, 100, 10, 20)
    _test_compare_random_data_helper("angular_shape_metric", {"p": 5, "alpha": 0.5}, 100, 10, 20)
    _test_compare_random_data_helper("angular_shape_metric", {"p": 5, "alpha": 1.0}, 100, 10, 20)


def test_angular_shape_metric_one_hot_data():
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 2, "alpha": 0.0}, 100, 10, 20)
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 2, "alpha": 0.5}, 100, 10, 20)
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 2, "alpha": 1.0}, 100, 10, 20)
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 5, "alpha": 0.0}, 100, 10, 20)
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 5, "alpha": 0.5}, 100, 10, 20)
    _test_compare_one_hot_helper("angular_shape_metric", {"p": 5, "alpha": 1.0}, 100, 10, 20)


def _test_compare_points_helper(metric, m, nx, ny):
    x, y = torch.randn(m, nx), torch.randn(m, ny)
    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)
    val_xy = metric.length(pt_x, pt_y)
    val_yx = metric.length(pt_y, pt_x)
    assert not torch.isnan(val_yx) and not torch.isnan(val_xy), \
        f"NaN value in compare() using method {metric.string_id()}"
    assert torch.isclose(val_yx, val_xy, rtol=1e-3), \
        f"Asymmetry in comparison using method {metric.string_id()}: {val_yx.item()} vs {val_xy.item()}"


def test_stress_points():
    _test_compare_points_helper(Stress(100, kernel=SquaredExponential()), 100, 10, 20)


def test_angular_cka_points():
    _test_compare_points_helper(AngularCKA(100, kernel=SquaredExponential()), 100, 10, 20)


def test_riemannian_points():
    _test_compare_points_helper(AffineInvariantRiemannian(100, kernel=SquaredExponential()), 100, 10, 20)


def test_angular_shape_metric_points():
    _test_compare_points_helper(AngularShapeMetric(100, p=2), 100, 10, 20)
    _test_compare_points_helper(AngularShapeMetric(100, p=5), 100, 10, 20)


def test_euclidean_shape_metric_points():
    _test_compare_points_helper(EuclideanShapeMetric(100, p=2), 100, 10, 20)
    _test_compare_points_helper(EuclideanShapeMetric(100, p=5), 100, 10, 20)


def test_riemmannian_rank_deficient_inf_distance():
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    unregularized = AffineInvariantRiemannian(m=10, shrinkage=0.0)
    # We expect the unregularized method to fail when x,y have more rows than columns.
    # (Note that in @test_compare_random_data above, specifying method="riemannian" defaults to a regularized version)
    assert torch.isinf(compare(x, y, method=unregularized, kernel_x=Linear(), kernel_y=Linear()))


def test_cka_scale_invariant():
    _test_scale_invariant_helper(100, 10, 20, AngularCKA(100), expect_invariant=True)


def test_stress_scale_variant():
    _test_scale_invariant_helper(100, 10, 20, Stress(100, kernel=Linear()), expect_invariant=False)
    _test_scale_invariant_helper(100, 10, 20, Stress(100, kernel=Linear(), rescale=True), expect_invariant=True)
    _test_scale_invariant_helper(100, 10, 20, Stress(100, kernel=SquaredExponential()), expect_invariant=True)
    _test_scale_invariant_helper(100, 10, 20, Stress(100, kernel=Laplace()), expect_invariant=True)


def test_riemannian_scale_invariant():
    metric = AffineInvariantRiemannian(m=100, kernel=SquaredExponential())
    _test_scale_invariant_helper(100, 10, 20, metric, expect_invariant=True)


def test_shape_metric_scale_alpha_zero():
    metric = EuclideanShapeMetric(m=100, p=4, alpha=0.0)
    _test_scale_invariant_helper(100, 10, 20, metric, expect_invariant=True)
    metric = AngularShapeMetric(m=100, p=4, alpha=0.0)
    _test_scale_invariant_helper(100, 10, 20, metric, expect_invariant=True)


def test_shape_metric_scale_alpha_nonzero():
    # We do *not* expect scale-invariance when using Euclidean metric, as long as alpha>0
    metric = EuclideanShapeMetric(m=100, p=4, alpha=torch.rand(1).item()*0.8+0.1)
    _test_scale_invariant_helper(100, 10, 20, metric, expect_invariant=False)
    # We *do* expect scale-invariance when using Angular metric regardless of alpha
    metric = AngularShapeMetric(m=100, p=4, alpha=torch.rand(1).item()*0.8+0.1)
    _test_scale_invariant_helper(100, 10, 20, metric, expect_invariant=True)


def _test_scale_invariant_helper(m, nx, ny, metric, expect_invariant):
    x, y = torch.randn(m, nx), torch.randn(m, ny)
    d_x = metric.neural_data_to_point(x)
    d_x_halved = metric.neural_data_to_point(x * 0.5)
    d_x_doubled = metric.neural_data_to_point(x * 2.0)
    d_y = metric.neural_data_to_point(y)

    base_distance = metric.length(d_x, d_y)
    x_half_x_distance = metric.length(d_x, d_x_halved)
    half_x_distance = metric.length(d_x_halved, d_y)
    x_double_x_distance = metric.length(d_x, d_x_doubled)
    double_x_distance = metric.length(d_x_doubled, d_y)

    rtol, atol = 1e-3, 1e-4
    if metric.is_spherical:
        atol = np.arccos(1 - atol)

    if expect_invariant:
        assert torch.isclose(base_distance, half_x_distance, rtol=rtol, atol=atol), "Failed scale invariance"
        assert torch.isclose(base_distance, double_x_distance, rtol=rtol, atol=atol), "Failed scale invariance"
        assert torch.isclose(x_half_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale invariance"
        assert torch.isclose(x_double_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale invariance"
    else:
        assert not torch.isclose(base_distance, half_x_distance, rtol=rtol, atol=atol), "Failed scale variance"
        assert not torch.isclose(base_distance, double_x_distance, rtol=rtol, atol=atol), "Failed scale variance"
        assert not torch.isclose(x_half_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale variance"
        assert not torch.isclose(x_double_x_distance, torch.zeros(1), rtol=rtol, atol=atol), "Failed scale variance"
