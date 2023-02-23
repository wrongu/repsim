import torch
import numpy as np
from tests.constants import size_m, size_n, num_repeats, rtol, atol, spherical_atol
from repsim.kernels import Linear, Laplace, SquaredExponential
from repsim.metrics import AngularCKA, AngularShapeMetric, EuclideanShapeMetric, Stress, AffineInvariantRiemannian
import pytest


_list_of_metrics = [
    # Angular CKA with a linear kernel is not affine invariant
    {"metric": AngularCKA(size_m),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Next two tests: assert that scale invariance is due to adaptive length scale in the kernel when using sqexp
    {"metric": AngularCKA(size_m, kernel=SquaredExponential(length_scale='median')),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularCKA(size_m, kernel=SquaredExponential(length_scale=0.5)),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    # Next two tests: Stress with a linear kernel is scale-invariant only if rescale=True is set
    {"metric": Stress(size_m, kernel=Linear()),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    {"metric": Stress(size_m, kernel=Linear(), rescale=True),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Next two tests: Stress with an auto-length-scale kernel is scale invariant
    {"metric": Stress(size_m, kernel=SquaredExponential()),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": Stress(size_m, kernel=Laplace()),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # The 'affine invariant' metric is not, in fact, affine-invariant when using a kernel
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=SquaredExponential()),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Laplace()),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # The 'affine invariant' metric *is* affine-invariant using a linear kernel, but only works if n > size_m
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Linear()),
     "high-rank-data": True,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    # ...If we instead use a Linear kernel with eps regularization, we lose the scale and affine invariance again
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Linear(), eps=0.01),
     "high-rank-data": True,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    # Shape metrics with alpha=0 (full whitening) are invariant to everything...
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=0.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    # ...unless p<n, in which case they will in general lose affine invariance because top-p PCs changed
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n//2, alpha=0.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Shape metrics are also not affine-invariant when doing partial (alpha=0.5) or no (alpha=1.0) whitening
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=0.5),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=1.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # ...repeat the last 4 tests for the AngularShapeMetric
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=0.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    {"metric": AngularShapeMetric(m=size_m, p=size_n//2, alpha=0.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=0.5),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=1.0),
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
]


def _randomly_translate(x):
    return x + torch.randn(1, x.size(1), dtype=x.dtype, device=x.device)


def _randomly_rotate(x):
    q, _ = torch.qr(torch.randn(x.size(1), x.size(1), dtype=x.dtype, device=x.device))
    return x @ q


def _randomly_scale(x):
    return x * np.exp(np.random.randn())


def _randomly_affine(x):
    affine = torch.matrix_exp(torch.randn(x.size(1), x.size(1), dtype=x.dtype, device=x.device) / x.size(1))
    return x @ affine


def _assert_invariant(metric, name, operation, x, y):
    pt_x = metric.neural_data_to_point(x)
    pt_y = metric.neural_data_to_point(y)
    base_distance = metric._length_impl(pt_x, pt_y)

    for repeat in range(num_repeats):
        altered_pt_x = metric.neural_data_to_point(operation(x))
        assert torch.isclose(metric._length_impl(pt_x, altered_pt_x),
                             pt_x.new_zeros((1,)),
                             rtol=rtol,
                             atol=spherical_atol if metric.is_spherical else atol), \
            f"{metric.string_id()} failed {name} invariance: expected d(x, x') = 0"
        assert torch.isclose(metric._length_impl(altered_pt_x, pt_y),
                             base_distance,
                             rtol=rtol,
                             atol=spherical_atol if metric.is_spherical else atol), \
            f"{metric.string_id()} failed {name} invariance: expected d(x', y) = d(x, y)"


def _assert_variant(metric, name, operation, x, y):
    pt_x = metric.neural_data_to_point(x)
    pt_y = metric.neural_data_to_point(y)
    base_distance = metric._length_impl(pt_x, pt_y)

    for repeat in range(num_repeats):
        altered_pt_x = metric.neural_data_to_point(operation(x))
        assert not torch.isclose(metric._length_impl(pt_x, altered_pt_x),
                                 pt_x.new_zeros((1,)),
                                 rtol=rtol,
                                 atol=spherical_atol if metric.is_spherical else atol), \
            f"{metric.string_id()} failed {name} variance: expected d(x, x') != 0"
        assert not torch.isclose(metric._length_impl(altered_pt_x, pt_y),
                                 base_distance,
                                 rtol=rtol,
                                 atol=spherical_atol if metric.is_spherical else atol), \
            f"{metric.string_id()} failed {name} variance: expected d(x', y) != d(x, y)"


@pytest.mark.parametrize("metric,use_high_rank,expect_invariant",
                         [(d["metric"], d["high-rank-data"], d["translation-invariant"]) for d in _list_of_metrics])
def test_translation_invariance(metric, use_high_rank, expect_invariant, data_x, data_y, high_rank_x, high_rank_y):
    if use_high_rank:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if expect_invariant:
        _assert_invariant(metric, "translation", _randomly_translate, x, y)
    else:
        _assert_variant(metric, "translation", _randomly_translate, x, y)


@pytest.mark.parametrize("metric,use_high_rank,expect_invariant",
                         [(d["metric"], d["high-rank-data"], d["rotation-invariant"]) for d in _list_of_metrics])
def test_rotation_invariance(metric, use_high_rank, expect_invariant, data_x, data_y, high_rank_x, high_rank_y):
    if use_high_rank:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if expect_invariant:
        _assert_invariant(metric, "rotation", _randomly_rotate, x, y)
    else:
        _assert_variant(metric, "rotation", _randomly_rotate, x, y)


@pytest.mark.parametrize("metric,use_high_rank,expect_invariant",
                         [(d["metric"], d["high-rank-data"], d["scale-invariant"]) for d in _list_of_metrics])
def test_scale_invariance(metric, use_high_rank, expect_invariant, data_x, data_y, high_rank_x, high_rank_y):
    if use_high_rank:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if expect_invariant:
        _assert_invariant(metric, "scale", _randomly_scale, x, y)
    else:
        _assert_variant(metric, "scale", _randomly_scale, x, y)


@pytest.mark.parametrize("metric,use_high_rank,expect_invariant",
                         [(d["metric"], d["high-rank-data"], d["affine-invariant"]) for d in _list_of_metrics])
def test_affine_invariance(metric, use_high_rank, expect_invariant, data_x, data_y, high_rank_x, high_rank_y):
    if use_high_rank:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if expect_invariant:
        _assert_invariant(metric, "affine", _randomly_affine, x, y)
    else:
        _assert_variant(metric, "affine", _randomly_affine, x, y)
