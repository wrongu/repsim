import torch
import numpy as np
from tests.constants import num_repeats, rtol, atol, spherical_atol


def _randomly_translate(x):
    return x + torch.randn(1, x.size(1), dtype=x.dtype, device=x.device)


def _randomly_rotate(x):
    q, _ = torch.qr(torch.randn(x.size(1), x.size(1), dtype=x.dtype, device=x.device))
    return x @ q


def _randomly_scale(x):
    return x * np.exp(np.random.randn())


def _randomly_affine(x):
    affine = torch.matrix_exp(
        torch.randn(x.size(1), x.size(1), dtype=x.dtype, device=x.device) / x.size(1)
    )
    return x @ affine


def _assert_invariant(metric, name, operation, x, y):
    pt_x = metric.neural_data_to_point(x)
    pt_y = metric.neural_data_to_point(y)
    base_distance = metric._length_impl(pt_x, pt_y)

    for repeat in range(num_repeats):
        altered_pt_x = metric.neural_data_to_point(operation(x))
        length = metric._length_impl(pt_x, altered_pt_x)
        assert torch.isclose(
            length,
            pt_x.new_zeros((1,)),
            rtol=rtol,
            atol=spherical_atol(np.cos(length.item())) if metric.is_spherical else atol,
        ), f"{metric.string_id()} failed {name} invariance: expected d(x, x') = 0"
        length = metric._length_impl(altered_pt_x, pt_y)
        assert torch.isclose(
            length,
            base_distance,
            rtol=rtol,
            atol=spherical_atol(np.cos(length.item())) if metric.is_spherical else atol,
        ), f"{metric.string_id()} failed {name} invariance: expected d(x', y) = d(x, y)"


def _assert_variant(metric, name, operation, x, y):
    pt_x = metric.neural_data_to_point(x)
    pt_y = metric.neural_data_to_point(y)
    base_distance = metric._length_impl(pt_x, pt_y)

    for repeat in range(num_repeats):
        altered_pt_x = metric.neural_data_to_point(operation(x))
        length = metric._length_impl(pt_x, altered_pt_x)
        assert not torch.isclose(
            length,
            pt_x.new_zeros((1,)),
            rtol=rtol,
            atol=spherical_atol(np.cos(length.item())) if metric.is_spherical else atol,
        ), f"{metric.string_id()} failed {name} variance: expected d(x, x') != 0"
        length = metric._length_impl(altered_pt_x, pt_y)
        assert not torch.isclose(
            length,
            base_distance,
            rtol=rtol,
            atol=spherical_atol(np.cos(length.item())) if metric.is_spherical else atol,
        ), f"{metric.string_id()} failed {name} variance: expected d(x', y) != d(x, y)"


def test_translation_invariance(metric, data_x, data_y, high_rank_x, high_rank_y):
    # Note: tests/metrics.py defines all metrics and creates metric.test_high_rank_data and
    # metric.test_translation_invariant
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if metric.test_translation_invariant:
        _assert_invariant(metric, "translation", _randomly_translate, x, y)
    else:
        _assert_variant(metric, "translation", _randomly_translate, x, y)


def test_rotation_invariance(metric, data_x, data_y, high_rank_x, high_rank_y):
    # Note: tests/metrics.py defines all metrics and creates metric.test_high_rank_data and
    # metric.test_rotation_invariant
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if metric.test_rotation_invariant:
        _assert_invariant(metric, "rotation", _randomly_rotate, x, y)
    else:
        _assert_variant(metric, "rotation", _randomly_rotate, x, y)


def test_scale_invariance(metric, data_x, data_y, high_rank_x, high_rank_y):
    # Note: tests/metrics.py defines all metrics and creates metric.test_high_rank_data and
    # metric.test_scale_invariant
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if metric.test_scale_invariant:
        _assert_invariant(metric, "scale", _randomly_scale, x, y)
    else:
        _assert_variant(metric, "scale", _randomly_scale, x, y)


def test_affine_invariance(metric, data_x, data_y, high_rank_x, high_rank_y):
    # Note: tests/metrics.py defines all metrics and creates metric.test_high_rank_data and
    # metric.test_affine_invariant
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    if metric.test_affine_invariant:
        _assert_invariant(metric, "affine", _randomly_affine, x, y)
    else:
        _assert_variant(metric, "affine", _randomly_affine, x, y)
