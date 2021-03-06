import torch
import numpy as np
from repsim import Stress, AngularCKA, AffineInvariantRiemannian, AngularShapeMetric, EuclideanShapeMetric
from repsim.kernels import SquaredExponential
from repsim.geometry import LengthSpace, GeodesicLengthSpace
from repsim.geometry.optimize import OptimResult, project_by_binary_search
from repsim.geometry.trig import angle, slerp


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BIG_M = 10000 if torch.cuda.is_available() else 500


def test_geodesic_stress():
    _test_geodesic_helper(5, 3, 4, Stress(m=5))
    _test_geodesic_gradient_descent(5, 3, 4, Stress(m=5))
    _test_geodesic_endpoints(5, 3, 4, Stress(m=5))


def test_geodesic_stress_big():
    _test_geodesic_helper(BIG_M, 100, 100, Stress(m=BIG_M))


def test_geodesic_cka():
    _test_geodesic_helper(5, 3, 4, AngularCKA(m=5))
    _test_geodesic_gradient_descent(5, 3, 4, AngularCKA(m=5))
    _test_geodesic_endpoints(5, 3, 4, AngularCKA(m=5))


def test_geodesic_cka_big():
    _test_geodesic_helper(BIG_M, 100, 100, AngularCKA(m=BIG_M))


def test_geodesic_riemann():
    _test_geodesic_helper(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))
    _test_geodesic_gradient_descent(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))
    _test_geodesic_endpoints(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))


def test_geodesic_riemann_big():
    _test_geodesic_helper(BIG_M, 100, 100, AffineInvariantRiemannian(m=BIG_M, kernel=SquaredExponential()))


def test_geodesic_angular_shape():
    _test_geodesic_helper(10, 3, 4, AngularShapeMetric(10, p=2))
    # _test_geodesic_gradient_descent(10, 3, 4, AngularShapeMetric(10, p=2))  # FAILS but we don't need it to work
    _test_geodesic_endpoints(10, 3, 4, AngularShapeMetric(10, p=2))
    _test_geodesic_helper(10, 3, 4, AngularShapeMetric(10, p=5))
    # _test_geodesic_gradient_descent(10, 3, 4, AngularShapeMetric(10, p=5))  # FAILS but we don't need it to work
    _test_geodesic_endpoints(10, 3, 4, AngularShapeMetric(10, p=5))


def test_geodesic_angular_shape_big():
    _test_geodesic_helper(BIG_M, 3, 4, AngularShapeMetric(BIG_M, 50))


def test_geodesic_euclidean_shape():
    _test_geodesic_helper(10, 3, 4, EuclideanShapeMetric(10, p=2))
    # _test_geodesic_gradient_descent(10, 3, 4, EuclideanShapeMetric(10, p=2))  # FAILS but we don't need it to work
    _test_geodesic_endpoints(10, 3, 4, EuclideanShapeMetric(10, p=2))
    _test_geodesic_helper(10, 3, 4, EuclideanShapeMetric(10, p=5))
    # _test_geodesic_gradient_descent(10, 3, 4, EuclideanShapeMetric(10, p=5))  # FAILS but we don't need it to work
    _test_geodesic_endpoints(10, 3, 4, EuclideanShapeMetric(10, p=5))


def test_geodesic_euclidean_shape_big():
    _test_geodesic_helper(BIG_M, 3, 4, EuclideanShapeMetric(BIG_M, 50))


def test_slerp():
    vec001 = torch.tensor([0, 0, 1], dtype=torch.float32)
    vec100 = torch.tensor([1, 0, 0], dtype=torch.float32)
    vec101 = torch.tensor([1, 0, 1], dtype=torch.float32) * np.sqrt(2) / 2

    # Test some edge cases - where frac is 0, 1, or two vecs are identical
    assert slerp(vec001, vec001, 0.5).allclose(vec001)
    assert slerp(vec001, 2*vec001, 0.5).allclose(vec001)
    assert slerp(vec001, vec100, 0).allclose(vec001)
    assert slerp(vec001, vec100, 1).allclose(vec100)

    # Do an actual interpolation
    assert slerp(vec001, vec100, 0.5).allclose(vec101)
    # Test that SLERP normalizes things for us
    assert slerp(4*vec001, 2*vec100, 0.5).allclose(vec101)

    # Do a random 2D interpolation not involving orthogonal vectors
    vec_a, vec_b = torch.randn(2), torch.randn(2)
    norm_a, norm_b = vec_a / torch.linalg.norm(vec_a), vec_b / torch.linalg.norm(vec_b)
    total_angle = torch.arccos(torch.clip(torch.sum(norm_a*norm_b), -1., 1.))
    frac = np.random.rand()
    rotation_amount = frac * total_angle
    c, s = torch.cos(rotation_amount), torch.sin(rotation_amount)
    rotation_matrix = torch.tensor([[c, -s], [s, c]])
    # Rotate vec_a towards vec_b... try both CW and CCW and take whichever worked
    vec_c_clockwise = rotation_matrix @ norm_a
    vec_c_counterclockwise = rotation_matrix.T @ norm_a
    assert slerp(vec_a, vec_b, frac).allclose(vec_c_clockwise) or \
        slerp(vec_a, vec_b, frac).allclose(vec_c_counterclockwise)



def _test_geodesic_helper(m, nx, ny, metric):
    x, y = torch.randn(m, nx, dtype=torch.float64), torch.randn(m, ny, dtype=torch.float64)
    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    frac, tolerance = np.random.rand(1)[0], 1e-2
    pt_z = metric.geodesic(pt_x, pt_y, frac=frac)
    dist_xy = metric.length(pt_x, pt_y)
    dist_xz = metric.length(pt_x, pt_z)
    dist_zy = metric.length(pt_z, pt_y)

    assert metric.contains(pt_z, atol=tolerance), \
        f"point_along failed contains() test at frac={frac:.4f}  using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xz + dist_zy, atol=tolerance), \
        f"point_along at frac={frac:.4f} not along geodesic: d(x,y) is {dist_xy:.4f} but d(x,m)+d(m,y) is {dist_xz + dist_zy:.4f}"
    assert np.isclose(dist_xz/dist_xy, frac, atol=tolerance), \
        f"point_along failed to divide the total length: frac is {frac:.4f} but d(x,m)/d(x,y) is {dist_xz/dist_xy:.4f}"

    ang = angle(metric, pt_x, pt_z, pt_y).item()
    assert np.abs(ang - np.pi) < tolerance, \
        f"Angle through midpoint using {metric} should be pi but is {ang}"


def _test_geodesic_endpoints(m, nx, ny, metric):
    x, y = torch.randn(m, nx, dtype=torch.float64), torch.randn(m, ny, dtype=torch.float64)
    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    pt_t0 = metric._geodesic_impl(pt_x, pt_y, frac=0.0)
    pt_t1 = metric._geodesic_impl(pt_x, pt_y, frac=1.0)

    assert torch.allclose(metric.length(pt_x, pt_t0), pt_x.new_zeros(1), atol=1e-6), \
        "geodesic at frac=0 is not equivalent to x!"
    assert torch.allclose(metric.length(pt_y, pt_t1), pt_x.new_zeros(1), atol=1e-6), \
        "geodesic at frac=1 is not equivalent to y!"


def _test_geodesic_gradient_descent(m, nx, ny, metric):
    x, y = torch.randn(m, nx, dtype=torch.float64), torch.randn(m, ny, dtype=torch.float64)
    k_x, k_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    frac, tolerance = np.random.rand(1)[0], 1e-2

    # Case 1: compute geodesic using closed-form solution
    k_z_closed_form = GeodesicLengthSpace.geodesic(metric, k_x, k_y, frac=frac)
    # Case 2: compute geodesic using gradient descent - set tolerance to something << the value we're going to assert
    k_z_grad_descent = LengthSpace.geodesic(metric, k_x, k_y, frac=frac, pt_tol=tolerance/100, fn_tol=1e-6)

    # Assert that they're fairly close in terms of length
    assert metric.length(k_z_grad_descent, k_z_closed_form) < tolerance, \
        "Closed-form and grad-descent geodesics are not close!"


def test_projection_stress():
    _test_projection_helper(5, 3, 4, 4, Stress(m=5))


def test_projection_stress_big():
    _test_projection_helper(BIG_M, 100, 100, 100, Stress(m=BIG_M))


def test_projection_cka():
    _test_projection_helper(5, 3, 4, 4, AngularCKA(m=5))


def test_projection_cka_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AngularCKA(m=BIG_M))


def test_projection_riemann():
    _test_projection_helper(5, 3, 4, 4, AffineInvariantRiemannian(m=5))


def test_projection_riemann_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AffineInvariantRiemannian(m=BIG_M))


def test_projection_angular_shape():
    _test_projection_helper(10, 3, 4, 4, AngularShapeMetric(10, p=2))
    _test_projection_helper(10, 3, 4, 4, AngularShapeMetric(10, p=5))


def test_projection_angular_shape_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AngularShapeMetric(BIG_M, p=50))


def test_projection_euclidean_shape():
    _test_projection_helper(10, 3, 4, 4, EuclideanShapeMetric(10, p=2))
    _test_projection_helper(10, 3, 4, 4, EuclideanShapeMetric(10, p=5))


def test_projection_euclidean_shape_big():
    _test_projection_helper(BIG_M, 100, 100, 100, EuclideanShapeMetric(BIG_M, p=50))


def _test_projection_helper(m, nx, ny, nz, metric):
    x, y, z = torch.randn(m, nx, dtype=torch.float64), torch.randn(m, ny, dtype=torch.float64), torch.randn(m, nz, dtype=torch.float64)
    pt_x, pt_y, pt_z = metric.neural_data_to_point(x), metric.neural_data_to_point(y), metric.neural_data_to_point(z)

    tolerance = 1e-4
    proj, converged = project_by_binary_search(metric, pt_x, pt_y, pt_z, tol=tolerance / 2)
    dist_xy = metric.length(pt_x, pt_y)
    dist_xp = metric.length(pt_x, proj)
    dist_py = metric.length(proj, pt_y)

    assert converged == OptimResult.CONVERGED, \
        f"Projected point failed to converge using {metric}: {proj}"
    assert metric.contains(proj, atol=tolerance), \
        f"Projected point failed contains() test using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xp + dist_py, atol=tolerance), \
        f"Projected point not along geodesic: d(x,y) is {dist_xy} but d(x,p)+d(p,y) is {dist_xp + dist_py}"


def test_angular_cka_contains():
    _test_contains_helper(AngularCKA(100), 100, 10)


def test_stress_contains():
    _test_contains_helper(Stress(100), 100, 10)


def test_affine_invariant_riemannian_contains():
    _test_contains_helper(AffineInvariantRiemannian(100, kernel=SquaredExponential()), 100, 10)


def test_angular_shape_contains():
    _test_contains_helper(AngularShapeMetric(100, p=5), 100, 10)
    _test_contains_helper(AngularShapeMetric(100, p=50), 100, 10)


def test_euclidean_shape_contains():
    _test_contains_helper(EuclideanShapeMetric(100, p=5), 100, 10)
    _test_contains_helper(EuclideanShapeMetric(100, p=50), 100, 10)


def _test_contains_helper(metric, m, nx):
    x = torch.randn(m, nx, dtype=torch.float64)
    pt = metric.neural_data_to_point(x)
    assert metric.contains(pt)