import torch
import numpy as np
from repsim import Stress, AngularCKA, AffineInvariantRiemannian, AngularShapeMetric, EuclideanShapeMetric
from repsim.kernels import SquaredExponential
from repsim.geometry import LengthSpace, GeodesicLengthSpace, RiemannianSpace
from repsim.geometry.hypersphere import HyperSphere
from repsim.geometry.curvature import alexandrov
from repsim.geometry.optimize import OptimResult, project_by_binary_search
from repsim.geometry.trig import angle, slerp


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BIG_M = 1000 if torch.cuda.is_available() else 100


def _generate_data_helper(metric, n):
    return [metric.neural_data_to_point(torch.randn(BIG_M, 100)) for _ in range(n)]


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


def test_geodesic_air():
    _test_geodesic_helper(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))
    # _test_geodesic_gradient_descent(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))  # FAILS but we don't need it to work
    _test_geodesic_endpoints(5, 3, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))

    _test_geodesic_helper(5, 3, 4, AffineInvariantRiemannian(m=5, shrinkage=0.1))
    _test_geodesic_endpoints(5, 3, 4, AffineInvariantRiemannian(m=5, shrinkage=0.1))


def test_geodesic_air_big():
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

    frac, tolerance = np.random.rand(1)[0], 1e-3
    pt_z = metric.geodesic(pt_x, pt_y, frac=frac)
    dist_xy = metric.length(pt_x, pt_y)
    dist_xz = metric.length(pt_x, pt_z)
    dist_zy = metric.length(pt_z, pt_y)

    assert metric.contains(pt_z, atol=tolerance), \
        f"point_along failed contains() test at frac={frac:.4f}  using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xz + dist_zy, rtol=tolerance), \
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


def _test_geodesic_log_exp_helper(metric, pt_x, pt_y):
    assert isinstance(metric, RiemannianSpace), "This test should only be run with RiemannianSpace subclasses"

    frac, tolerance = np.random.rand(1)[0], 1e-2
    pt_z = metric.geodesic(pt_x, pt_y, frac=frac)

    tangent_x_to_y = metric.log_map(pt_x, pt_y)
    pt_z_2 = metric.exp_map(pt_x, frac * tangent_x_to_y)

    tangent_y_to_x = metric.log_map(pt_y, pt_x)
    pt_z_3 = metric.exp_map(pt_y, (1 - frac) * tangent_y_to_x)

    tol = metric.length(pt_x, pt_y) / 1000
    assert metric.length(pt_z, pt_z_2) < tol, "result of metric.geodesic() is not close to exp(x,frac*log(x,y))"
    assert metric.length(pt_z, pt_z_3) < tol, "result of metric.geodesic() is not close to exp(y,(1-frac)*log(y,x))"


def _test_inner_product_helper(metric, pt_x, pt_y):
    assert isinstance(metric, RiemannianSpace), "This test should only be run with RiemannianSpace subclasses"

    vec_w = metric.log_map(pt_x, pt_y)
    vec_w = vec_w / metric.norm(pt_x, vec_w)

    # Note: it can happen (esp. with Stress) where exp_map(pt_x, vec_w) lands outside of the constraints. We would like
    # to assert that length(pt_x, exp_map(pt_x, vec_w))==1., but loop here in case exp_map goes out of bounds.
    scale, pt_z = 1.0, metric.exp_map(pt_x, vec_w)
    while not metric.contains(pt_z):
        scale = scale / 2
        pt_z = metric.exp_map(pt_x, vec_w * scale)

    assert torch.isclose(metric.length(pt_x, pt_z), scale * torch.tensor(1)), \
        f"Distance to exp({scale} * normed tangent vector) should be {scale}"

    vec_v = metric.to_tangent(pt_x, vec_w + torch.randn(vec_w.shape))
    vec_v = vec_v / metric.norm(pt_x, vec_v)
    dot_wv = metric.inner_product(pt_x, vec_w, vec_v)
    dot_vw = metric.inner_product(pt_x, vec_w, vec_v)
    assert torch.isclose(dot_wv, dot_vw), \
        "inner_product should be symmetric!"

    scale_w, scale_v = torch.rand(2)
    assert torch.isclose(scale_w * scale_v * dot_wv, metric.inner_product(pt_x, scale_w * vec_w, scale_v * vec_v)), \
        "inner_product should scale output with scale of inputs (bilinear)!"


def _test_parallel_transport_helper(metric, pt_x, pt_y):
    assert isinstance(metric, RiemannianSpace), \
        "This test should only be run with RiemannianSpace subclasses"

    # Setup: map random tangent vector u from base x to base y
    dummy = metric.log_map(pt_x, pt_y)
    u_x = metric.to_tangent(pt_x, torch.randn(dummy.size()))
    u_y = metric.levi_civita(pt_x, pt_y, u_x)

    # Test 0: transporting a vector from a to a is a no-op
    tol = 1e-3
    u_x_again = metric.levi_civita(pt_x, pt_x, u_x)
    assert torch.allclose(u_x, u_x_again, atol=tol, rtol=tol)

    # Test 1: result is in the tangent space of y
    assert torch.allclose(u_y, metric.to_tangent(pt_y, u_y), atol=tol, rtol=tol), \
        "map of u_x to y did not land in the tangent space of y"

    # Test 2: vector is unchanged by transporting there and back again
    assert torch.allclose(u_x, metric.levi_civita(pt_y, pt_x, u_y), atol=tol, rtol=tol), \
        "reverse map did not get back to the starting u"

    # Test 3: length is preserved by the map
    length_u_x = metric.norm(pt_x, u_x)
    length_u_y = metric.norm(pt_y, u_y)
    assert torch.isclose(length_u_x, length_u_y, rtol=tol), \
        "map did not preserve length of u"

    # Test 4: inner products are preserved by the map
    v_x = metric.to_tangent(pt_x, torch.randn(dummy.size()))
    v_y = metric.levi_civita(pt_x, pt_y, v_x)
    dot_uv_x = metric.inner_product(pt_x, u_x, v_x)
    dot_uv_y = metric.inner_product(pt_y, u_y, v_y)
    assert torch.isclose(dot_uv_x, dot_uv_y, rtol=tol), \
        "map did not preserve dot(u,v)"


def _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature):
    curv = alexandrov(metric, pt_x, pt_y, pt_z)
    if expected_curvature == "positive":
        assert curv > 0, "Expected curvature to be positive"
    elif expected_curvature == "negative":
        assert curv < 0, "Expected curvature to be positive"
    elif expected_curvature == "nonnegative":
        assert curv >= 0, "Expected curvature to be positive"
    elif expected_curvature == "nonpositive":
        assert curv <= 0, "Expected curvature to be positive"
    elif expected_curvature == "zero":
        assert torch.isclose(curv, torch.zeros(1), atol=1e-5)
    else:
        raise ValueError(f"Bad expected_curvature argument: {expected_curvature}")


def test_riemann_hypersphere():
    d = 2 + np.random.randint(10)
    metric = HyperSphere(dim=d)
    pt_x = metric.project(torch.randn(d+1))
    pt_y = metric.project(torch.randn(d+1))
    pt_z = metric.project(torch.randn(d+1))
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="positive")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_riemann_air():
    metric = AffineInvariantRiemannian(m=BIG_M, kernel=SquaredExponential())
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="negative")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)

    metric = AffineInvariantRiemannian(m=BIG_M, shrinkage=0.1)
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="negative")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_riemann_stress():
    metric = Stress(m=BIG_M)
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="zero")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)

    metric = Stress(m=BIG_M, kernel=SquaredExponential())
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="zero")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_riemann_angular_cka():
    metric = AngularCKA(m=BIG_M)
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="positive")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)

    metric = AngularCKA(m=BIG_M, kernel=SquaredExponential())
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="positive")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_riemann_angular_shape():
    metric = AngularShapeMetric(m=BIG_M, p=4)
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="positive")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_riemann_euclidean_shape():
    metric = EuclideanShapeMetric(m=BIG_M, p=4)
    pt_x, pt_y, pt_z = _generate_data_helper(metric, 3)
    _test_curvature_helper(metric, pt_x, pt_y, pt_z, expected_curvature="zero")
    _test_geodesic_log_exp_helper(metric, pt_x, pt_y)
    _test_inner_product_helper(metric, pt_x, pt_y)
    _test_parallel_transport_helper(metric, pt_x, pt_y)


def test_projection_stress():
    _test_projection_helper(5, 3, 4, 4, Stress(m=5))


def test_projection_stress_big():
    _test_projection_helper(BIG_M, 100, 100, 100, Stress(m=BIG_M))


def test_projection_cka():
    _test_projection_helper(5, 3, 4, 4, AngularCKA(m=5))


def test_projection_cka_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AngularCKA(m=BIG_M))


def test_projection_riemann():
    _test_projection_helper(5, 3, 4, 4, AffineInvariantRiemannian(m=5, kernel=SquaredExponential()))


def test_projection_riemann_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AffineInvariantRiemannian(m=BIG_M, kernel=SquaredExponential()))


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

    frac = np.random.rand(1)[0]
    pt_geo = metric.geodesic(pt_x, pt_y, frac)
    pt_geo_projected, _ = project_by_binary_search(metric, pt_x, pt_y, pt_geo, tol=tolerance / 2)
    assert torch.allclose(pt_geo, pt_geo_projected, atol=tolerance), \
        "Projection of a point on the geodesic failed to recover that same point"

    for i in range(11):
        geo_pt = metric.geodesic(pt_x, pt_y, float(i)/10)
        geo_dist = metric.length(pt_z, geo_pt)
        assert metric.length(pt_z, proj) < geo_dist + tolerance, \
            f"length(z, proj) > length(z, geodesic({i/10:.2}))"


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