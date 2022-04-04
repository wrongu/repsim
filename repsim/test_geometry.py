import torch
import numpy as np
from repsim import Stress, GeneralizedShapeMetric, AffineInvariantRiemannian
from repsim.geometry.optimize import OptimResult
from repsim.geometry.trig import angle
from repsim.geometry.geodesic import midpoint, project_along


def test_geodesic_stress():
    x, y = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_geodesic_helper(x, y, Stress(n=5))


def test_geodesic_shape():
    x, y = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_geodesic_helper(x, y, GeneralizedShapeMetric(n=5))


def test_geodesic_riemann():
    x, y = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_geodesic_helper(x, y, AffineInvariantRiemannian(n=5))


def _test_geodesic_helper(x, y, metric):
    k_x, k_y = metric.to_rdm(x), metric.to_rdm(y)

    assert metric.contains(k_x), \
        f"Manifold {metric} does not contain k_x of type {metric.compare_type}"
    assert metric.contains(k_y), \
        f"Manifold {metric} does not contain k_y of type {metric.compare_type}"

    # Note: we'll insist on computing the midpoint with tolerance/2, then do later checks up to tolerance. This just
    # gives a slight margin.
    tolerance = 1e-4
    mid, converged = midpoint(k_x, k_y, metric, pt_tol=tolerance/2, fn_tol=1e-6)
    dist_xy = metric.length(k_x, k_y)
    dist_xm = metric.length(k_x, mid)
    dist_my = metric.length(mid, k_y)

    print(F"{metric}: {converged}")

    assert converged == OptimResult.CONVERGED, \
        f"Midpoint failed to converge using {metric}: {mid}"
    assert metric.contains(mid, atol=tolerance), \
        f"Midpoint failed contains() test using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xm + dist_my, atol=tolerance), \
        f"Midpoint not along geodesic: d(x,y) is {dist_xy} but d(x,m)+d(m,y) is {dist_xm + dist_my}"
    assert np.isclose(dist_xm, dist_my, atol=tolerance), \
        f"Midpoint failed to split the total length into equal parts: d(x,m) is {dist_xm} but d(m,y) is {dist_my}"

    ang = angle(k_x, mid, k_y, metric).item()
    assert np.abs(ang - np.pi) < tolerance, \
        f"Angle through midpoint using {metric} should be pi but is {ang}"


def test_project_stress():
    x, y, z = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_project_helper(x, y, z, Stress(n=5))


def test_project_shape():
    x, y, z = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_project_helper(x, y, z, GeneralizedShapeMetric(n=5))


def test_project_riemann():
    x, y, z = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    _test_project_helper(x, y, z, AffineInvariantRiemannian(n=5))


def _test_project_helper(x, y, z, metric):
    k_x, k_y, k_z = metric.to_rdm(x), metric.to_rdm(y), metric.to_rdm(z)

    tolerance = 1e-4
    proj, converged = project_along(k_x, k_y, k_z, metric, tol=tolerance/2)
    dist_xy = metric.length(k_x, k_y)
    dist_xp = metric.length(k_x, proj)
    dist_py = metric.length(proj, k_y)

    assert converged == OptimResult.CONVERGED, \
        f"Projected point failed to converge using {metric}: {proj}"
    assert metric.contains(proj, atol=tolerance), \
        f"Projected point failed contains() test using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xp + dist_py, atol=tolerance), \
        f"Projected point not along geodesic: d(x,y) is {dist_xy} but d(x,p)+d(p,y) is {dist_xp + dist_py}"
