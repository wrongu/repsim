import torch
import numpy as np
from repsim import Stress, GeneralizedShapeMetric, AffineInvariantRiemannian
from repsim.geometry.optimize import OptimResult
from repsim.geometry.trig import angle
from repsim.geometry.geodesic import midpoint


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
    mid, converged = midpoint(k_x, k_y, metric)
    dist_xy = metric.length(k_x, k_y)
    dist_xm = metric.length(k_x, mid)
    dist_my = metric.length(mid, k_y)

    print(F"{metric}: {converged}")

    tolerance=1e-4
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
