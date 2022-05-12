import torch
import numpy as np
from repsim import Stress, ScaleInvariantStress, AngularCKA, AffineInvariantRiemannian
from repsim.geometry.optimize import OptimResult
from repsim.geometry.trig import angle, slerp
from repsim.geometry.geodesic import point_along, project_along

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BIG_M = 10000 if torch.cuda.is_available() else 500


def test_geodesic_stress():
    _test_geodesic_helper(5, 3, 4, Stress(n=5))


def test_geodesic_stress_big():
    _test_geodesic_helper(BIG_M, 100, 100, Stress(n=BIG_M))


def test_geodesic_scale_invariant_stress():
    _test_geodesic_helper(5, 3, 4, ScaleInvariantStress(n=5))


def test_geodesic_scale_invariant_stress_big():
    _test_geodesic_helper(BIG_M, 100, 100, ScaleInvariantStress(n=BIG_M))


def test_geodesic_cka():
    _test_geodesic_helper(5, 3, 4, AngularCKA(n=5))


def test_geodesic_cka_big():
    _test_geodesic_helper(BIG_M, 100, 100, AngularCKA(n=BIG_M))


def test_geodesic_riemann():
    _test_geodesic_helper(5, 3, 4, AffineInvariantRiemannian(n=5))


def test_geodesic_riemann_big():
    _test_geodesic_helper(BIG_M, 100, 100, AffineInvariantRiemannian(n=BIG_M))


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
    k_x, k_y = metric.to_rdm(x), metric.to_rdm(y)

    assert metric.contains(k_x), \
        f"Manifold {metric} does not contain k_x of type {metric.compare_type}"
    assert metric.contains(k_y), \
        f"Manifold {metric} does not contain k_y of type {metric.compare_type}"

    # Note: we'll insist on computing the midpoint with tolerance/10, then do later checks up to tolerance. This just
    # gives a slight margin.
    frac, tolerance = np.random.rand(1)[0], 1e-2
    vals = point_along(k_x, k_y, metric, frac=frac, pt_tol=tolerance/10, fn_tol=1e-6)
    k_z, converged = vals
    dist_xy = metric.length(k_x, k_y)
    dist_xz = metric.length(k_x, k_z)
    dist_zy = metric.length(k_z, k_y)

    print(F"{metric}: {converged}")

    assert converged in [OptimResult.CONVERGED, OptimResult.NO_OPT_NEEDED], \
        f"point_along failed to converge at frac={frac:.4f} using {metric}: {k_z}"
    assert metric.contains(k_z, atol=tolerance), \
        f"point_along failed contains() test at frac={frac:.4f}  using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xz + dist_zy, atol=tolerance), \
        f"point_along at frac={frac:.4f} not along geodesic: d(x,y) is {dist_xy:.4f} but d(x,m)+d(m,y) is {dist_xz + dist_zy:.4f}"
    assert np.isclose(dist_xz/dist_xy, frac, atol=tolerance), \
        f"point_along failed to divide the total length: frac is {frac:.4f} but d(x,m)/d(x,y) is {dist_xz/dist_xy:.4f}"

    ang = angle(k_x, k_z, k_y, metric).item()
    assert np.abs(ang - np.pi) < tolerance, \
        f"Angle through midpoint using {metric} should be pi but is {ang}"


def test_projection_stress():
    _test_projection_helper(5, 3, 4, 4, Stress(n=5))


def test_projection_stress_big():
    _test_projection_helper(BIG_M, 100, 100, 100, Stress(n=BIG_M))


def test_projection_scale_invariant_stress():
    _test_projection_helper(5, 3, 4, 4, ScaleInvariantStress(n=5))


def test_projection_scale_invariant_stress_big():
    _test_projection_helper(BIG_M, 100, 100, 100, ScaleInvariantStress(n=BIG_M))


def test_projection_cka():
    _test_projection_helper(5, 3, 4, 4, AngularCKA(n=5))


def test_projection_cka_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AngularCKA(n=BIG_M))


def test_projection_riemann():
    _test_projection_helper(5, 3, 4, 4, AffineInvariantRiemannian(n=5))


def test_projection_riemann_big():
    _test_projection_helper(BIG_M, 100, 100, 100, AffineInvariantRiemannian(n=BIG_M))


def _test_projection_helper(m, nx, ny, nz, metric):
    x, y, z = torch.randn(m, nx, dtype=torch.float64), torch.randn(m, ny, dtype=torch.float64), torch.randn(m, nz, dtype=torch.float64)
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
