import torch
from repsim.geometry.hypersphere import HyperSphere
from repsim.geometry.trig import angle
from repsim.embedding.spherical_mds import SphericalMDS
from repsim.metrics.generalized_shape_metrics import _orthogonal_procrustes


def test_project():
    _test_project_helper(2)
    _test_project_helper(10)


def _test_project_helper(d):
    sphere = HyperSphere(dim=d)
    pt = sphere.project(torch.randn(d+1))
    assert sphere.contains(pt)


def test_geodesic():
    _test_geodesic_helper(2)
    _test_geodesic_helper(10)


def _test_geodesic_helper(d):
    sphere = HyperSphere(dim=d)
    pt_a = sphere.project(torch.randn(d+1))
    pt_b = sphere.project(torch.randn(d+1))
    t = torch.rand(1).item()

    pt_c = sphere.geodesic(pt_a, pt_b, t)

    dist_ab = sphere.length(pt_a, pt_b)
    dist_ac = sphere.length(pt_a, pt_c)
    dist_cb = sphere.length(pt_c, pt_b)

    assert torch.isclose(dist_ab, dist_ac + dist_cb, atol=1e-4)
    assert torch.isclose(dist_ac / dist_ab, torch.tensor([t]), atol=1e-4)

    a = angle(sphere, pt_a, pt_c, pt_b)
    assert torch.isclose(a, torch.tensor([torch.pi]), atol=1e-3)


def test_log_exp():
    _test_log_exp_helper(2)
    _test_log_exp_helper(10)


def _test_log_exp_helper(d):
    sphere = HyperSphere(dim=d)
    pt_a = sphere.project(torch.randn(d+1))
    pt_b = sphere.project(torch.randn(d+1))

    log_ab = sphere.log_map(pt_a, pt_b)
    assert torch.allclose(log_ab, sphere.to_tangent(pt_a, log_ab)), \
        "log map not close to tangent vector!"
    assert torch.isclose(torch.linalg.norm(log_ab), sphere.length(pt_a, pt_b)), \
        "length of log map is not length(a,b)"

    exp_log_ab = sphere.exp_map(pt_a, log_ab)
    assert torch.allclose(pt_b, exp_log_ab), \
        "exp(a, log(a, b)) does not return pt_b"


def test_spherical_mds():
    _test_spherical_mds_helper(2, 2, 10, None)
    _test_spherical_mds_helper(2, 2, 10, 2)
    _test_spherical_mds_helper(10, 10, 100, 4)
    _test_spherical_mds_helper(10, 2, 10, 2)


def _test_spherical_mds_helper(true_d, fit_d, n, n_jobs):
    sphere = HyperSphere(dim=true_d)
    points = torch.stack([sphere.project(torch.randn(true_d+1)) for _ in range(n)], dim=0)
    mds = SphericalMDS(dim=fit_d, n_jobs=n_jobs)
    new_points = mds.fit_transform(points).float()

    print(true_d, fit_d, n, mds.n_iter_)

    if true_d == fit_d:
        assert torch.allclose(*_orthogonal_procrustes(points, new_points), atol=1e-2), \
            "MDS failed to recover the correct points when true_d == fit_d == " + str(true_d)