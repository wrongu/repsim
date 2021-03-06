import torch
from repsim.geometry.hypersphere import HyperSphere
from repsim.geometry.trig import angle
from repsim.stats import SphericalMDS, ManifoldPCA
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
    _test_spherical_mds_helper(2, 2, 10, None, None)
    _test_spherical_mds_helper(2, 2, 10, 2, None)
    _test_spherical_mds_helper(10, 10, 100, 4, 10)
    _test_spherical_mds_helper(10, 2, 10, 2, None)


def _test_spherical_mds_helper(true_d, fit_d, n, n_jobs, max_inner_loop):
    sphere = HyperSphere(dim=true_d)
    points = torch.stack([sphere.project(torch.randn(true_d+1)) for _ in range(n)], dim=0)
    mds = SphericalMDS(dim=fit_d, n_jobs=n_jobs, max_inner_loop=max_inner_loop)
    new_points = mds.fit_transform(points).float()

    if true_d == fit_d:
        assert torch.allclose(*_orthogonal_procrustes(points, new_points), atol=2e-2), \
            "MDS failed to recover the correct points when true_d == fit_d == " + str(true_d)


def test_spherical_pca():
    _test_spherical_pca_helper(2, 2, 10)
    _test_spherical_pca_helper(10, 2, 10)
    _test_spherical_pca_helper(10, 10, 100)


def _test_spherical_pca_helper(true_d, fit_d, n):
    sphere = HyperSphere(dim=true_d)
    points = torch.stack([sphere.project(torch.randn(true_d+1)) for _ in range(n)], dim=0)
    pca = ManifoldPCA(space=sphere, n_components=fit_d)
    coordinates = pca.fit_transform(points).float()

    assert coordinates.shape == (n, fit_d), \
        "Expected size of output of ManifoldPCA.transform to be n by n_components"

    new_points = pca.inverse_transform(coordinates)
    assert new_points.shape == points.shape, \
        "Expected size of output of ManifoldPCA.inverse_transform to be same as original data"

    if true_d == fit_d:
        assert torch.allclose(points, new_points, atol=1e-3), \
            "PCA failed to recover the correct points when true_d == fit_d == " + str(true_d)
