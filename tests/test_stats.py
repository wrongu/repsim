import torch
import numpy as np
from repsim.geometry.hypersphere import HyperSphere
from repsim.geometry.trig import angle
from repsim.stats import SphericalMDS, ManifoldPCA
from repsim.metrics.generalized_shape_metrics import _orthogonal_procrustes
from repsim import AngularCKA, AngularShapeMetric
from repsim.util import pdist2
from tests.constants import size_m, size_n
import pytest


@pytest.fixture(params=[2, 5, 10])
def sphere_dim(request):
    return request.param


def test_project(sphere_dim):
    sphere = HyperSphere(dim=sphere_dim)
    pt = sphere.project(torch.randn(sphere_dim + 1))
    assert sphere.contains(pt)


def test_hypersphere_geodesic(sphere_dim):
    sphere = HyperSphere(dim=sphere_dim)
    pt_a = sphere.project(torch.randn(sphere_dim + 1))
    pt_b = sphere.project(torch.randn(sphere_dim + 1))
    t = torch.rand(1).item()

    pt_c = sphere.geodesic(pt_a, pt_b, t)

    dist_ab = sphere.length(pt_a, pt_b)
    dist_ac = sphere.length(pt_a, pt_c)
    dist_cb = sphere.length(pt_c, pt_b)

    assert torch.isclose(dist_ab, dist_ac + dist_cb, atol=1e-4)
    assert torch.isclose(dist_ac / dist_ab, torch.tensor([t]), atol=1e-4)

    a = angle(sphere, pt_a, pt_c, pt_b)
    assert torch.isclose(a, torch.tensor([np.pi]), atol=1e-3)


def test_hypersphere_log_exp_maps(sphere_dim):
    sphere = HyperSphere(dim=sphere_dim)
    pt_a = sphere.project(torch.randn(sphere_dim + 1))
    pt_b = sphere.project(torch.randn(sphere_dim + 1))

    log_ab = sphere.log_map(pt_a, pt_b)
    assert torch.allclose(
        log_ab, sphere.to_tangent(pt_a, log_ab)
    ), "log map not close to tangent vector!"
    assert torch.isclose(
        torch.linalg.norm(log_ab), sphere.length(pt_a, pt_b)
    ), "length of log map is not length(a,b)"

    exp_log_ab = sphere.exp_map(pt_a, log_ab)
    assert torch.allclose(pt_b, exp_log_ab), "exp(a, log(a, b)) does not return pt_b"


@pytest.mark.parametrize(
    "true_d, fit_d, n, n_jobs, max_inner_loop",
    [
        (2, 2, 10, None, None),
        (2, 2, 10, 2, None),
        (10, 10, 100, 4, 10),
        (10, 2, 10, 2, None),
    ],
)
def test_spherical_mds(true_d, fit_d, n, n_jobs, max_inner_loop):
    sphere = HyperSphere(dim=true_d)
    points = torch.stack(
        [sphere.project(torch.randn(true_d + 1)) for _ in range(n)], dim=0
    )
    mds = SphericalMDS(dim=fit_d, n_jobs=n_jobs, max_inner_loop=max_inner_loop)
    new_points = mds.fit_transform(points).float()

    if true_d == fit_d:
        _, recovered_points = _orthogonal_procrustes(points, new_points, anchor="a")
        avg_dist = torch.mean(
            torch.sqrt(torch.sum((points - recovered_points) ** 2, dim=1))
            / np.sqrt(fit_d)
        )
        assert (
            avg_dist < 1e-2
        ), "MDS failed to recover the correct points when true_d == fit_d == " + str(
            true_d
        )


def test_procrustes():
    tolerance = 1e-6
    a, b = torch.randn(10, 4), torch.randn(10, 4)
    new_a, new_b = _orthogonal_procrustes(a, b, anchor="middle")
    assert (
        new_a.shape == a.shape and new_b.shape == b.shape
    ), "Procrustes must not change dimensions of inputs"
    dist_ab = pdist2(a, b).diag().sqrt().mean()
    dist_new_ab = pdist2(new_a, new_b).diag().sqrt().mean()
    assert dist_new_ab < dist_ab, "Procrustes alignment should make dist(a,b) smaller"
    assert torch.allclose(
        pdist2(a, a), pdist2(new_a, new_a), atol=tolerance
    ), "Procrustes should preserved pairwise dist a to a"
    assert torch.allclose(
        pdist2(b, b), pdist2(new_b, new_b), atol=tolerance
    ), "Procrustes should preserved pairwise dist b to b"

    new_a, new_b = _orthogonal_procrustes(a, b, anchor="a")
    assert (
        new_a.shape == a.shape and new_b.shape == b.shape
    ), "Procrustes must not change dimensions of inputs"
    assert torch.allclose(
        a, new_a, atol=tolerance
    ), "procrustes with anchor='a' should leave a unchanged"
    dist_new_new_ab = pdist2(new_a, new_b).diag().sqrt().mean()
    assert torch.isclose(
        dist_new_new_ab, dist_new_ab, atol=tolerance
    ), "procrustes with anchor='a' dot product is different"
    assert torch.allclose(
        pdist2(a, a), pdist2(new_a, new_a), atol=tolerance
    ), "Procrustes should preserved pairwise dist a to a"
    assert torch.allclose(
        pdist2(b, b), pdist2(new_b, new_b), atol=tolerance
    ), "Procrustes should preserved pairwise dist b to b"

    new_a, new_b = _orthogonal_procrustes(a, b, anchor="b")
    assert (
        new_a.shape == a.shape and new_b.shape == b.shape
    ), "Procrustes must not change dimensions of inputs"
    assert torch.allclose(
        b, new_b, atol=tolerance
    ), "procrustes with anchor='b' should leave b unchanged"
    dist_new_new_ab = pdist2(new_a, new_b).diag().sqrt().mean()
    assert torch.isclose(
        dist_new_new_ab, dist_new_ab, atol=tolerance
    ), "procrustes with anchor='b' dot product is different"
    assert torch.allclose(
        pdist2(a, a), pdist2(new_a, new_a), atol=tolerance
    ), "Procrustes should preserved pairwise dist a to a"
    assert torch.allclose(
        pdist2(b, b), pdist2(new_b, new_b), atol=tolerance
    ), "Procrustes should preserved pairwise dist b to b"


@pytest.mark.parametrize("true_d,fit_d,n", [(2, 2, 10), (10, 2, 10), (5, 5, 50)])
def test_spherical_pca(true_d, fit_d, n):
    sphere = HyperSphere(dim=true_d)
    points = torch.stack(
        [sphere.project(torch.randn(true_d + 1)) for _ in range(n)], dim=0
    )
    pca = ManifoldPCA(space=sphere, n_components=fit_d)
    coordinates = pca.fit_transform(points).float()

    assert coordinates.shape == (
        n,
        fit_d,
    ), "Expected size of output of ManifoldPCA.transform to be n by n_components"

    new_points = pca.inverse_transform(coordinates)
    assert (
        new_points.shape == points.shape
    ), "Expected size of output of ManifoldPCA.inverse_transform to be same as original data"

    if true_d == fit_d:
        assert torch.allclose(
            points, new_points, atol=1e-3
        ), "PCA failed to recover the correct points when true_d == fit_d == " + str(
            true_d
        )


@pytest.mark.parametrize("true_d,fit_d,n", [(2, 2, 10), (10, 2, 10), (5, 5, 50)])
def test_spherical_pca_offset_scaled(true_d, fit_d, n):
    sphere = HyperSphere(dim=true_d)
    scales = torch.exp(-torch.arange(true_d + 1).float())
    points = torch.stack(
        [sphere.project(torch.randn(true_d + 1) * scales + 1) for _ in range(n)], dim=0
    )
    pca = ManifoldPCA(space=sphere, n_components=fit_d)
    coordinates = pca.fit_transform(points).float()

    assert coordinates.shape == (
        n,
        fit_d,
    ), "Expected size of output of ManifoldPCA.transform to be n by n_components"

    new_points = pca.inverse_transform(coordinates)
    assert (
        new_points.shape == points.shape
    ), "Expected size of output of ManifoldPCA.inverse_transform to be same as original data"

    if true_d == fit_d:
        assert torch.allclose(
            points, new_points, atol=1e-3
        ), "PCA failed to recover the correct points when true_d == fit_d == " + str(
            true_d
        )


def test_acka_pca(data_x, data_y, data_z):
    sphere = AngularCKA(m=size_m)
    points = [sphere.neural_data_to_point(x) for x in [data_x, data_y, data_z]]
    pca = ManifoldPCA(space=sphere, n_components=1)
    coordinates = pca.fit_transform(points).float()
    assert coordinates.shape == (
        3,
        1,
    ), "Expected size of output of ManifoldPCA.transform to be 3 by 1"


def test_shape_pca(data_x, data_y, data_z):
    sphere = AngularShapeMetric(m=size_m, p=size_n // 2)
    points = [sphere.neural_data_to_point(x) for x in [data_x, data_y, data_z]]
    pca = ManifoldPCA(space=sphere, n_components=1)
    coordinates = pca.fit_transform(points).float()
    assert coordinates.shape == (
        3,
        1,
    ), "Expected size of output of ManifoldPCA.transform to be 3 by 1"
