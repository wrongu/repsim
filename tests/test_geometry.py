import torch
import numpy as np
from repsim.geometry import LengthSpace, GeodesicLengthSpace
from repsim.geometry.curvature import alexandrov
from repsim.geometry.optimize import (
    OptimResult,
    project_by_binary_search,
    project_by_tangent_iteration,
)
from repsim.geometry.trig import angle, slerp
from tests.constants import rtol, atol, spherical_atol


def test_slerp():
    vec001 = torch.tensor([0, 0, 1], dtype=torch.float32)
    vec100 = torch.tensor([1, 0, 0], dtype=torch.float32)
    vec101 = torch.tensor([1, 0, 1], dtype=torch.float32) * np.sqrt(2) / 2

    # Test some edge cases - where frac is 0, 1, or two vecs are identical
    assert slerp(vec001, vec001, 0.5).allclose(vec001)
    assert slerp(vec001, 2 * vec001, 0.5).allclose(vec001)
    assert slerp(vec001, vec100, 0).allclose(vec001)
    assert slerp(vec001, vec100, 1).allclose(vec100)

    # Do an actual interpolation
    assert slerp(vec001, vec100, 0.5).allclose(vec101)
    # Test that SLERP normalizes things for us
    assert slerp(4 * vec001, 2 * vec100, 0.5).allclose(vec101)

    # Do a random 2D interpolation not involving orthogonal vectors
    vec_a, vec_b = torch.randn(2), torch.randn(2)
    norm_a, norm_b = vec_a / torch.linalg.norm(vec_a), vec_b / torch.linalg.norm(vec_b)
    total_angle = torch.arccos(torch.clip(torch.sum(norm_a * norm_b), -1.0, 1.0))
    frac = np.random.rand()
    rotation_amount = frac * total_angle
    c, s = torch.cos(rotation_amount), torch.sin(rotation_amount)
    rotation_matrix = torch.tensor([[c, -s], [s, c]])
    # Rotate vec_a towards vec_b... try both CW and CCW and take whichever worked
    vec_c_clockwise = rotation_matrix @ norm_a
    vec_c_counterclockwise = rotation_matrix.T @ norm_a
    assert slerp(vec_a, vec_b, frac).allclose(vec_c_clockwise) or slerp(
        vec_a, vec_b, frac
    ).allclose(vec_c_counterclockwise)


def test_geodesic(metric, data_x, data_y, high_rank_x, high_rank_y):
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    frac = np.random.rand(1)[0]
    pt_z = metric.geodesic(pt_x, pt_y, frac=frac)
    dist_xy = metric.length(pt_x, pt_y)
    dist_xz = metric.length(pt_x, pt_z)
    dist_zy = metric.length(pt_z, pt_y)

    assert metric.contains(
        pt_z, atol=atol
    ), f"point_along failed contains() test at frac={frac:.4f}  using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xz + dist_zy, atol=atol, rtol=rtol), (
        f"point_along at frac={frac:.4f} not along geodesic: "
        f"d(x,y) is {dist_xy:.4f} but d(x,m)+d(m,y) is {dist_xz + dist_zy:.4f}"
    )
    assert np.isclose(dist_xz / dist_xy, frac, atol=atol, rtol=rtol), (
        f"point_along failed to divide the total length: "
        f"frac is {frac:.4f} but d(x,m)/d(x,y) is {dist_xz/dist_xy:.4f}"
    )

    ang = angle(metric, pt_x, pt_z, pt_y).item()
    assert np.abs(ang - np.pi) < spherical_atol(
        1.0
    ), f"Angle through midpoint using {metric} should be pi but is {ang}"


def test_geodesic_endpoints(metric, data_x, data_y, high_rank_x, high_rank_y):
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    pt_t0 = metric._geodesic_impl(pt_x, pt_y, frac=0.0)
    pt_t1 = metric._geodesic_impl(pt_x, pt_y, frac=1.0)

    assert torch.allclose(
        metric.length(pt_x, pt_t0), pt_x.new_zeros(1), atol=atol
    ), "geodesic at frac=0 is not equivalent to x!"
    assert torch.allclose(
        metric.length(pt_y, pt_t1), pt_x.new_zeros(1), atol=atol
    ), "geodesic at frac=1 is not equivalent to y!"


def test_geodesic_gradient_descent(metric, data_x, data_y, high_rank_x, high_rank_y):
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    p_x, p_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    frac = np.random.rand(1)[0]

    # Case 1: compute geodesic using closed-form solution
    k_z_closed_form = GeodesicLengthSpace.geodesic(metric, p_x, p_y, frac=frac)
    # Case 2: compute geodesic using gradient descent - set tolerance to something << the value
    # we're going to assert
    k_z_grad_descent = LengthSpace.geodesic(
        metric,
        p_x,
        p_y,
        init_pt=k_z_closed_form,
        frac=frac,
        pt_tol=atol / 100,
        fn_tol=1e-6,
    )

    # Assert that they're fairly close in terms of length
    assert (
        metric.length(k_z_grad_descent, k_z_closed_form) < atol
    ), "Closed-form and grad-descent geodesics are not close!"


def test_log_exp_maps(metric, data_x, data_y, high_rank_x, high_rank_y):
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    frac = np.random.rand(1)[0]
    pt_z = metric.geodesic(pt_x, pt_y, frac=frac)

    tangent_x_to_y = metric.log_map(pt_x, pt_y)
    pt_y_2 = metric.exp_map(pt_x, tangent_x_to_y)
    assert metric.contains(pt_y_2), "exp(log(...)) landed outside the manifold"
    assert metric.length(pt_y, pt_y_2) < atol, "exp(x, log(x, y)) failed to recover y"
    assert np.isclose(
        metric.length(pt_x, pt_y_2), metric.norm(pt_x, tangent_x_to_y)
    ), "l(x, exp(x, log(x, y))) != ||log(x, y)||"

    tangent_y_to_x = metric.log_map(pt_y, pt_x)
    pt_x_2 = metric.exp_map(pt_y, tangent_y_to_x)
    assert metric.contains(pt_x_2), "exp(log(...)) landed outside the manifold"
    assert metric.length(pt_x, pt_x_2) < atol, "exp(y, log(y, x)) failed to recover x"
    assert np.isclose(
        metric.length(pt_y, pt_x_2), metric.norm(pt_y, tangent_y_to_x)
    ), "l(y, exp(y, log(y, x))) != ||log(y, x)||"

    pt_z_2 = metric.exp_map(pt_x, frac * tangent_x_to_y)
    assert metric.contains(pt_z_2), "exp(log(...)) landed outside the manifold"
    assert (
        metric.length(pt_z, pt_z_2) < atol
    ), "result of metric.geodesic() is not close to exp(x,frac*log(x,y))"

    pt_z_3 = metric.exp_map(pt_y, (1 - frac) * tangent_y_to_x)
    assert metric.contains(pt_z_3), "exp(log(...)) landed outside the manifold"
    assert (
        metric.length(pt_z, pt_z_3) < atol
    ), "result of metric.geodesic() is not close to exp(y,(1-frac)*log(y,x))"


def test_inner_product(metric, data_x, data_y, high_rank_x, high_rank_y):
    if metric.test_high_rank_data:
        x, y = high_rank_x, high_rank_y
    else:
        x, y = data_x, data_y

    pt_x, pt_y = metric.neural_data_to_point(x), metric.neural_data_to_point(y)

    vec_w = metric.log_map(pt_x, pt_y)
    norm_vec_w = vec_w / metric.norm(pt_x, vec_w)

    assert torch.isclose(
        metric.length(pt_x, pt_y), metric.norm(pt_x, vec_w)
    ), "Norm of tangent != length from x to y"

    # Note: it can happen (esp. with Stress) where exp_map(pt_x, norm_vec_w) lands outside of the
    # constraints. We would like to assert that length(pt_x, exp_map(pt_x, norm_vec_w))==1.,
    # but loop here in case exp_map goes out of bounds.
    scale, pt_z = 1.0, metric.exp_map(pt_x, norm_vec_w)
    while not metric.contains(pt_z):
        scale = scale / 2
        pt_z = metric.exp_map(pt_x, norm_vec_w * scale)

    assert torch.isclose(
        metric.length(pt_x, pt_z), scale * torch.tensor(1, dtype=pt_x.dtype)
    ), f"Distance to exp({scale} * normed tangent vector) should be {scale}"

    vec_v = metric.to_tangent(pt_x, norm_vec_w + torch.randn(norm_vec_w.shape))
    vec_v = vec_v / metric.norm(pt_x, vec_v)
    dot_wv = metric.inner_product(pt_x, norm_vec_w, vec_v)
    dot_vw = metric.inner_product(pt_x, norm_vec_w, vec_v)
    assert torch.isclose(dot_wv, dot_vw), "inner_product should be symmetric!"

    scale_w, scale_v = torch.rand(2)
    assert torch.isclose(
        scale_w * scale_v * dot_wv,
        metric.inner_product(pt_x, scale_w * norm_vec_w, scale_v * vec_v),
    ), "inner_product should scale output with scale of inputs (bilinear)!"


def test_parallel_transport(
    metric, data_x, data_y, data_z, high_rank_x, high_rank_y, high_rank_z
):
    if metric.test_high_rank_data:
        x, y, z = high_rank_x, high_rank_y, high_rank_z
    else:
        x, y, z = data_x, data_y, data_z

    pt_x, pt_y, pt_z = (
        metric.neural_data_to_point(x),
        metric.neural_data_to_point(y),
        metric.neural_data_to_point(z),
    )

    # Transport (x->z) tangent vector from x to y
    u_x = metric.to_tangent(pt_x, pt_z)
    u_y = metric.levi_civita(pt_x, pt_y, u_x)

    # Test 0: transporting a vector from a to a is a no-op
    u_x_again = metric.levi_civita(pt_x, pt_x, u_x)
    assert torch.allclose(u_x, u_x_again, atol=atol, rtol=rtol)

    # Test 1: result is in the tangent space of y
    assert torch.allclose(
        u_y, metric.to_tangent(pt_y, u_y), atol=atol, rtol=rtol
    ), "map of u_x to y did not land in the tangent space of y"

    # Test 2: vector is unchanged by transporting there and back again
    assert torch.allclose(
        u_x, metric.levi_civita(pt_y, pt_x, u_y), atol=atol, rtol=rtol
    ), "reverse map did not get back to the starting u"

    # Test 3: length is preserved by the map
    length_u_x = metric.norm(pt_x, u_x)
    length_u_y = metric.norm(pt_y, u_y)
    assert torch.isclose(
        length_u_x, length_u_y, rtol=rtol
    ), "map did not preserve length of u"

    # Test 4: inner products are preserved by the map (this involves creating a new random
    # tangent v_x at x)
    v_x = metric.to_tangent(
        pt_x, torch.randn(u_x.size(), dtype=pt_x.dtype) / np.sqrt(u_x.numel())
    )
    v_y = metric.levi_civita(pt_x, pt_y, v_x)
    dot_uv_x = metric.inner_product(pt_x, u_x, v_x)
    dot_uv_y = metric.inner_product(pt_y, u_y, v_y)
    assert torch.isclose(dot_uv_x, dot_uv_y, rtol=rtol), "map did not preserve dot(u,v)"


def test_curvature(
    metric, data_x, data_y, data_z, high_rank_x, high_rank_y, high_rank_z
):
    if metric.test_high_rank_data:
        x, y, z = high_rank_x, high_rank_y, high_rank_z
    else:
        x, y, z = data_x, data_y, data_z

    pt_x, pt_y, pt_z = (
        metric.neural_data_to_point(x),
        metric.neural_data_to_point(y),
        metric.neural_data_to_point(z),
    )

    curv = alexandrov(metric, pt_x, pt_y, pt_z)
    if metric.test_expected_curvature == "positive":
        assert curv > 0, "Expected curvature to be positive"
    elif metric.test_expected_curvature == "negative":
        assert curv < 0, "Expected curvature to be negative"
    elif metric.test_expected_curvature == "nonnegative":
        assert curv >= 0, "Expected curvature to be nonnegative"
    elif metric.test_expected_curvature == "nonpositive":
        assert curv <= 0, "Expected curvature to be nonpositive"
    elif metric.test_expected_curvature == "zero":
        assert torch.isclose(
            curv, torch.zeros(1, dtype=curv.dtype), atol=1e-5
        ), "expected curvature to be zero"
    else:
        raise ValueError(
            f"Bad expected_curvature argument: {metric.test_expected_curvature}"
        )


def test_projection_by_binary_search(
    metric, data_x, data_y, data_z, high_rank_x, high_rank_y, high_rank_z
):
    if metric.test_high_rank_data:
        x, y, z = high_rank_x, high_rank_y, high_rank_z
    else:
        x, y, z = data_x, data_y, data_z

    pt_x, pt_y, pt_z = (
        metric.neural_data_to_point(x),
        metric.neural_data_to_point(y),
        metric.neural_data_to_point(z),
    )

    proj, converged = project_by_binary_search(metric, pt_x, pt_y, pt_z, tol=atol / 2)
    dist_xy = metric.length(pt_x, pt_y)
    dist_xp = metric.length(pt_x, proj)
    dist_py = metric.length(proj, pt_y)

    assert (
        converged == OptimResult.CONVERGED
    ), f"Projected point failed to converge using {metric}: {proj}"
    assert metric.contains(
        proj, atol=atol
    ), f"Projected point failed contains() test using {metric}, {metric}"
    assert np.isclose(dist_xy, dist_xp + dist_py, atol=atol), (
        f"Projected point not along geodesic: "
        f"d(x,y) is {dist_xy} but d(x,p)+d(p,y) is {dist_xp + dist_py}"
    )

    frac = np.random.rand(1)[0]
    pt_geo = metric.geodesic(pt_x, pt_y, frac)
    pt_geo_projected, _ = project_by_binary_search(
        metric, pt_x, pt_y, pt_geo, tol=atol / 2
    )
    assert torch.allclose(
        pt_geo, pt_geo_projected, atol=atol
    ), "Projection of a point on the geodesic failed to recover that same point"

    for i in range(11):
        geo_pt = metric.geodesic(pt_x, pt_y, float(i) / 10)
        geo_dist = metric.length(pt_z, geo_pt)
        assert (
            metric.length(pt_z, proj) < geo_dist + atol
        ), f"length(z, proj) > length(z, geodesic({i/10:.2}))"


def test_projection_by_tangent_iteration(
    metric, data_x, data_y, data_z, high_rank_x, high_rank_y, high_rank_z
):
    if metric.test_high_rank_data:
        x, y, z = high_rank_x, high_rank_y, high_rank_z
    else:
        x, y, z = data_x, data_y, data_z

    pt_x, pt_y, pt_z = (
        metric.neural_data_to_point(x),
        metric.neural_data_to_point(y),
        metric.neural_data_to_point(z),
    )

    proj, converged = project_by_tangent_iteration(
        metric, pt_x, pt_y, pt_z, tol=atol / 2
    )
    dist_xy = metric.length(pt_x, pt_y)
    dist_xp = metric.length(pt_x, proj)
    dist_py = metric.length(proj, pt_y)

    direction = np.sign(
        metric.inner_product(
            pt_x, metric.log_map(pt_x, pt_y), metric.log_map(pt_x, pt_z)
        ).item()
    )
    a = angle(metric, pt_x, proj, pt_y).item()

    assert (
        converged == OptimResult.CONVERGED
    ), f"Projected point failed to converge using {metric}: {proj}"
    assert metric.contains(
        proj, atol=atol
    ), f"Projected point failed contains() test using {metric}, {metric}"
    if direction == +1 and dist_xp < dist_xy:
        # order is [x, proj, y] and so d(x,p)+d(p,y) should be equal to d(x,y)
        assert np.isclose(dist_xy, dist_xp + dist_py, atol=atol), (
            f"Projected point not along geodesic (case [x,y,p]): "
            f"d(x,y) is {dist_xy} but d(x,p)+d(p,y) is {dist_xp + dist_py}"
        )
        # Angle(x,p,y) should be pi
        assert np.abs(a - np.pi) < spherical_atol(-1.0)
    elif direction == +1 and dist_xp >= dist_xy:
        # order is [x, y, proj] and so d(x,y) should be equal to d(x,p)-d(p,y)
        assert np.isclose(dist_xy, dist_xp - dist_py, atol=atol), (
            f"Projected point not along geodesic (case [x,y,p]): "
            f"d(x,y) is {dist_xy} but d(x,p)-d(p,y) is {dist_xp - dist_py}"
        )
        # Angle(x,p,y) should be 0
        assert np.abs(a) < spherical_atol(1.0)
    elif direction == -1:
        # order is [proj, x, y] and so d(x,y) should be equal to d(p,y)-d(p,x)
        assert np.isclose(dist_xy, dist_py - dist_xp, atol=atol), (
            f"Projected point not along geodesic (case [p,x,y]): "
            f"d(x,y) is {dist_xy} but d(p,y)-d(p,x) is {dist_py - dist_xp}"
        )
        # Angle(x,p,y) should be 0
        assert np.abs(a) < spherical_atol(1.0)

    frac = np.random.rand(1)[0]
    pt_geo = metric.geodesic(pt_x, pt_y, frac)
    pt_geo_projected, _ = project_by_tangent_iteration(
        metric, pt_x, pt_y, pt_geo, tol=atol / 2
    )
    assert (
        metric.length(pt_geo, pt_geo_projected) < atol
    ), "Projection of a point on the geodesic failed to recover that same point"


def test_contains(metric, data_x, high_rank_x):
    x = high_rank_x if metric.test_high_rank_data else data_x
    pt = metric.neural_data_to_point(x)
    assert metric.contains(pt)
