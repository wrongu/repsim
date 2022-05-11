import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.geodesic import point_along
import warnings


def slerp(
    pt_a: Point,
    pt_b: Point,
    frac: float,
) -> Point:
    """
    Arc slerp between two points.

    https://en.m.wikipedia.org/wiki/Slerp

    Arguments:
        pt_a (Point): First point.
        pt_b (Point): Second point.
        frac (float): Fraction of the way from pt_a to pt_b.

    Returns:
        Point: The slerp between pt_a and pt_b.

    """
    assert frac >= 0.0 and frac <= 1.0, "frac must be between 0 and 1"

    # ab = torch.matmul(pt_a, pt_b)
    # dot of the ravels:
    ab = torch.dot(pt_a.view(-1), pt_b.view(-1))
    # omega = torch.acos(torch.clamp(ab, -1.0, 1.0))
    omega = torch.acos(ab)
    print("F", pt_a.view(-1), pt_b.view(-1), ab, omega)
    pt_a_frac = pt_a * torch.sin((1 - frac) * omega) / torch.sin(omega)
    pt_b_frac = pt_b * torch.sin(frac * omega) / torch.sin(omega)
    n = (pt_a_frac + pt_b_frac).reshape(pt_a.shape)
    return n


def angle(pt_a: Point, pt_b: Point, pt_c: Point, space: Manifold, **kwargs) -> Scalar:
    """
    Angle B of triangle ABC, based on incident angle of geodesics AB and CB.
    If B is along the geodesic from A to C, then the angle is pi (180 degrees).
    If A=C, then the angle is zero.
    """
    pt_ba, converged_ba = point_along(pt_b, pt_a, space, frac=0.02, **kwargs)
    pt_bc, converged_bc = point_along(pt_b, pt_c, space, frac=0.02, **kwargs)
    if not (converged_ba and converged_bc):
        warnings.warn("point_along failed to converge; angle may be inaccurate")
    # Law of cosines using small local distances
    d_c, d_a, d_b = (
        space.length(pt_b, pt_ba),
        space.length(pt_b, pt_bc),
        space.length(pt_ba, pt_bc),
    )
    cos_b = 0.5 * (d_a * d_a + d_c * d_c - d_b * d_b) / (d_a * d_c)
    return torch.arccos(torch.clip(cos_b, -1.0, 1.0))


__all__ = ["angle"]
