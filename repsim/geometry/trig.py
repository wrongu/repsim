import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.geodesic import point_along
import warnings


def angle(pt_a: Point, pt_b: Point, pt_c: Point, space: Manifold, **kwargs) -> Scalar:
    """
    Angle B of triangle ABC, based on incident angle of geodesics AB and CB.

    If B is along the geodesic from A to C, then the angle is pi (180 degrees).
    If A=C, then the angle is zero.

    Arguments:
        pt_a (Point): The first point of the triangle.
        pt_b (Point): The second point of the triangle.
        pt_c (Point): The third point of the triangle.
        space (Manifold): The manifold in which the triangle lives.

    Returns:
        Scalar: The angle B of the triangle.

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
