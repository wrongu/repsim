import torch
from repsim.geometry import LengthSpace, RiemannianSpace, Point, Scalar


def slerp(pt_a: Point, pt_b: Point, frac: float) -> Point:
    """Spherical Linear intERPolation between two points -- see [1]. The interpolated point will
    always have unit norm.

    [1] https://en.m.wikipedia.org/wiki/Slerp

    :param pt_a: starting point. Will be normalized to unit length.
    :param pt_b: ending point. Will be normalized to unit length.
    :param frac: fraction of arc length from pt_a to pt_b
    :return: unit vector on the great-circle connecting pt_a to pt_b that is 'frac' of the distance
        from pt_a to pt_b
    """
    assert 0.0 <= frac <= 1.0, "frac must be between 0 and 1"

    def _norm(vec):
        return vec / torch.sqrt(torch.sum(vec * vec))

    # Normalize a and b to unit vectors
    a, b = _norm(pt_a), _norm(pt_b)

    # Check cases where we can break early (and doing so avoids things like divide-by-zero later!)
    if frac == 0.0:
        return a
    elif frac == 1.0:
        return b

    # Use dot product between (normed) a and b to test for colinearity
    dot_ab = torch.sum(a * b)

    # Check some more break-early cases based on dot product result.
    eps = 1e-6
    if dot_ab > 1.0 - eps:
        # dot(a,b) is effectively 1, so A and B are effectively the same vector. Do Euclidean
        # interpolation.
        return _norm(a * (1 - frac) + b * frac)
    elif dot_ab < -1 + eps:
        # dot(a,b) is effectively -1, so A and B are effectively at opposite poles. There are
        # infinitely many geodesics.
        raise ValueError("A and B are andipodal - cannot SLERP")

    # Get 'omega' - the angle between a and b, clipping for numerical stability
    omega = torch.acos(torch.clip(dot_ab, -1.0, 1.0))
    # Do interpolation using the SLERP formula
    a_frac = a * torch.sin((1 - frac) * omega) / torch.sin(omega)
    b_frac = b * torch.sin(frac * omega) / torch.sin(omega)
    return (a_frac + b_frac).reshape(a.shape)


def angle(
    space: LengthSpace, pt_a: Point, pt_b: Point, pt_c: Point, **kwargs
) -> Scalar:
    """Angle B of triangle ABC, based on incident angle of geodesics AB and CB.

    If B is along the geodesic from A to C, then the angle is pi (180 degrees). If A=C, then the
    angle is zero.

    :param space: a LengthSpace defining the metric and geodesics
    :param pt_a: point A
    :param pt_b: point B
    :param pt_c: point C :key delta: at what scale does the space look locally Euclidean? Default
        0.01
    :param kwargs: optional arguments passed to geodesic optimization, if needed
    :return:
    """
    if isinstance(space, RiemannianSpace):
        # Riemannian manifolds have tangent spaces and inner products that we can use to compute
        # the angle easily
        tangent_ba, tangent_bc = space.log_map(pt_b, pt_a), space.log_map(pt_b, pt_c)
        norm_ba = tangent_ba / space.norm(pt_b, tangent_ba)
        norm_bc = tangent_bc / space.norm(pt_b, tangent_bc)
        cos_b = space.inner_product(pt_b, norm_ba, norm_bc)
        return torch.arccos(torch.clip(cos_b, -1.0, 1.0))
    else:
        # In general length spaces, we'll approximate the angle by constructing a point 1/1000th
        # of the way from B to each of A and C, then use the law of cosines locally
        delta = kwargs.pop("delta", 1e-3)
        pt_ba = space.geodesic(pt_b, pt_a, frac=delta, **kwargs)
        pt_bc = space.geodesic(pt_b, pt_c, frac=delta, **kwargs)

        # Law of cosines using small local distances around B
        d_c, d_a, d_b = (
            space.length(pt_b, pt_ba),
            space.length(pt_b, pt_bc),
            space.length(pt_ba, pt_bc),
        )
        cos_b = 0.5 * (d_a * d_a + d_c * d_c - d_b * d_b) / (d_a * d_c)
        return torch.arccos(torch.clip(cos_b, -1.0, 1.0))


__all__ = ["slerp", "angle"]
