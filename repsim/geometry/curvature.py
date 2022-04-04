import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.geodesic import midpoint
import warnings


def _bisector_length(x: Scalar, y: Scalar, z: Scalar) -> Scalar:
    """
    Given a triangle ABC with side lengths AB=x, BC=y, AC=z, returns
    the length of BD, where D is the midpoint of AC.

    Arguments:
        x (Scalar): The length of side AB.
        y (Scalar): The length of side BC.
        z (Scalar): The length of side AC.

    Returns:
        Scalar: The length of side BD.

    """
    return torch.sqrt(x * x / 2 + y * y / 2 - z * z / 4)


def alexandrov(pt_a: Point, pt_b: Point, pt_c: Point, space: Manifold) -> Scalar:
    """
    Compute the Alexandrov curvature of a triangle.

    Arguments:
        pt_a (Point): The first point of the triangle.
        pt_b (Point): The second point of the triangle.
        pt_c (Point): The third point of the triangle.
        space (Manifold): The manifold in which the triangle lives.

    Returns:
        Scalar: The Alexandrov curvature of the triangle.

    """
    midpt_ac, converged = midpoint(pt_a, pt_c, space)
    if not converged:
        warnings.warn("midoint() failed to converge. Curvature may be inaccurate.")
    bisector_lenth = space.length(pt_b, midpt_ac)
    euclidean_bisector_length = _bisector_length(
        space.length(pt_a, pt_b), space.length(pt_b, pt_c), space.length(pt_a, pt_c)
    )
    return (bisector_lenth - euclidean_bisector_length) / euclidean_bisector_length
