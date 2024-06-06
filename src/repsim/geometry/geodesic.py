from typing import List

# Avoid circular import of LengthSpace, Point - only import if in type_checking mode
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repsim.geometry import LengthSpace, Point


def midpoint(space: "LengthSpace", pt_a: "Point", pt_b: "Point", **kwargs) -> "Point":
    """Get midpoint, in terms of equally dividing the geodesic, of two points.

    :param space: "LengthSpace" defining the metric and geodesic
    :param pt_a: start point
    :param pt_b: end point
    :param kwargs: optional configuration passed through to optimization, if needed
    :return: pt_c : the midpoint between pt_a and pt_b such that
        space.length(pt_a,pt_c)==space.length(pt_c,pt_b)
    """
    return space.geodesic(pt_a, pt_b, frac=0.5, **kwargs)


def subdivide_geodesic(
    space: "LengthSpace", pt_a: "Point", pt_b: "Point", octaves: int = 5, **kwargs
) -> List["Point"]:
    """Compute multiple points along geodesic from pt_a to pt_b by recursively halving 'octaves'
    times. Result is a list of points where index [0] is pt_a and index [-1] is pt_b.

    :param space: "LengthSpace" defining the metric and geodesic
    :param pt_a: start point
    :param pt_b: end point
    :param octaves: number of halvings. Total number of points will be 2**octaves+1
    :param kwargs: optional configuration passed through to optimization, if needed
    :return: List of points [pt_a, ..., pt_b] along geodesic connecting pt_a to pt_b
    """
    midpt = midpoint(space, pt_a, pt_b, **kwargs)

    if octaves > 1:
        # Recursively subdivide each half
        left_half = subdivide_geodesic(space, pt_a, midpt, octaves - 1)
        right_half = subdivide_geodesic(space, midpt, pt_b, octaves - 1)
        return left_half + right_half[1:]
    else:
        # Base case
        return [pt_a, midpt, pt_b]


__all__ = ["midpoint", "subdivide_geodesic"]
