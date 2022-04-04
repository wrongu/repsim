import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.optimize import minimize, OptimResult
from typing import Union, Iterable, List, Tuple
import warnings


def path_length(pts: Iterable[Point], space: Manifold) -> Scalar:
    """
    Compute the length of a path in a manifold.

    Arguments:
        pts (Iterable[Point]): a sequence of points
        space (Manifold): a manifold

    Returns:
        Scalar: the length of the path

    """
    l, pt_a = Scalar([0.0]), None
    for pt_b in pts:
        if pt_a is not None:
            l += space.length(pt_a, pt_b)
        pt_a = pt_b
    return l


def subdivide_geodesic(
    pt_a: Point, pt_b: Point, space: Manifold, octaves: int = 5, **kwargs
) -> List[Point]:
    """
    Given two points on a geodesic, subdivide the geodesic into a list of points.

    Arguments:
        pt_a (Point): The first point of the geodesic.
        pt_b (Point): The second point of the geodesic.
        space (Manifold): The manifold in which the geodesic lives.
        octaves (int): The number of subdivisions to make.

    Returns:
        List[Point]: A list of points along the geodesic.

    """
    midpt, converged = midpoint(pt_a, pt_b, space, **kwargs)
    if not converged:
        warnings.warn(
            f"midpoint() failed to converge; remaining {octaves} subdivisions may be inaccurate"
        )
    if octaves > 1 and converged:
        # Recursively subdivide each half
        left_half = subdivide_geodesic(pt_a, midpt, space, octaves - 1)
        right_half = subdivide_geodesic(midpt, pt_b, space, octaves - 1)
        return left_half + right_half[1:]
    else:
        # Base case
        return [pt_a, midpt, pt_b]


def project_along(
    pt_fro: Point, pt_to: Point, pt_a: Point, space: Manifold, tol=1e-6
) -> Tuple[Point, OptimResult]:
    """
    Find 'projection' of pt_a onto a geodesic that spans [pt_fro, pt_to]

    Arguments:
        pt_fro (Point): The first point of the geodesic.
        pt_to (Point): The second point of the geodesic.
        pt_a (Point): The point to project onto the geodesic.
        space (Manifold): The manifold in which the geodesic lives.
        tol (float): The tolerance for convergence, in units of `space.length`.

    Returns:
        Tuple[Point, OptimResult]: a point on the manifold that lies along a
            geodesic connecting [pt_fro, pt_to], such that the length from pt_a
            to pt_x is minimized

    """
    dist_a_fro, dist_a_to = space.length(pt_fro, pt_a), space.length(pt_a, pt_to)
    # Break-early case 1: pt_a is already along a geodesic
    if torch.isclose(dist_a_fro + dist_a_to, space.length(pt_fro, pt_to), atol=tol):
        return pt_a.clone(), OptimResult.CONVERGED
    # Break-early case 2: pt_fro and pt_to are the same point
    elif space.length(pt_fro, pt_to) < tol:
        return space.project((pt_fro + pt_to) / 2), OptimResult.CONVERGED

    # Get a midpoint between 'fro' and 'to'. TODO if multiple geodesics, need
    # to pick whichever is closest to pt_a
    mid, status = midpoint(pt_fro, pt_to, space)
    if status != OptimResult.CONVERGED:
        warnings.warn(
            "midpoint() failed to converge. result of project_along() may be inaccurate"
        )

    # Distance from a to mid
    dist_a_mid = space.length(mid, pt_a)

    # Recursively subdivide the geodesic
    if dist_a_mid < min(dist_a_fro, dist_a_to):
        # Midpoint is min.. recurse to whichever side is closer to pt_a
        if dist_a_fro < dist_a_to:
            return project_along(pt_fro, mid, pt_a, space, tol=tol)
        else:
            return project_along(mid, pt_to, pt_a, space, tol=tol)
    elif dist_a_fro < dist_a_to:
        # Dist to 'pt_fro' is min. Recurse left.
        return project_along(pt_fro, mid, pt_a, space, tol=tol)
    else:
        # Dist to 'pt_to' is min. Recurse right.
        return project_along(mid, pt_fro, pt_a, space, tol=tol)


def point_along(
    pt_a: Point,
    pt_b: Point,
    space: Manifold,
    frac: float,
    guess: Union[Point, None] = None,
    **kwargs,
) -> Tuple[Point, OptimResult]:
    """
    Given ptA and ptB, return ptC along the geodesic between them, such that
    d(ptA,ptC) is frac percent of the total length ptA to ptB.

    Arguments:
        pt_a (Point): The first point of the geodesic.
        pt_b (Point): The second point of the geodesic.
        space (Manifold): The manifold in which the geodesic lives.
        frac (float): The fraction of the total length of the geodesic to
            project along.
        guess (Union[Point, None]): A point to use as a guess for the result.
            If None, a midpoint is used.

    Returns:
        Tuple[Point, OptimResult]: a point on the manifold that lies along a
            geodesic connecting [pt_a, pt_b], such that the length from pt_a
            to pt_x is minimized and d(ptA,ptC) is frac percent of the total
            length ptA to ptB

    """

    if frac < 0.0 or frac > 1.0:
        raise ValueError(f"'frac' must be in [0, 1] but is {frac}")

    # Three cases where we can just break early without optimizing
    if frac == 0.0:
        return pt_a, OptimResult.CONVERGED
    elif frac == 1.0:
        return pt_b, OptimResult.CONVERGED
    elif torch.allclose(pt_a, pt_b, atol=kwargs.get("pt_tol", 1e-6)):
        return space.project((pt_a + pt_b) / 2), OptimResult.CONVERGED

    # For reference, we know we're on the geodesic when dist_ap + dist_pb = dist_ab
    dist_ab = space.length(pt_a, pt_b)

    # Default initial guess to projection of euclidean interpolated point
    pt = (
        space.project(guess)
        if guess is not None
        else space.project((1 - frac) * pt_a + frac * pt_b)
    )

    def _calc_error(pt_c):
        # Two sources of error: total length should be dist_ab, and dist_a/(dist_a+dist_b) should equal 'frac'
        dist_a, dist_b = space.length(pt_a, pt_c), space.length(pt_c, pt_b)
        total_length = dist_a + dist_b
        length_error = torch.clip(total_length - dist_ab, 0.0, None)
        frac_error = torch.abs(dist_a - frac * dist_ab)
        return length_error + frac_error

    return minimize(_calc_error, pt, space, **kwargs)


def midpoint(
    pt_a: Point, pt_b: Point, space: Manifold, **kwargs
) -> Tuple[Point, OptimResult]:
    """
    Compute the midpoint of the geodesic between pt_a and pt_b.

    Arguments:
        pt_a (Point): The first point of the geodesic.
        pt_b (Point): The second point of the geodesic.
        space (Manifold): The manifold in which the geodesic lives.

    Returns:
        Tuple[Point, OptimResult]: a midpoint on the manifold that lies along a
            geodesic connecting [pt_a, pt_b]

    """
    return point_along(pt_a, pt_b, space, frac=0.5, **kwargs)


__all__ = ["path_length", "subdivide_geodesic", "point_along", "midpoint"]
