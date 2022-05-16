import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.optimize import minimize, OptimResult
from typing import Union, Iterable, List, Tuple
import warnings


def path_length(pts: Iterable[Point],
                space: Manifold) -> Scalar:
    l, pt_a = Scalar([0.]), None
    for pt_b in pts:
        if pt_a is not None:
            l += space.length(pt_a, pt_b)
        pt_a = pt_b
    return l


def subdivide_geodesic(pt_a: Point,
                       pt_b: Point,
                       space: Manifold,
                       octaves: int = 5,
                       **kwargs) -> List[Point]:
    midpt, converged = midpoint(pt_a, pt_b, space , **kwargs)
    if not converged:
        warnings.warn(f"midpoint() failed to converge; remaining {octaves} subdivisions may be inaccurate")
    if octaves > 1 and converged:
        # Recursively subdivide each half
        left_half = subdivide_geodesic(pt_a, midpt, space, octaves-1)
        right_half = subdivide_geodesic(midpt, pt_b, space, octaves-1)
        return left_half + right_half[1:]
    else:
        # Base case
        return [pt_a, midpt, pt_b]


def project_along(pt_fro: Point,
                  pt_to: Point,
                  pt_a: Point,
                  space: Manifold,
                  tol=1e-6,
                  max_recurse=20) -> Tuple[Point, OptimResult]:
    """Find 'projection' of pt_a onto a geodesic that spans [pt_fro, pt_to]

    :param pt_fro: start point of the geodesic
    :param pt_to: end point of the geodesic
    :param pt_a: point to be projected
    :param space: a Manifold
    :param tol: result will be within this tolerance, as measured by space.length
    :param max_recurse: how many halvings is too many halvings? 0.5^20 gives a resolution of about 1 part per million
    :return: pt_x, a point on the manifold that lies along a geodesic connecting [pt_fro, pt_to], such that the length
    from pt_a to pt_x is minimized
    """
    dist_a_fro, dist_a_to = space.length(pt_fro, pt_a), space.length(pt_a, pt_to)
    # Break-early case 1: pt_a is already along a geodesic
    if torch.isclose(dist_a_fro + dist_a_to, space.length(pt_fro, pt_to), atol=tol):
        return pt_a.clone(), OptimResult.CONVERGED
    # Break-early case 2: pt_fro and pt_to are the same point
    elif space.length(pt_fro, pt_to) < tol:
        return space.project((pt_fro+pt_to)/2), OptimResult.CONVERGED

    # Get a midpoint between 'fro' and 'to'. TODO if multiple geodesics, need to pick whichever is closest to pt_a
    mid, status = midpoint(pt_fro, pt_to, space)
    if status != OptimResult.CONVERGED:
        warnings.warn("midpoint() failed to converge. result of project_along() may be inaccurate")

    # Break-early case 3: we've recursed and subdivided too many times.
    if max_recurse == 0:
        return mid, OptimResult.MAX_STEPS_REACHED

    # Distance from a to mid
    dist_a_mid = space.length(mid, pt_a)

    # Recursively subdivide the geodesic
    if dist_a_mid < min(dist_a_fro, dist_a_to):
        # Midpoint is min.. recurse to whichever side is closer to pt_a
        if dist_a_fro < dist_a_to:
            return project_along(pt_fro, mid, pt_a, space, tol=tol, max_recurse=max_recurse-1)
        else:
            return project_along(mid, pt_to, pt_a, space, tol=tol, max_recurse=max_recurse-1)
    elif dist_a_fro < dist_a_to:
        # Dist to 'pt_fro' is min. Recurse left.
        return project_along(pt_fro, mid, pt_a, space, tol=tol, max_recurse=max_recurse-1)
    else:
        # Dist to 'pt_to' is min. Recurse right.
        return project_along(mid, pt_fro, pt_a, space, tol=tol, max_recurse=max_recurse-1)


def point_along(pt_a: Point,
                pt_b: Point,
                space: Manifold,
                frac: float,
                guess: Union[Point, None] = None,
                **kwargs) -> Tuple[Point, OptimResult]:
    """Given ptA and ptB, return ptC along the geodesic between them, such that d(ptA,ptC) is frac percent of the
    total length ptA to ptB.
    """

    if frac < 0. or frac > 1.:
        raise ValueError(f"'frac' must be in [0, 1] but is {frac}")

    # Three cases where we can just break early without optimizing
    if frac == 0:
        return pt_a, OptimResult.NO_OPT_NEEDED
    elif frac == 1:
        return pt_b, OptimResult.NO_OPT_NEEDED
    elif torch.allclose(pt_a, pt_b, atol=kwargs.get('pt_tol', 1e-6)):
        return space.project((pt_a+pt_b)/2), OptimResult.NO_OPT_NEEDED
    
    # We can also use a closed-form geodesic computation from the space itself
    # if one is available.
    if space._has_implemented_closed_form_geodesic():
        return space.geodesic_from(pt_a, pt_b, frac), OptimResult.NO_OPT_NEEDED

    # For reference, we know we're on the geodesic when dist_ap + dist_pb = dist_ab
    # dist_ab = space.length(pt_a, pt_b)

    # Default initial guess to projection of euclidean interpolated point
    pt = space.project(guess) if guess is not None else space.project((1-frac)*pt_a + frac*pt_b)

    def calc_error(pt_c):
        # Two sources of error: total length should be dist_ab, and dist_a/(dist_a+dist_b) should equal 'frac'
        dist_ac, dist_bc = space.length(pt_a, pt_c), space.length(pt_c, pt_b)
        return dist_ac*dist_ac*(1-frac) + dist_bc*dist_bc*frac

    return minimize(calc_error, pt, space, **kwargs)


def midpoint(pt_a: Point,
             pt_b: Point,
             space: Manifold,
             **kwargs) -> Tuple[Point, OptimResult]:
    return point_along(pt_a, pt_b, space, frac=0.5, **kwargs)


__all__ = ["path_length", "subdivide_geodesic", "point_along", "midpoint", "project_along"]
