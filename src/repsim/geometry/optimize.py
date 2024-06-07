import enum
import torch
import numpy as np
from typing import Callable, Tuple

# Avoid circular import of LengthSpace, Point, Scalar - only import if in type_checking mode
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repsim.geometry import LengthSpace, RiemannianSpace, Point, Scalar

from repsim.geometry.geodesic import midpoint


class OptimResult(enum.Enum):
    CONVERGED = 0
    MAX_STEPS_REACHED = 1
    CONDITIONS_VIOLATED = 2
    ILL_POSED = 3
    NO_OPT_NEEDED = 0


def minimize(
    space: "LengthSpace",
    fun: Callable[["Point"], "Scalar"],
    init: "Point",
    *,
    pt_tol: float = 1e-6,
    fn_tol: float = 1e-6,
    init_step_size: float = 0.1,
    max_step_size: float = 100.0,
    wolfe_c1=1e-4,
    wolfe_c2=0.9,
    wolfe_c2_min=1e-2,
    max_iter: int = 10000
) -> Tuple["Point", OptimResult]:
    """Function minimization on a Length Space by gradient descent and line search.

    :param space: Length Space
    :param fun: torch-differentiable function to be minimized
    :param init: initial point
    :param pt_tol: convergence tolerance for changes in the coordinate
    :param fn_tol: convergence tolerance for changes in the function
    :param init_step_size: initial gradient descent step size
    :param max_step_size: largest sane gradient descent step size
    :param wolfe_c1: threshold on first Wolfe condition (progress check - is function improving at
        least a little?)
    :param wolfe_c2: threshold on second Wolfe condition (curvature check - is gradient changing by
        not too much?)
    :param wolfe_c2_min: rarely both conditions fail, so we reduce c2. Stop trying to do thise when
        wolfe_c2 < wolfe_c2_min
    :param max_iter: break regardless of convergence if this many steps reached
    :return: tuple of (point, OptimResult) where the OptimResult indicates convergence status
    """

    step_size, pt = init_step_size, init.clone()
    fval, grad = fun(pt), torch.autograd.functional.jacobian(fun, pt)
    for itr in range(max_iter):
        # Update by gradient descent + line search
        step_direction = -1 * grad
        new_pt = space.project(pt + step_size * step_direction)
        new_fval, new_grad = fun(new_pt), torch.autograd.functional.jacobian(
            fun, new_pt
        )

        # Test for convergence
        if (
            space.length(pt, new_pt) <= pt_tol
            and new_fval <= fval
            and fval - new_fval <= fn_tol
        ):
            return new_pt if new_fval < fval else pt, OptimResult.CONVERGED

        # Check Wolfe conditions
        sq_step_size = (grad * step_direction).sum()
        condition_i = new_fval <= fval + wolfe_c1 * step_size * sq_step_size
        condition_ii = (step_direction * new_grad).sum() >= wolfe_c2 * sq_step_size
        if condition_i and condition_ii:
            # Both conditions met! Update pt and loop.
            pt, fval, grad = new_pt.clone(), new_fval, new_grad
        elif condition_i and not condition_ii:
            # Step size is too small - accept the new value but adjust step_size for next loop
            pt, fval, grad = new_pt.clone(), new_fval, new_grad
            step_size = min(max_step_size, step_size * 1.1)
        elif condition_ii and not condition_i:
            # Step size is too big - adjust and loop, leaving pt, fval, and grad unchanged
            step_size *= 0.8
        else:
            # Both conditions violated, indicating that the curvature is high (so condition 2
            # fails) and the function is barely changing (so condition 1 fails). When this first
            # happens, we can make some more (slow) progress by making the threshold on c2 less
            # strict. But eventually we will give up when wolfe_c2 < wolfe_c2_min
            wolfe_c2 *= 0.8
            if wolfe_c2 < wolfe_c2_min:
                return pt, OptimResult.CONDITIONS_VIOLATED
            elif new_fval < fval:
                # Despite condition weirdness, the new fval still improved. Accept the new point
                # then loop.
                pt, fval, grad = new_pt.clone(), new_fval, new_grad

    # Max iterations reached – return final value of 'pt' with flag indicating max steps reached
    return pt, OptimResult.MAX_STEPS_REACHED


def project_by_binary_search(
    space: "LengthSpace",
    pt_a: "Point",
    pt_b: "Point",
    pt_c: "Point",
    *,
    dist_a_b=None,
    dist_a_c=None,
    dist_c_b=None,
    tol=1e-6,
    max_recurse=20
) -> Tuple["Point", OptimResult]:
    """Find 'projection' of pt_c onto a geodesic that spans [pt_a, pt_b] by recursively halving the
    geodesic that connects pt_a to pt_b.

    Note: because this *subdivides* the geodesic, it can only interpolate between pt_a and pt_b
    not extrapolate. If extrapolation is required, use project_by_tangent_iteration

    :param space: a LengthSpace defining the metric and geodesic
    :param pt_a: start point of the geodesic
    :param pt_b: end point of the geodesic
    :param pt_c: point to be projected
    :param tol: result will be within this tolerance, as measured by space.length
    :param max_recurse: how many halvings is too many halvings? 0.5^20 gives a resolution of about 1
        part per million
    :return: pt_x, a point on the manifold that lies along a geodesic connecting [pt_fro, pt_to],
        such that the length
    from pt_a to pt_x is minimized
    """
    if dist_a_b is None:
        dist_a_b = space.length(pt_a, pt_b)
    if dist_a_c is None:
        dist_a_c = space.length(pt_a, pt_c)
    if dist_c_b is None:
        dist_c_b = space.length(pt_c, pt_b)

    # Break-early case 1: pt_c is already along a geodesic
    if torch.isclose(dist_a_c + dist_c_b, dist_a_b, atol=tol):
        return pt_c.clone(), OptimResult.CONVERGED
    # Break-early case 2: pt_a and pt_b are equivalent (note that this does not mean they are
    # 'identical'). Return a midpoint between a and b.
    elif dist_a_b < tol:
        return midpoint(space, pt_a, pt_b), OptimResult.CONVERGED
    # Break-early case 3: we've recursed and subdivided too many times. Return a copy of pt_a or
    # pt_b - whichever is closer
    if max_recurse == 0:
        if dist_a_c < dist_c_b:
            return pt_a.clone(), OptimResult.MAX_STEPS_REACHED
        else:
            return pt_b.clone(), OptimResult.MAX_STEPS_REACHED

    # Get a midpoint between 'fro' and 'to'.
    # TODO if multiple geodesics, need to pick whichever is closest to pt_c
    mid = midpoint(space, pt_a, pt_b)

    # Distance from midpoint to pt_c
    dist_mid_c = space.length(mid, pt_c)

    # Recurse LEFT if A is closer to C than B is, or RIGHT otherwise
    if dist_a_c < dist_c_b:
        return project_by_binary_search(
            space,
            pt_a=pt_a,
            pt_b=mid,
            pt_c=pt_c,
            tol=tol,
            dist_a_b=dist_a_b / 2,
            dist_a_c=dist_a_c,
            dist_c_b=dist_mid_c,
            max_recurse=max_recurse - 1,
        )
    else:
        return project_by_binary_search(
            space,
            pt_a=mid,
            pt_b=pt_b,
            pt_c=pt_c,
            tol=tol,
            dist_a_b=dist_a_b / 2,
            dist_a_c=dist_mid_c,
            dist_c_b=dist_c_b,
            max_recurse=max_recurse - 1,
        )


def project_by_tangent_iteration(
    space: "RiemannianSpace",
    pt_a: "Point",
    pt_b: "Point",
    pt_c: "Point",
    *,
    tol=1e-6,
    max_iterations=100
) -> Tuple["Point", OptimResult]:
    """Find 'projection' of pt_c onto a geodesic that spans (pt_a, pt_b) by iteratively nudging pt_a
    towards or away from pt_b until (pt_a, pt_c) is orthogonal to (pt_a, pt_b)

    Note: resulting point may be extrapolated outside of the (a, b) interval

    :param space: a RiemannianSpace defining the metric and geodesic.
    :param pt_a: start point of the geodesic
    :param pt_b: end point of the geodesic
    :param pt_c: point to be projected
    :param tol: declare convergence once inner-product of tangent vectors is within 'tol' of zero
    :param max_iterations: max iterations
    :return: pt_x, a point on the manifold that lies along a geodesic connecting [pt_fro, pt_to],
        such that the length from pt_a to pt_x is minimized
    """

    dist_ab = space.length(pt_a, pt_b)

    # Break-early if pt_a and pt_b are equivalent (note that this does not mean they are
    # 'identical'). Return a midpoint between a and b and flag result as ill-posed, since we
    # don't have a good sense of the 'direction' from a to b and therefore no good sense of how
    # to extrapolate the geodesic
    if dist_ab < tol:
        return midpoint(space, pt_a, pt_b), OptimResult.ILL_POSED

    projected_lengths = []
    proj, tangent_ab_norm, t = pt_a.clone(), space.log_map(pt_a, pt_b) / dist_ab, 0.0
    for itr in range(max_iterations):
        # Get inner-product between (p, c) vector and (a, b) after transporting the first back to a
        tangent_pc = space.log_map(proj, pt_c)
        tangent_pc_at_a = space.levi_civita(proj, pt_a, tangent_pc)
        length_pc_along_ab = space.inner_product(pt_a, tangent_pc_at_a, tangent_ab_norm)
        if torch.isnan(length_pc_along_ab):
            print("WTF: NaN")
        if (
            len(projected_lengths) >= 2
            and np.abs(projected_lengths[-1]) - np.abs(projected_lengths[-2]) > 0
        ):
            print("WTF: growing")
        projected_lengths.append(length_pc_along_ab.item())
        # If length of step size is less than tol, declare convergence
        if torch.abs(length_pc_along_ab) < tol:
            return proj, OptimResult.CONVERGED
        # Get new value for pt_a by moving 'length_ac_along_ab' distance in the ab direction,
        # which may be negative
        t = t + length_pc_along_ab
        proj = space.exp_map(pt_a, t * tangent_ab_norm)
    return pt_a, OptimResult.MAX_STEPS_REACHED
