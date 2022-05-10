import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from typing import Callable, Tuple
import enum


class OptimResult(enum.Enum):
    CONVERGED = 0
    MAX_STEPS_REACHED = 1
    CONDITIONS_VIOLATED = 2


def minimize(fun: Callable[[Point], Scalar],
             init: Point,
             space: Manifold,
             *,
             pt_tol: float = 1e-6,
             fn_tol: float = 1e-6,
             init_step_size: float = 0.1,
             wolfe_c1=1e-4,
             wolfe_c2=0.9,
             max_iter: int = 10000) -> Tuple[Point, OptimResult]:

    def fn_wrapper(x):
        return fun(x), torch.autograd.functional.jacobian(fun, x)

    step_size, pt = init_step_size, init.clone()
    fval, grad = fn_wrapper(pt)
    for itr in range(max_iter):
        # Update by gradient descent + line search
        step_direction = -grad
        new_pt = space.project(pt.detach() + step_size * step_direction)
        new_fval, new_grad = fn_wrapper(new_pt)

        # Test for convergence
        if space.length(pt.detach(), new_pt.detach()) <= pt_tol and \
                new_fval <= fval and \
                fval - new_fval <= fn_tol:
            return new_pt.detach() if new_fval < fval else pt.detach(), OptimResult.CONVERGED

        # Check Wolfe conditions
        sq_step_size = (grad * step_direction).sum()
        condition_i = new_fval <= fval + wolfe_c1 * step_size * sq_step_size
        condition_ii = -(step_direction*new_grad).sum() <= -wolfe_c2 * sq_step_size
        if condition_i and condition_ii:
            # Both conditions met! Update pt and loop.
            pt, fval, grad = new_pt.detach().clone(), new_fval, new_grad
        elif condition_i and not condition_ii:
            # Step size is too small - adjust and loop, leaving pt, fval, and grad unchanged
            step_size *= 1.1
        elif condition_ii and not condition_i:
            # Step size is too big - adjust and loop, leaving pt, fval, and grad unchanged
            step_size *= 0.8
        else:
            return pt.detach(), OptimResult.CONDITIONS_VIOLATED

    # Max iterations reached – return final value of 'pt' with flag indicating max steps reached
    return pt.detach(), OptimResult.MAX_STEPS_REACHED
