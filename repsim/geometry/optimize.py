import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from typing import Callable, Tuple
import enum


class OptimResult(enum.Enum):
    MAX_STEPS_REACHED = 0
    CONVERGED = 1
    POINT_CONVERGED = 3
    GRAD_CONVERGED = 5


def minimize(fun: Callable[[Point], Scalar],
             init: Point,
             space: Manifold,
             *,
             pt_tol: float = 1e-6,
             grad_tol: float = 1e-6,
             init_step_size: float = 0.1,
             finite_difference_size: float = 1e-3,
             max_iter: int = 10000) -> Tuple[Point, OptimResult]:
    step_size = init_step_size
    pt = init.clone()
    pt.requires_grad_(True)
    for itr in range(max_iter):
        fval = fun(pt)

        # Get grad in terms of ambient space
        grad = torch.autograd.grad(fval, pt)[0]

        # If all entries in grad are tiny, consider us converged; projecting into tangent space below will likely only
        # make entries smaller
        if torch.all(torch.abs(grad) < grad_tol):
            return pt.detach(), OptimResult.GRAD_CONVERGED

        # Update by gradient descent + line search to reduce step size
        with torch.no_grad():
            # To get grad in tangent space, take a small step and project back to manifold, then normalize
            tangent_grad = space.project(pt + finite_difference_size * grad)
            tangent_grad = tangent_grad / torch.linalg.norm(tangent_grad.flatten())
            # Take gradient step in the direction of tangent_grad, with size set by step_size
            new_pt = space.project(pt - step_size * tangent_grad)
            new_err = fun(new_pt)
            converged = space.length(pt, new_pt) < pt_tol
            while new_err > fval and not converged:
                step_size = step_size / 2
                new_pt = space.project(pt - step_size * tangent_grad)
                new_err = fun(new_pt)
                converged = space.length(pt, new_pt) < pt_tol
            if converged:
                return new_pt if new_err < fval else pt.detach(), OptimResult.POINT_CONVERGED
            else:
                pt[:] = new_pt

    # Max iterations reached – return final value of 'pt' along with converged=True
    return pt.detach(), OptimResult.MAX_STEPS_REACHED
