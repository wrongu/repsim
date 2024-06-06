import numpy as np
from typing import Iterable
from . import Point
from .manifold import RiemannianSpace
from .optimize import minimize
import warnings


class IterativeFrechetMean(object):
    """The Frechet Mean is the generalization of 'mean' to data on a manifold. Here, the manifold is
    a sphere. In other words, we're computing the point on the sphere that minimizes the sum of
    squared distances to all rows of X.

    Algorithm iteratively refines an estimate of the mean, analogous to other familiar 'running
    mean' algorithms. Result is only approximate and depends on data order, so it is encouraged that
    calls to update() are shuffled.
    """

    def __init__(self, space: RiemannianSpace):
        self.space = space
        self.n = 0
        self.mean = None

    def update(self, x):
        self.n += 1
        x = self.space.project(x)
        if self.mean is None:
            self.mean = x.clone()
        else:
            mean_to_x = self.space.log_map(self.mean, x)
            self.mean = self.space.exp_map(self.mean, mean_to_x / self.n)

    def reset(self, n=0):
        self.n = n


def iterate_frechet_mean(space: RiemannianSpace, points: Iterable[Point]) -> Point:
    """Convenience wrapper around IterativeFrechetMean."""
    ifm = IterativeFrechetMean(space)
    for pt in points:
        ifm.update(pt)
    return ifm.mean


def optimize_frechet_mean(
    space: RiemannianSpace, points: Iterable[Point], init_method: str = "random point"
) -> Point:
    """The Frechet Mean is the generalization of 'mean' to data on a manifold.

    Here, the manifold is a sphere. In other words, we're computing the point on the sphere that
    minimizes the sum of squared distances to all rows of X.
    """

    points = [space.project(x) for x in points]

    if init_method == "random point":
        init = np.random.choice(points).clone()
    elif init_method == "euclidean":
        init = space.project(sum(points) / len(points))
    elif init_method == "iterative":
        init = iterate_frechet_mean(space, points)
    else:
        raise ValueError(f"Invalid initialization method: {init_method}")

    # loss function
    def sum_squared_distance(m):
        return sum([space.length(m, p) ** 2 for p in points])

    frechet_mean, converged = minimize(space, sum_squared_distance, init, max_iter=1000)

    if not converged:
        warnings.warn("minimize() failed to converge!")

    return frechet_mean


__all__ = ["IterativeFrechetMean", "iterate_frechet_mean", "optimize_frechet_mean"]
