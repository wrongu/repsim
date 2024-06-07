import abc
import torch
import warnings
from repsim.util import prod
from repsim.geometry.optimize import OptimResult, minimize
from typing import Optional

# Typing hints... ideally we would specify sizes here, but can't do that with the current type system
Point = torch.Tensor
Scalar = torch.Tensor
Vector = torch.Tensor


class LengthSpace(abc.ABC):
    """Base class for metric spaces with an associated length(). A metric space (X, d) is any space
    that satisfies.

    (i) the identity property, or d(pt_a, pt_a) = 0
    (ii) the symmetry property, or d(pt_a, pt_b) = d(pt_b, pt_a)
    (iii) the triangle inequality, or d(pt_a, pt_c) <= d(pt_a, pt_b) + d(pt_b, pt_c)

    And a *length* space is a metric space with the additional property that distances describe the
    length of a path:
    (iv) d(pt_a, pt_b) = sum_i^N d(x_{i-1}, x_i) along a shortest path where x_0=pt_a and x_N=pt_b

    Properties of a LengthSpace include
    - dim : the dimensionality of a manifold. e.g. a sphere in 3d has dim=2
    - shape : tuple describing the tensor shape of points, as in numpy.ndarray.shape
    - ambient : the ambient dimensionality, equal to prod(shape). E.g. a sphere in 3d has ambient=3
    """

    def __init__(self, *, dim: int, shape: tuple):
        self.dim = dim
        self._shape = shape
        self.ambient = prod(shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape
        self.ambient = prod(new_shape)

    def project(self, pt: Point) -> Point:
        """Project a point from the ambient space onto the manifold.

        :param pt: a point in the ambient space
        :return: a point on the manifold that is 'as close as possible' to pt
        """
        return self._project_impl(pt) if not self.contains(pt) else pt.clone()

    @abc.abstractmethod
    def _project_impl(self, pt: Point) -> Point:
        """Implementation of project() without checking contains() first."""

    def contains(self, pt: Point, atol: float = 1e-6) -> bool:
        """Check whether the given point is within 'atol' tolerance of the manifold.

        LengthSpace.contains checks match to ambient shape only. Further specialization done by
        subclasses.
        """
        if pt.shape != self.shape:
            return False
        return self._contains_impl(pt, atol)

    @abc.abstractmethod
    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        """Implementation of contains() to check if a point is in the space.

        :param pt: a point in the ambient space
        :param atol: tolerance
        :return: True if the point is within atol of being in the space, and False otherwise
        """

    def length(self, pt_a: Point, pt_b: Point) -> Scalar:
        """Compute length from pt_a to pt_b, projecting them into the space first if needed.

        :param pt_a: starting point in the space
        :param pt_b: ending point in the space
        :return: scalar length (or 'distance' or 'metric') from a to b
        """
        if not self.contains(pt_a):
            warnings.warn("pt_a is not on the manifold - trying to project")
            pt_a = self._project_impl(pt_a)

        if not self.contains(pt_b):
            warnings.warn("pt_b is not on the manifold - trying to project")
            pt_b = self._project_impl(pt_b)

        return self._length_impl(pt_a, pt_b)

    @abc.abstractmethod
    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        """Implementation of length(pt_a, pt_b) without checking for contains() first.

        Subclasses should make this a differentiable function of both pt_a and pt_b.

        :param pt_a: starting point in the space
        :param pt_b: ending point in the space
        :return: scalar length (or 'distance' or 'metric') from a to b
        """

    def geodesic(
        self,
        pt_a: Point,
        pt_b: Point,
        init_pt: Optional[Point] = None,
        frac: float = 0.5,
        **kwargs,
    ) -> Point:
        """Compute a point along the (a) geodesic connecting pt_a to pt_b. In a basic LengthSpace,
        this falls back on numerical optimization. Subclasses that inherit from GeodesicLengthSpace
        are more efficient.

        :param pt_a: starting point of the geodesic
        :param pt_b: ending point of the geodesic
        :param init_pt: (optional) starting point for the search. Defaults to
               self.project(frac*pt_b + (1-frac)*pt_a)
        :param frac: fraction of distance from a to b
        :param **kwargs: configure optimization
        :return: a new Point, pt_c, such that
            1. it is on the geodesic, so length(pt_a, pt_c)+length(pt_c, pt_b) = length(pt_a, pt_b)
            2. it divides the total length by 'frac'; frac = length(pt_a, pt_c) / length(pt_a, pt_b)
        """
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"'frac' must be in [0, 1] but is {frac}")

        # Three cases where we can just break early without optimizing
        if frac == 0:
            return pt_a
        elif frac == 1:
            return pt_b
        elif torch.allclose(pt_a, pt_b, atol=kwargs.get("pt_tol", 1e-6)):
            return self.project((pt_a + pt_b) / 2)

        if init_pt is not None:
            pt = init_pt.clone()
        else:
            # Default initial guess to projection of euclidean interpolated point
            pt = self.project((1 - frac) * pt_a + frac * pt_b)

        def loss_fn(pt_c):
            """This loss function is minimized when pt_c is on the geodesic and 'frac' percent of
            the distance from a to b.

            The fact that it is quadratic in dist_ac and dist_bc makes it more numerically stable to
            work with in gradient descent.
            """
            dist_ac, dist_bc = self.length(pt_a, pt_c), self.length(pt_c, pt_b)
            return dist_ac * dist_ac * (1 - frac) + dist_bc * dist_bc * frac

        pt, status = minimize(self, loss_fn, pt, **kwargs)
        if status != OptimResult.CONVERGED:
            warnings.warn(
                f"Minimization failed to converge! Status is {status}. "
                f"Geodesic point may be unreliable"
            )
        return pt


class GeodesicLengthSpace(LengthSpace):
    """GeodesicLengthSpace is an abstract base class for LengthSpaces that additionally provide a
    closed-form function to compute points along a geodesic."""

    def geodesic(self, pt_a: Point, pt_b: Point, frac: float = 0.5, **kwargs) -> Point:
        """Compute a point along the (a) geodesic connecting pt_a to pt_b.

        :param pt_a: starting point of the geodesic
        :param pt_b: ending point of the geodesic
        :param frac: fraction of distance from a to b
        :return: a new Point, pt_c, such that
            1. it is on the geodesic, so length(pt_a, pt_c)+length(pt_c, pt_b) = length(pt_a, pt_b)
            2. it divides the total length by 'frac'; frac = length(pt_a, pt_c) / length(pt_a, pt_b)
        """
        if len(kwargs) > 0:
            warnings.warn(
                f"{self.__class__.__name__}.geodesic takes no kwargs, but got {kwargs.keys()}"
            )

        # Check some cases where we can take a shortcut
        if torch.allclose(pt_a, pt_b):
            return pt_a.clone()
        elif frac == 0.0:
            return pt_a.clone()
        elif frac == 1.0:
            return pt_b.clone()
        else:
            return self._geodesic_impl(pt_a, pt_b, frac)

    @abc.abstractmethod
    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        """Implementation of geodesic() without checks."""


class RiemannianSpace(GeodesicLengthSpace):
    """RiemannianSpace is an abstract base class for Riemannian Manifolds, which must be
    GeodesicLengthSpaces.

    Further, a Riemannian space provides functions for doing things with tangent spaces.
    """

    @abc.abstractmethod
    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        """Project a vector into the tangent space at pt_a.

        :param pt_a: point on the manifold
        :param vec_w: a vector in the ambient space whose base is at pt_a
        :return: projection of vec_w into the tangent space at pt_a. (If already in it, the returned
            vector is vec_w)
        """

    @abc.abstractmethod
    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        """Inner-product between two tangent vectors (defined at pt_a)

        :param pt_a: point defining the tangent space
        :param vec_w: first vector
        :param vec_v: second vector
        :return: inner product between w and v
        """

    def squared_norm(self, pt_a: Point, vec_w: Vector):
        """Compute squared norm of a tangent vector at a point.

        :param pt_a: point defining the tangent space
        :param vec_w: first vector
        :return: squared length of w according to the metric, AKA <w,w>
        """
        return self.inner_product(pt_a, vec_w, vec_w)

    def norm(self, pt_a: Point, vec_w: Vector):
        """Compute norm of a tangent vector at a point.

        :param pt_a: point defining the tangent space
        :param vec_w: first vector
        :return: length of w according to the metric
        """
        return torch.sqrt(self.squared_norm(pt_a, vec_w))

    @abc.abstractmethod
    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        """Compute exponential map, which intuitively means finding the point pt_b that you get
        starting from pt_a and moving in the direction vec_w, which must be in the tangent space of
        pt_a.

        :param pt_a: base point
        :param vec_w: tangent vector
        :return: pt_b, the point you get starting from pt_a and moving in the direction vec_w
        """

    @abc.abstractmethod
    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        """Compute logarithmic map, which can be thought of as the inverse of the exponential map.

        :param pt_a: base point. This defines where the tangent space is.
        :param pt_b: target point such that exp_map(pt_a, log_map(pt_a, pt_b)) = pt_b
        :return: vec_w, the vector in the tangent space at pt_a pointing in the direction (and
            magnitude) of pt_b
        """

    @abc.abstractmethod
    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        """Parallel-transport a tangent vector vec_w from pt_a to pt_b. The Levi-Civita connection
        is a nice way of defining "parallel lines" originating at two different places in a curved
        space. We say that vec_v at pt_b is parallel to vec_w at pt_a if, locally at b,
        levi_civita(pt_a, pt_b, vec_w) is colinear with vec_v.

        :param pt_a: base point where vec_w is a tangent vector
        :param pt_b: target point to transport to.
        :param vec_w: the tangent vector at pt_a to be transported to pt_b
        :return: vec_v, a vector in the tangent space of pt_b, corresponding to the parallel
            transport of vec_w
        """


__all__ = [
    "Point",
    "Scalar",
    "Vector",
    "LengthSpace",
    "GeodesicLengthSpace",
    "RiemannianSpace",
]
