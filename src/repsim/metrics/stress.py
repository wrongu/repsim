import torch
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.kernels import DEFAULT_KERNEL
from repsim.util import upper_triangle
from repsim import pairwise


class Stress(RepresentationMetricSpace, RiemannianSpace):
    """Mean squared difference in pairwise distances, AKA 'stress' from the MDS literature."""

    def __init__(self, m, kernel=None, rescale=False):
        super().__init__(dim=m * (m - 1) / 2, shape=(m, m))
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL
        self._rescale = rescale

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,m) matrix of Euclidean distances between xs
        using self._kernel."""
        if x.shape[0] != self.shape[0]:
            raise ValueError(
                f"Expected x to be size ({self.shape[0]}, ?) but is size {x.shape}"
            )
        pairwise_dist = pairwise.euclidean(x, kernel=self._kernel)

        # TODO - we should treat Stress as a spherical metric when _rescale=True since this acts
        #  as a constraint
        if self._rescale:
            # When 'rescale' flag is set, pairwise distances are normalized so that their median
            # distance is 1. This rescaling step can be used to make Stress scale-invariant even
            # when using a Linear kernel.
            pairwise_dist = pairwise_dist / torch.median(upper_triangle(pairwise_dist))

        return pairwise_dist

    def string_id(self) -> str:
        return f"Stress.{self._kernel.string_id()}{'.scaled' if self._rescale else ''}.{self.shape[0]}"

    @property
    def is_spherical(self) -> bool:
        return False

    #################################
    # Implement LengthSpace methods #
    #################################

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # sqrt of mean squared difference (euclidean, scaled by 2/m(m-1)) in distances from
        # upper-triangle of RDMs
        diff_in_dist = upper_triangle(pt_a - pt_b)
        return torch.sqrt(torch.mean(diff_in_dist**2))

    def _project_impl(self, pt: Point) -> Point:
        # Ensure symmetric
        pt = (pt + pt.T) / 2
        # Ensure nonnegative entries
        pt = torch.clip(pt, min=0.0, max=None)
        # Ensure diagonal is zero
        pt[torch.arange(self.m), torch.arange(self.m)] = 0.0
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test nonnegative
        if not torch.all(pt >= 0.0):
            return False
        # Test symmetric
        if not torch.allclose(pt, pt.T):
            return False
        # Test diagonal is zero
        if not torch.allclose(torch.diag(pt), pt.new_zeros((self.m,)), atol=atol):
            return False
        return True

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        # Stress geodesics are linear in the space of RDMs
        return self.project((1 - frac) * pt_a + frac * pt_b)

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        # Find the nearest matrix to vec_w - in the Frobenius sense - that is symmetric and has
        # zero diagonal
        return (vec_w + vec_w.T) / 2 * (1.0 - torch.eye(self.m))

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        # The usual inner-product in the ambient space, with a twist: since we defined
        # _length_impl using only the upper triangle of RDMs, we need to do the same here.
        return torch.mean(upper_triangle(vec_w * vec_v))

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # Stress = Euclidean in the space of distance matrices... just add w to a
        return pt_a + vec_w

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # Stress = Euclidean in the space of distance matrices... just subtract a from b
        return pt_b - pt_a

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Stress = Euclidean in the space of distance matrices... transport is a no-op
        return vec_w.clone()
