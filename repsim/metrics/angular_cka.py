import torch
import numpy as np
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.geometry.trig import slerp
from repsim.kernels import center, DEFAULT_KERNEL
from repsim import pairwise


class AngularCKA(RepresentationMetricSpace, RiemannianSpace):
    """Compute the angular distance between two representations x and y using the arccos(CKA) method described in the
    supplement of Williams et al (2021).

    This metric is equivalent to (i) computing the Gram matrix of inner products of neural data (possibly with a
    kernel), (ii) centering the matrix, and (iii) normalizing it to unit Frobenius norm. Then, AngularCKA is arc length
    on the hypersphere with normalized-and-centered Gram matrices

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS. https://arxiv.org/abs/2110.14739
    """

    def __init__(self, m, kernel=None):
        super().__init__(dim=m*(m+1)/2-1, shape=(m, m))
        self.m = m
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (n,d) neural data to a size (n,n) Gram matrix of inner products between xs using self._kernel.
        """
        if x.shape[0] != self.shape[0]:
            raise ValueError(f"Expected x to be size ({self.shape[0]}, ?) but is size {x.shape}")

        centered_gram_matrix = center(pairwise.inner_product(x, kernel=self._kernel))
        return _matrix_unit_norm(centered_gram_matrix)

    def string_id(self) -> str:
        return f"AngularCKA.{self._kernel.string_id()}.{self.m}"

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        assert pt.shape == (self.m, self.m), \
            f"Input to AngularCKA.project() must be a m by m matrix but is size {pt.shape}!"
        # 1. Ensure matrix is symmetric
        pt = (pt + pt.T) / 2
        # 2. Ensure matrix is positive (semi-) definite by clipping its eigenvalues
        e, v = torch.linalg.eigh(pt)
        e = torch.clip(e, min=0., max=None)
        pt = v @ torch.diag(e) @ v.T
        # 3. Center and normalize
        pt = _matrix_unit_norm(center(pt))
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.m):
            return False
        # Test symmetric
        if not torch.allclose(pt, pt.T, atol=atol):
            return False
        # Test positive (semi-) definiteness
        e = torch.linalg.eigvalsh(pt)
        if not all(e >= -atol):
            return False
        # Test centered; TODO - can we test this without needing to call center()?
        if not torch.allclose(pt, center(pt), atol=atol):
            return False
        # Test unit Frobenius norm
        if not torch.isclose(torch.linalg.norm(pt, ord='fro'), pt.new_ones((1,)), atol=atol):
            return False
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # Assume pt_a and pt_b pass all tests in self.contains(), i.e. they are normalized and centered Gram matrices
        # TODO - ideally we would compute CKA using the unbiased HSIC, but then what does that do to the geodesic?
        cka = torch.sum(pt_a * pt_b)
        # Clipping because arccos(1.00000000001) gives NaN, and some numerical error can cause that to happen
        return torch.arccos(torch.clip(cka, -1.0, 1.0))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        """Compute the geodesic between two points pt_a and pt_b.

        The main idea is to use SLERP, since AngularCKA is an arc length on the unit hypersphere of centered Gram matrices.
        """
        # TODO - ideally we would compute CKA using the unbiased HSIC, but then what does that do to the geodesic?
        # Note: slerp normalizes for us, so the returned point will have unit norm even if ctr_a and ctr_b don't
        return self.project(slerp(pt_a, pt_b, frac))

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        # Identical to Hypersphere.to_tangent (note that pt_a is already a unit-norm matrix)
        dot_a_w = torch.sum(pt_a * vec_w)
        return vec_w - dot_a_w * pt_a

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        # No special sauce required -- inner product is just the usual in the ambient space
        return torch.sum(vec_w * vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # Identical to Hypersphere.exp_map
        # See https://math.stackexchange.com/a/1930880
        vec_w = self.to_tangent(pt_a, vec_w)
        norm = torch.sqrt(torch.sum(vec_w * vec_w))
        c1 = torch.cos(norm)
        c2 = torch.sinc(norm / np.pi)
        return c1 * pt_a + c2 * vec_w

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # Identical to Hypersphere.log_map
        unscaled_w = self.to_tangent(pt_a, pt_b)
        norm_w = unscaled_w / torch.sqrt(torch.sum(unscaled_w * unscaled_w))
        return norm_w * self.length(pt_a, pt_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Refer to Hypersphere.levi_civita
        vec_v = self.log_map(pt_a, pt_b)
        angle = self.length(pt_a, pt_b)
        unit_v = vec_v / angle  # the length of tangent vector v *is* the length from a to b
        w_along_v = torch.sum(unit_v * vec_w)
        orth_part = vec_w - w_along_v * unit_v
        return orth_part + torch.cos(angle) * w_along_v * unit_v - torch.sin(angle) * w_along_v * pt_a


def _matrix_unit_norm(A):
    """Scale a matrix to have unit Frobenius norm.

    :param A: a matrix
    :return: the matrix divided by its Frobenius norm
    """
    return A / torch.linalg.norm(A, ord='fro')


__all__ = ["AngularCKA"]
