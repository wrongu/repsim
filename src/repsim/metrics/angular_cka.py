import torch
import numpy as np
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.geometry.trig import slerp
from repsim.kernels import center, is_centered, DEFAULT_KERNEL
from repsim import pairwise


class AngularCKA(RepresentationMetricSpace, RiemannianSpace):
    """Compute the angular distance between two representations x and y using the arccos(CKA) method
    described in the supplement of Williams et al (2021).

    This metric is equivalent to (i) computing the Gram matrix of inner products of neural data
    (possibly with a kernel), (ii) centering the matrix, and (iii) normalizing it to unit Frobenius
    norm. Then, AngularCKA is arc length on the hypersphere with normalized-and-centered Gram
    matrices

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics
    on Neural     Representations. NeurIPS.
    https://arxiv.org/abs/2110.14739
    """

    def __init__(self, m, kernel=None, use_unbiased_hsic=True):
        super().__init__(dim=m * (m + 1) / 2 - 1, shape=(m, m))
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL
        self._unbiased = use_unbiased_hsic

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,m) Gram matrix of inner products between xs
        using self._kernel."""
        if x.shape[0] != self.shape[0]:
            raise ValueError(
                f"Expected x to be size ({self.shape[0]}, ?) but is size {x.shape}"
            )

        centered_gram_matrix = center(pairwise.inner_product(x, kernel=self._kernel))
        if self._unbiased:
            # What makes this 'unbiased' is, essentially, that we get rid of all of the k(x,
            # x) terms in the Gram matrix, leaving only the k(x,x') terms for x!=x'. This makes
            # the inner product inside of CKA depend only on independent samples of x,
            # x'. However, once we mask the centered data it is no longer centered. Hence the
            # _contains_impl() and _project_impl() functions are also modified so that
            # centeredness is only asserted in the biased case, and having all-zero diagonal
            # entries is only asserted in the unbiased case.
            mask = torch.ones_like(centered_gram_matrix) - torch.eye(
                self.m, dtype=x.dtype, device=x.device
            )
            return _matrix_unit_norm(centered_gram_matrix * mask)
        else:
            return _matrix_unit_norm(centered_gram_matrix)

    def string_id(self) -> str:
        bias_str = ".unb" if self._unbiased else ""
        return f"AngularCKA{bias_str}.{self._kernel.string_id()}.{self.m}"

    @property
    def is_spherical(self) -> bool:
        return True

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        assert pt.shape == (
            self.m,
            self.m,
        ), f"Input to AngularCKA.project() must be a m by m matrix but is size {pt.shape}!"
        # 1. Ensure matrix is symmetric
        pt = (pt + pt.T) / 2
        if self._unbiased:
            # 2a. Normally we would enforce positive (semi-) definiteness, but we can't guarantee
            # that here due to the masking of the diagonal of the pt matrix
            pass
            # 3a. If the unbiased flag is set, we cannot risk calling center() a second time on
            # the same data, since this would be essentially center(mask * center(data)),
            # and the outer center will interact with the mask
            mask = torch.ones_like(pt) - torch.eye(
                self.m, dtype=pt.dtype, device=pt.device
            )
            pt = _matrix_unit_norm(mask * pt)
        else:
            # 2b. Ensure matrix is positive (semi-) definite by clipping its eigenvalues
            e, v = torch.linalg.eigh(pt)
            e = torch.clip(e, min=0.0, max=None)
            pt = v @ torch.diag(e) @ v.T
            # 3b. If the unbiased flag is not set, we can be extra careful here and both center
            # and normalize. Centering twice is the same as centering once, so we don't need to
            # worry whether pt is already centered.
            pt = _matrix_unit_norm(center(pt))
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.m):
            return False
        # Test symmetric
        if not torch.allclose(pt, pt.T, atol=atol):
            return False
        if self._unbiased:
            # Test that the diagonal is zero
            if not torch.allclose(torch.diag(pt), pt.new_zeros((self.m,)), atol=atol):
                return False
        else:
            # Test positive (semi-) definiteness
            e = torch.linalg.eigvalsh(pt)
            if not all(e >= -atol):
                return False
            # Test that the matrix is centered
            if not is_centered(pt, atol=atol):
                return False
        # Test unit Frobenius norm
        if not torch.isclose(
            torch.linalg.norm(pt, ord="fro"), pt.new_ones((1,)), atol=atol
        ):
            return False
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # Assume pt_a and pt_b pass all tests in self.contains(), i.e. they are normalized and
        # centered Gram matrices Note that no denominator is needed here because of the
        # normalization step already included in neural_data_to_point() and project()
        cka = torch.sum(pt_a * pt_b)
        # Clipping because arccos(1.00000000001) gives NaN, and some numerical error can cause
        # that to happen
        return torch.arccos(torch.clip(cka, -1.0, 1.0))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        """Compute the geodesic between two points pt_a and pt_b.

        The main idea is to use SLERP, since AngularCKA is an arc length on the unit hypersphere of
        centered Gram matrices.
        """
        # TODO - ideally we would compute CKA using the unbiased HSIC, but then what does that do
        #  to the geodesic?
        # Note: slerp normalizes for us, so the returned point will have unit norm even if ctr_a
        # and ctr_b don't
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
        norm_w = unscaled_w / torch.clip(
            torch.sqrt(torch.sum(unscaled_w * unscaled_w)), 1e-7
        )
        return norm_w * self._length_impl(pt_a, pt_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Refer to Hypersphere.levi_civita
        vec_v = self.log_map(pt_a, pt_b)
        angle = self._length_impl(pt_a, pt_b)
        unit_v = vec_v / torch.clip(
            angle, 1e-7
        )  # the length of tangent vector v *is* the length from a to b
        w_along_v = torch.sum(unit_v * vec_w)
        orth_part = vec_w - w_along_v * unit_v
        return (
            orth_part
            + torch.cos(angle) * w_along_v * unit_v
            - torch.sin(angle) * w_along_v * pt_a
        )


def _matrix_unit_norm(A):
    """Scale a matrix to have unit Frobenius norm.

    :param A: a matrix
    :return: the matrix divided by its Frobenius norm
    """
    return A / torch.linalg.norm(A, ord="fro")


__all__ = ["AngularCKA"]
