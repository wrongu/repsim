import torch
from repsim.geometry.trig import slerp
from repsim.kernels import center
from repsim.geometry.manifold import (
    SymmetricMatrix,
    SPDMatrix,
    DistMatrix,
    Point,
    Scalar,
)
from repsim import pairwise
from repsim.kernels import DEFAULT_KERNEL
from repsim.util import upper_triangle
from repsim.util import MetricType, CompareType


# Typing hints: neural data of size (n, d)
NeuralData = torch.Tensor


class RepresentationMetricSpace(SymmetricMatrix):
    """Base mixin class for all representational similarity/representational distance comparisons. Subclasses will
    inherit from *both* RepresentationMetricSpace and *one of* SPDMatrix or DistMatrix.
    """

    def __init__(self, n: int, kernel=None):
        super(RepresentationMetricSpace, self).__init__(rows=n)
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL()

    @property
    def metric_type(self) -> MetricType:
        raise NotImplementedError("type must be specified by a subclass")

    @property
    def compare_type(self) -> CompareType:
        raise NotImplementedError("compare_type must be specified by a subclass")

    def to_rdm(self, x: NeuralData) -> Point:
        """Convert (n,d) sized neural data into (n,n) pairwise comparison (representational distance) matrix, where the
        latter is a Point in the metric space.
        """
        return pairwise.compare(x, kernel=self._kernel, type=self.compare_type)

    def representational_distance(self, x: torch.Tensor, y: torch.Tensor) -> Scalar:
        return self.length(self.to_rdm(x), self.to_rdm(y))

    def string_id(self):
        return f"{self.__class__.__name__}.{self._kernel.string_id()}.{self.shape[0]}"


class AngularCKA(RepresentationMetricSpace, SPDMatrix):
    """Compute the angular distance between two representations x and y using the arccos(CKA) method described in the
    supplement of Williams et al (2021)

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS. http://arxiv.org/abs/2110.14739
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ANGLE

    @property
    def compare_type(self) -> CompareType:
        return CompareType.INNER_PRODUCT

    def length(self, rdm_x: Point, rdm_y: Point) -> Scalar:
        # Note: use clipping in case of numerical imprecision. arccos(1.00000000001) will give NaN!
        return torch.arccos(torch.clip(cka(rdm_x, rdm_y), -1.0, 1.0))

    def _has_implemented_closed_form_geodesic(self) -> bool:
        return True

    def geodesic_from(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        """
        Compute the geodesic between two points pt_a and pt_b.

        The main idea is to use SLERP, since AngularCKA is an arc length on the unit hypersphere of centered RDMs.
        """
        # Apply centering to each RDM so that inner-products alone are HSIC
        ctr_a, ctr_b = center(pt_a), center(pt_b)
        # Note: slerp normalizes for us, so the returned point will have unit norm even if ctr_a and ctr_b don't
        return self.project(slerp(ctr_a, ctr_b, frac))


class Stress(RepresentationMetricSpace, DistMatrix):
    """Difference-in-pairwise-distance, AKA 'stress' from the MDS literature."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LENGTH

    @property
    def compare_type(self) -> CompareType:
        return CompareType.DISTANCE

    def length(self, rdm_x: Point, rdm_y: Point) -> Scalar:
        diff_in_dist = upper_triangle(rdm_x - rdm_y)
        return torch.sqrt(torch.mean(diff_in_dist**2))

    def _has_implemented_closed_form_geodesic(self) -> bool:
        return True

    def geodesic_from(self, pt_a: Point, pt_b: Point, frac: float = 0.5):
        """Compute the geodesic between two points pt_a and pt_b.
        """
        # Stress geodesics are Euclidean in the space of RDMs
        return self.project((1 - frac) * pt_a + frac * pt_b)


class ScaleInvariantStress(RepresentationMetricSpace, DistMatrix):
    """Like 'Stress', but applies an isotropic rescaling operation on all distances first"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LENGTH

    @property
    def compare_type(self) -> CompareType:
        return CompareType.DISTANCE

    @staticmethod
    def _rescale(rdm_x: Point):
        """Rescale the given RDM so that the median pairwise distance (using the given kernel) is 1.0.

        Note that this is *not* the same as rescaling the underlying data before applying the kernel. For that, use a
        kernel with length_scale set to 'auto'.
        """
        upper_x = upper_triangle(rdm_x)
        return rdm_x / torch.median(upper_x)

    def length(self, rdm_x: Point, rdm_y: Point) -> Scalar:
        rescaled_x, rescaled_y = ScaleInvariantStress._rescale(rdm_x), ScaleInvariantStress._rescale(rdm_y)
        # The following is identical to Stress.length on the rescaled RDMs
        diff_in_dist = upper_triangle(rescaled_x - rescaled_y)
        return torch.sqrt(torch.mean(diff_in_dist**2))

    def _has_implemented_closed_form_geodesic(self) -> bool:
        return True

    def geodesic_from(self, pt_a: Point, pt_b: Point, frac: float = 0.5):
        """Compute the geodesic between two points pt_a and pt_b.
        """
        # Euclidean interpolation between rescaled RDMs
        rescaled_a, rescaled_b = ScaleInvariantStress._rescale(pt_a), ScaleInvariantStress._rescale(pt_b)
        return self.project((1 - frac) * rescaled_a + frac * rescaled_b)


class AffineInvariantRiemannian(RepresentationMetricSpace, SPDMatrix):
    """Compute the 'affine-invariant Riemannian metric', as advocated for by [1].

    NOTE: given (n,d) sized inputs, this involves inverting a (n,n)-sized matrix, which might be rank-deficient. The
    authors of [1] got around this by switching the inner-product to be across conditions, and compared (d,d)-sized
    matrices. However, this no longer suffices as a general RSA tool, since in general d_x will not equal d_y.

    We get around this by regularizing the n by n matrix, shrinking it towards its diagonal (see Yatsenko et al (2015))

    [1] Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian manifold to compare
        representations in brain and in models. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118271
    """

    def __init__(self, **kwargs):
        shrinkage = kwargs.pop("shrinkage", 0.1)
        super().__init__(**kwargs)
        if shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError(
                "Shrinkage parameter must be in [0,1], where 0 means no regularization."
            )
        self._shrink = shrinkage

    @property
    def metric_type(self) -> MetricType:
        return MetricType.RIEMANN

    @property
    def compare_type(self) -> CompareType:
        return CompareType.INNER_PRODUCT

    def length(self, rdm_x: Point, rdm_y: Point) -> Scalar:
        n = rdm_x.size()[0]
        # Apply shrinkage regularizer: down-weight all off-diagonal parts of each RSM by self._shrink.
        off_diag_n = 1.0 - torch.eye(n, device=rdm_x.device, dtype=rdm_x.dtype)
        rdm_x = rdm_x - self._shrink * off_diag_n * rdm_x
        rdm_y = rdm_y - self._shrink * off_diag_n * rdm_y
        if (
            torch.linalg.matrix_rank(rdm_x) < self.shape[0]
            or torch.linalg.matrix_rank(rdm_y) < self.shape[0]
        ):
            raise ValueError(
                f"Cannot invert rank-deficient RDMs – set shrink > 0 and/or use a kernel!"
            )
        # Compute rdm_x^{-1} @ rdm_y
        x_inv_y = torch.linalg.solve(rdm_x, rdm_y)
        log_eigs = torch.log(torch.linalg.eigvals(x_inv_y).real)
        return torch.sqrt(torch.sum(log_eigs**2))


def hsic(
    k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True
) -> torch.Tensor:
    """Compute Hilbert-Schmidt Independence Criteron (HSIC)

    :param k_x: n by n values of kernel applied to all pairs of x data
    :param k_y: n by n values of kernel on y data
    :param centered: whether or not at least one kernel is already centered
    :param unbiased: if True, use unbiased HSIC estimator of Song et al (2007), else use original estimator of Gretton et al (2005)
    :return: scalar score in [0*, inf) measuring dependence of x and y

    * note that if unbiased=True, it is possible to get small values below 0.
    """
    if k_x.size() != k_y.size():
        raise ValueError(
            "RDMs must have the same size, but got {} and {}".format(
                k_x.size(), k_y.size()
            )
        )
    n = k_x.size()[0]

    if not centered:
        k_y = center(k_y)

    if unbiased:
        # Remove the diagonal
        k_x = k_x * (1 - torch.eye(n, device=k_x.device, dtype=k_x.dtype))
        k_y = k_y * (1 - torch.eye(n, device=k_y.device, dtype=k_y.dtype))
        # Equation (4) from Song et al (2007)
        return (
            (k_x * k_y).sum()
            - 2 * (k_x.sum(dim=0) * k_y.sum(dim=0)).sum() / (n - 2)
            + k_x.sum() * k_y.sum() / ((n - 1) * (n - 2))
        ) / (n * (n - 3))
    else:
        # The original estimator from Gretton et al (2005)
        return torch.sum(k_x * k_y) / (n - 1) ** 2


def cka(
    k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True
) -> torch.Tensor:
    """Compute Centered Kernel Alignment (CKA).

    :param k_x: n by n values of kernel applied to all pairs of x data
    :param k_y: n by n values of kernel on y data
    :param centered: whether or not at least one kernel is already centered
    :param unbiased: if True, use unbiased HSIC estimator of Song et al (2007), else use original estimator of Gretton et al (2005)
    :return: scalar score in [0*, 1] measuring normalized dependence between x and y.

    * note that if unbiased=True, it is possible to get small values below 0.
    """
    hsic_xy = hsic(k_x, k_y, centered, unbiased)
    hsic_xx = hsic(k_x, k_x, centered, unbiased)
    hsic_yy = hsic(k_y, k_y, centered, unbiased)
    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)


__all__ = [
    "RepresentationMetricSpace",
    "AngularCKA",
    "Stress",
    "AffineInvariantRiemannian",
]
