import tensorly as tl
import numpy as np
from repsim.kernels import Kernel, center, Linear
from repsim import pairwise
from repsim.util import upper_triangle, corrcoef
from repsim.util import MetricType, CompareType, CorrType
from typing import Union

e = np.e


class BaseRepSim(object):
    """
    Abstract base class for all representational similarity/representational
    distance comparisons.

    """

    @property
    def type(self) -> MetricType:
        raise NotImplementedError("type must be specified by a subclass")

    def compare(
        self,
        x: tl.tensor,
        y: tl.tensor,
        *,
        kernel_x: Union[Kernel, None] = None,
        kernel_y: Union[Kernel, None] = None,
    ) -> tl.tensor:
        raise NotImplementedError("compare() must be implemented by a subclass")


class GeneralizedShapeMetric(BaseRepSim):
    """
    Compute the 'generalized shape metric' between two representations x and y
    using the 'angle' method described by Williams et al (2021)

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021).
    Generalized Shape Metrics on Neural Representations. NeurIPS.
    http://arxiv.org/abs/2110.14739
    """

    @property
    def type(self) -> MetricType:
        return MetricType.ANGLE

    def compare(
        self,
        x: tl.tensor,
        y: tl.tensor,
        *,
        kernel_x: Union[Kernel, None] = None,
        kernel_y: Union[Kernel, None] = None,
    ) -> tl.tensor:
        rsm_x = pairwise.compare(x, type=CompareType.INNER_PRODUCT, kernel=kernel_x)
        rsm_y = pairwise.compare(y, type=CompareType.INNER_PRODUCT, kernel=kernel_y)
        # Note: use clipping in case of numerical imprecision. arccos(1.00000000001) will give NaN!
        return np.arccos(tl.clip(cka(rsm_x, rsm_y), -1.0, 1.0))


class Stress(BaseRepSim):
    """Difference-in-pairwise-distance, AKA 'stress' from the MDS literature."""

    @property
    def type(self):
        return MetricType.LENGTH

    def compare(
        self,
        x: tl.tensor,
        y: tl.tensor,
        *,
        kernel_x: Union[Kernel, None] = None,
        kernel_y: Union[Kernel, None] = None,
    ) -> tl.tensor:
        rdm_x = pairwise.compare(x, type=CompareType.DISTANCE, kernel=kernel_x)
        rdm_y = pairwise.compare(y, type=CompareType.DISTANCE, kernel=kernel_y)
        diff_in_dist = upper_triangle(rdm_x - rdm_y)
        return tl.sqrt(tl.mean(diff_in_dist**2))


class Corr(BaseRepSim):
    """Correlation between RDMs; this is 'classic' RSA when correlation type is 'spearman'"""

    def __init__(
        self,
        corr_type: CorrType = CorrType.SPEARMAN,
        cmp_type: CompareType = CompareType.INNER_PRODUCT,
    ):
        self._corr_type = corr_type
        self._cmp_type = cmp_type

    @property
    def type(self):
        return MetricType.CORR

    def compare(
        self,
        x: tl.tensor,
        y: tl.tensor,
        *,
        kernel_x: Union[Kernel, None] = None,
        kernel_y: Union[Kernel, None] = None,
    ) -> tl.tensor:
        rsm_x = pairwise.compare(x, type=self._cmp_type, kernel=kernel_x)
        rsm_y = pairwise.compare(y, type=self._cmp_type, kernel=kernel_y)
        return corrcoef(
            upper_triangle(rsm_x), upper_triangle(rsm_y), type=self._corr_type
        )


class AffineInvariantRiemannian(BaseRepSim):
    """Compute the 'affine-invariant Riemannian metric', as advocated for by [1].

    NOTE: given (n,d) sized inputs, this involves inverting a (n,n)-sized matrix, which might be rank-deficient. The
    authors of [1] got around this by switching the inner-product to be across conditions, and compared (d,d)-sized
    matrices. However, this no longer suffices as a general RSA tool, since in general d_x will not equal d_y.

    We get around this by regularizing the n by n matrix, shrinking it towards its diagonal (see Yatsenko et al (2015))

    [1] Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian manifold to compare
        representations in brain and in models. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118271
    """

    def __init__(self, shrinkage=0.1):
        super(AffineInvariantRiemannian, self).__init__()
        if shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError(
                "Shrinkage parameter must be in [0,1], where 0 means no regularization."
            )
        self._shrink = shrinkage

    @property
    def type(self) -> MetricType:
        return MetricType.RIEMANN

    def compare(
        self,
        x: tl.tensor,
        y: tl.tensor,
        *,
        kernel_x: Union[Kernel, None] = None,
        kernel_y: Union[Kernel, None] = None,
    ) -> tl.tensor:
        n, d = x.size()
        if self._shrink == 0.0 and (
            kernel_x is None
            or kernel_y is None
            or isinstance(kernel_x, Linear)
            or isinstance(kernel_y, Linear)
        ):
            # Linear kernels produce low-rank RSMs, which invalidates 'linalg.solve', giving negative eigenvalues
            # and other difficulty. Error out if we're trying to invert a rank-deficient RSM
            if n > d:
                raise ValueError(
                    f"Since x is size {(n, d)} and shrinkage is {self._shrink}, the Linear kernel will result in a rank-deficient RSM!"
                )
        rsm_x = center(
            pairwise.compare(x, type=CompareType.INNER_PRODUCT, kernel=kernel_x)
        )
        rsm_y = center(
            pairwise.compare(y, type=CompareType.INNER_PRODUCT, kernel=kernel_y)
        )
        # Apply shrinkage regularizer: down-weight all off-diagonal parts of each RSM by self._shrink.
        off_diag_n = 1.0 - tl.eye(n, device=rsm_x.device, dtype=rsm_x.dtype)
        rsm_x -= self._shrink * off_diag_n * rsm_x
        rsm_y -= self._shrink * off_diag_n * rsm_y
        # Compute rsm_x^{-1} @ rsm_y
        x_inv_y = tl.solve(rsm_x, rsm_y)
        eigs = tl.eigh(x_inv_y)[0].real
        log_eigs = tl.log2(eigs) / tl.log2(e)
        return tl.sqrt(tl.sum(log_eigs**2))


def hsic(
    k_x: tl.tensor, k_y: tl.tensor, centered: bool = False, unbiased: bool = True
) -> tl.tensor:
    """Compute Hilbert-Schmidt Independence Criteron (HSIC)

    :param k_x: n by n values of kernel applied to all pairs of x data
    :param k_y: n by n values of kernel on y data
    :param centered: whether or not at least one kernel is already centered
    :param unbiased: if True, use unbiased HSIC estimator of Song et al (2007), else use original estimator of Gretton et al (2005)
    :return: scalar score in [0*, inf) measuring dependence of x and y

    * note that if unbiased=True, it is possible to get small values below 0.
    """
    if k_x.size() != k_y.size():
        raise ValueError("RDMs must have the same size!")
    n = k_x.size()[0]

    if not centered:
        k_y = center(k_y)

    if unbiased:
        # Remove the diagonal
        k_x = k_x * (1 - tl.eye(n, device=k_x.device, dtype=k_x.dtype))
        k_y = k_y * (1 - tl.eye(n, device=k_y.device, dtype=k_y.dtype))
        # Equation (4) from Song et al (2007)
        return (
            (k_x * k_y).sum()
            - 2 * (k_x.sum(dim=0) * k_y.sum(dim=0)).sum() / (n - 2)
            + k_x.sum() * k_y.sum() / ((n - 1) * (n - 2))
        ) / (n * (n - 3))
    else:
        # The original estimator from Gretton et al (2005)
        return tl.sum(k_x * k_y) / (n - 1) ** 2


def cka(
    k_x: tl.tensor, k_y: tl.tensor, centered: bool = False, unbiased: bool = True
) -> tl.tensor:
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
    return hsic_xy / tl.sqrt(hsic_xx * hsic_yy)


__all__ = [
    "BaseRepSim",
    "GeneralizedShapeMetric",
    "Stress",
    "Corr",
    "AffineInvariantRiemannian",
]
