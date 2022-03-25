import torch
from repsim.kernels import Kernel
from repsim.pairwise import compare
from repsim.util import upper_triangle, corrcoef
from repsim.util import MetricType, CompareType, CorrType
from typing import Union


class BaseRepSim(object):
    """Abstract base class for all representational similarity/representational distance comparisons.
    """
    @property
    def type(self) -> MetricType:
        raise NotImplementedError('type must be specified by a subclass')

    def compare(self, x: torch.Tensor, y: torch.Tensor, *, kernel_x: Union[Kernel, None] = None, kernel_y: Union[Kernel, None] = None) -> torch.Tensor:
        raise NotImplementedError('compare() must be implemented by a subclass')


class GeneralizedShapeMetric(BaseRepSim):
    """Compute the 'generalized shape metric' between two representations x and y using the 'angle' method described by
    Williams et al (2021)

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS. http://arxiv.org/abs/2110.14739
    """
    @property
    def type(self) -> MetricType:
        return MetricType.ANGLE

    def compare(self, x: torch.Tensor, y: torch.Tensor, *, kernel_x: Union[Kernel, None] = None, kernel_y: Union[Kernel, None] = None) -> torch.Tensor:
        rsm_x = compare(x, type=CompareType.INNER_PRODUCT, kernel=kernel_x)
        rsm_y = compare(y, type=CompareType.INNER_PRODUCT, kernel=kernel_y)
        return torch.arccos(cka(rsm_x, rsm_y))


class Stress(BaseRepSim):
    """Difference-in-pairwise-distance, AKA 'stress' from the MDS literature.
    """
    @property
    def type(self):
        # TODO - is this a length or metric?
        return MetricType.LENGTH

    def compare(self, x: torch.Tensor, y: torch.Tensor, *, kernel_x: Union[Kernel, None] = None, kernel_y: Union[Kernel, None] = None) -> torch.Tensor:
        rdm_x = compare(x, type=CompareType.DISTANCE, kernel=kernel_x)
        rdm_y = compare(y, type=CompareType.DISTANCE, kernel=kernel_y)
        diff_in_dist = upper_triangle(rdm_x - rdm_y)
        return torch.sqrt(torch.mean(diff_in_dist**2))


class Corr(BaseRepSim):
    """Correlation between RDMs; this is 'classic' RSA when correlation type is 'spearman'
    """
    def __init__(self, corr_type: CorrType = CorrType.SPEARMAN, cmp_type: CompareType = CompareType.INNER_PRODUCT):
        self._corr_type = corr_type
        self._cmp_type = cmp_type

    @property
    def type(self):
        return MetricType.CORR

    def compare(self, x: torch.Tensor, y: torch.Tensor, *, kernel_x: Union[Kernel, None] = None, kernel_y: Union[Kernel, None] = None) -> torch.Tensor:
        rsm_x = compare(x, type=self._cmp_type, kernel=kernel_x)
        rsm_y = compare(y, type=self._cmp_type, kernel=kernel_y)
        return corrcoef(upper_triangle(rsm_x), upper_triangle(rsm_y), type=self._corr_type)


class AffineInvariantRiemannian(BaseRepSim):
    """Compute the 'affine-invariant Riemannian metric' between RDMs, as advocated for by Shahbazi et al (2021).

    Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian manifold to compare
        representations in brain and in models. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118271
    """
    @property
    def type(self) -> MetricType:
        return MetricType.RIEMANN

    def compare(self, x: torch.Tensor, y: torch.Tensor, *, kernel_x: Union[Kernel, None] = None, kernel_y: Union[Kernel, None] = None) -> torch.Tensor:
        rsm_x = compare(x, type=CompareType.INNER_PRODUCT, kernel=kernel_x)
        rsm_y = compare(y, type=CompareType.INNER_PRODUCT, kernel=kernel_y)

        x_inv_y = torch.linalg.solve(rsm_x, rsm_y)
        log_eigs = torch.log(torch.linalg.eigvalsh(x_inv_y))
        return torch.sqrt(torch.sum(log_eigs**2))


def hsic(k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True) -> torch.Tensor:
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
        k_y = kernels.center(k_y)

    if unbiased:
        # Remove the diagonal
        k_x = k_x * (1 - torch.eye(n, device=k_x.device, dtype=k_x.dtype))
        k_y = k_y * (1 - torch.eye(n, device=k_y.device, dtype=k_y.dtype))
        # Equation (4) from Song et al (2007)
        return ((k_x *k_y).sum() - 2*(k_x.sum(dim=0)*k_y.sum(dim=0)).sum()/(n-2) + k_x.sum()*k_y.sum()/((n-1)*(n-2))) / (n*(n-3))
    else:
        # The original estimator from Gretton et al (2005)
        return torch.sum(k_x * k_y) / (n - 1)**2


def cka(k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True) -> torch.Tensor:
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


__all__ = ["BaseRepSim", "GeneralizedShapeMetric", "Stress", "Corr", "AffineInvariantRiemannian"]
