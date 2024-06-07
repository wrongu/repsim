import torch
from torch.linalg import svd
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.kernels import DEFAULT_KERNEL
from repsim import pairwise
from repsim.util import (
    prod,
    eig_fun,
    inv_matrix,
    matrix_sqrt,
    inv_matrix_sqrt,
    matrix_log,
    matrix_exp,
    matrix_pow,
)
import warnings


class AffineInvariantRiemannian(RepresentationMetricSpace, RiemannianSpace):
    """Compute the 'affine-invariant Riemannian metric', as advocated for by [1].

    NOTE: given (m,d) sized inputs, this involves inverting a (m,m)-sized matrix, which might be
    rank-deficient. The authors of [1] got around this by switching the inner-product to be
    across conditions, and compared (d,d)-sized matrices. However, this no longer suffices as a
    general RSA tool, since in general d_x will not equal d_y.

    We get around this one of two ways. First, using a kernel with an infinite basis all but
    guarantees invertability. However, if some neural data contains duplicate rows (e.g. one-hot
    labels), then the resulting Gram matrix will still be rank-deficient. Second, we regularize
    the m by m Gram matrix by emphasizing its diagonal (adding eps times the identity matrix).
    Note that these regularization methods mean that the resulting metric may not actually be
    affine-invariant!

    [1] Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian
        manifold to compare representations in brain and in models. NeuroImage.
        https://doi.org/10.1016/j.neuroimage.2021.118271
    """

    def __init__(self, m, eps=0.0, *, kernel=None, p=None, mode="gram"):
        if mode == "gram":
            super().__init__(dim=m * (m + 1) / 2, shape=(m, m))
            if p is not None:
                warnings.warn("Parameter 'p' has no effect when mode='gram'")
        elif mode == "cov":
            if p is None:
                raise ValueError("Parameter 'p' is required when mode is 'cov'")
            super().__init__(dim=p * (p + 1) / 2, shape=(p, p))
            if kernel is not None:
                warnings.warn("Parameter 'kernel' has no effect when mode='cov'")
        else:
            raise ValueError(f"'mode' argument must be 'gram' or 'cov' but was {mode}")
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL
        if eps < 0.0:
            raise ValueError(f"eps regularization must be nonnegative but is {eps}")
        self._eps = eps
        self._mode = mode
        self._p = p
        self._m = m

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################
    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, new_m):
        self._m = new_m
        if self._mode == "gram":
            self.shape = (new_m, new_m)
            self.dim = new_m * (new_m + 1) / 2

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,m) Gram matrix of inner products between xs
        using self._kernel."""
        # Always preprocess by subtracting the mean to ensure translation invariance
        x = x - x.mean(dim=0, keepdims=True)
        if x.shape[0] != self.m:
            raise ValueError(
                f"Expected x to be size ({self.m}, ?) but is size {x.shape}"
            )
        if self._mode == "gram":
            gram_matrix = pairwise.inner_product(x, kernel=self._kernel)
            # Apply regularizer: add a small ridge to the diagonal
            return gram_matrix + self._eps * torch.eye(
                self.m, device=x.device, dtype=x.dtype
            )
        elif self._mode == "cov":
            # Flatten all but first dimension.
            x = torch.reshape(x, (self.m, -1))
            # Pad or truncate to p dimensions
            d = prod(x.shape) // self.m
            if d > self._p:
                # PCA to truncate -- project onto top p principal axes (no rescaling)
                _, _, v = svd(x)
                x = x @ v[:, : self._p]
            elif d < self._p:
                # Pad zeros
                num_pad = self._p - d
                x = torch.hstack([x.view(self.m, d), x.new_zeros(self.m, num_pad)])
            # Compute covariance
            cov_matrix = torch.einsum("mi,mj->ij", x, x) / (self.m - 1)
            # Apply regularizer: add small ridge
            return cov_matrix + self._eps * torch.eye(
                self._p, device=x.device, dtype=x.dtype
            )

    def string_id(self) -> str:
        eps_str = "" if self._eps == 0.0 else f"[{self._eps:.3f}]"
        if self._mode == "gram":
            return f"AffineInvariantRiemannian[gram]{eps_str}.{self._kernel.string_id()}.{self.m}"
        elif self._mode == "cov":
            return f"AffineInvariantRiemannian[cov]{eps_str}[{self._p}].{self.m}"

    @property
    def is_spherical(self) -> bool:
        return False

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        assert pt.shape == self.shape, (
            f"Input to AffineInvariantRiemannian.project() must be a {self.shape} matrix "
            f"but is size {pt.shape}!"
        )
        # 1. Ensure matrix is symmetric
        pt = (pt + pt.T) / 2
        # 2. Ensure matrix is positive definite by clipping its eigenvalues
        pt = eig_fun(pt, lambda e: torch.clip(e, min=0.0, max=None))
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != self.shape:
            return False
        # Test symmetric
        if not torch.allclose(pt, pt.T, atol=atol):
            return False
        # Test positive definiteness
        e = torch.linalg.eigvalsh(pt)
        if not all(e > -atol):
            return False
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # If rank-deficient, return infinity and be done early
        if (
            torch.linalg.matrix_rank(pt_a) < self.shape[0]
            or torch.linalg.matrix_rank(pt_b) < self.shape[0]
        ):
            return torch.tensor([float("inf")], dtype=pt_a.dtype, device=pt_a.device)
        # TODO - do we need eps and rank checks if we use a pseudo-inverse instead? Or will eigs
        #  be zero and therefore dist --> infinity?
        inv_a_half = inv_matrix_sqrt(pt_a)
        x_inv_y = inv_a_half @ pt_b @ inv_a_half
        log_eigs = torch.log(torch.linalg.eigvals(x_inv_y).real)
        return torch.sqrt(torch.sum(log_eigs**2))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        a_half = matrix_sqrt(pt_a)
        inv_a_half = inv_matrix_sqrt(pt_a)
        # See equations (3.12) and (3.13) in [1]. Here we combine them and simplify a bit
        # algebraically. [1] Pennec, X. (2019). Manifold-valued image processing with SPD
        # matrices. In Riemannian Geometric Statistics in Medical Image Analysis. Elsevier Ltd.
        # https://doi.org/10.1016/B978-0-12-814725-2.00010-8
        #
        # Long version:
        #   log_a_b = a_half @ _matrix_log(inv_a_half @ pt_b @ inv_a_half) @ a_half
        #   return a_half @ _matrix_exp(frac * inv_a_half @ log_a_b @ inv_a_half) @ a_half
        # Medium version:
        #   return a_half @ _matrix_exp(frac * _matrix_log(inv_a_half @ pt_b @ inv_a_half)) @ a_half
        # Short version:
        return a_half @ matrix_pow(inv_a_half @ pt_b @ inv_a_half, frac) @ a_half

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        # Find the nearest matrix to vec_w that is symmetric
        return (vec_w + vec_w.T) / 2

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        inv_base_point = inv_matrix(pt_a)
        return torch.einsum("ij,ji->", inv_base_point @ vec_w, inv_base_point @ vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # See equation (3.12) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In
        #   Riemannian Geometric Statistics
        a_half = matrix_sqrt(pt_a)
        inv_a_half = inv_matrix_sqrt(pt_a)
        return a_half @ matrix_exp(inv_a_half @ vec_w @ inv_a_half) @ a_half

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # See equation (3.13) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In
        #   Riemannian Geometric Statistics
        a_half = matrix_sqrt(pt_a)
        inv_a_half = inv_matrix_sqrt(pt_a)
        return a_half @ matrix_log(inv_a_half @ pt_b @ inv_a_half) @ a_half

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # See equation (3.15) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In
        #   Riemannian Geometric Statistics
        a_half = matrix_sqrt(pt_a)
        inv_a_half = inv_matrix_sqrt(pt_a)
        e = a_half @ matrix_sqrt(inv_a_half @ pt_b @ inv_a_half) @ inv_a_half
        # Include to_tangent() to be absolutely sure the output lands in the tangent space of
        # pt_b despite numerical imprecision
        return self.to_tangent(pt_b, e @ vec_w @ e.T)
