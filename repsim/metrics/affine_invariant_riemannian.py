import torch
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.kernels import DEFAULT_KERNEL
from repsim import pairwise


class AffineInvariantRiemannian(RepresentationMetricSpace, RiemannianSpace):
    """Compute the 'affine-invariant Riemannian metric', as advocated for by [1].

    NOTE: given (m,d) sized inputs, this involves inverting a (m,m)-sized matrix, which might be rank-deficient. The
    authors of [1] got around this by switching the inner-product to be across conditions, and compared (d,d)-sized
    matrices. However, this no longer suffices as a general RSA tool, since in general d_x will not equal d_y.

    We get around this one of two ways. First, using a kernel with an infinite basis all but guarantees invertability.
    Second, we can regularize the n by n matrix, shrinking it towards its diagonal [2]. The latter breaks some affine
    invariance properties, though, and should be avoided.

    [1] Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian manifold to compare
        representations in brain and in models. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118271
    [2] Yatsenko, D., Josić, K., Ecker, A. S., Froudarakis, E., Cotton, R. J., & Tolias, A. S. (2015). Improved
        Estimation and Interpretation of Correlations in Neural Circuits. PLoS Computational Biology, 11(3), 1–28.
        https://doi.org/10.1371/journal.pcbi.1004083
    """

    def __init__(self, m, shrinkage=0.0, kernel=None):
        super().__init__(dim=m*(m+1)/2, shape=(m, m))
        self._kernel = kernel if kernel is not None else DEFAULT_KERNEL
        if shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError(
                "Shrinkage parameter must be in [0,1], where 0 means no regularization."
            )
        self._shrink = shrinkage

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,m) Gram matrix of inner products between xs using self._kernel.
        """
        if x.shape[0] != self.m:
            raise ValueError(f"Expected x to be size ({self.m}, ?) but is size {x.shape}")
        return pairwise.inner_product(x, kernel=self._kernel)

    def string_id(self) -> str:
        if self._shrink > 0.:
            return f"AffineInvariantRiemannian[{self._shrink:.3f}].{self._kernel.string_id()}.{self.m}"
        else:
            return f"AffineInvariantRiemannian.{self._kernel.string_id()}.{self.m}"

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        assert pt.shape == (self.m, self.m), \
            f"Input to AngularCKA.project() must be a m by m matrix but is size {pt.shape}!"
        # 1. Ensure matrix is symmetric
        pt = (pt + pt.T) / 2
        # 2. Ensure matrix is positive definite by clipping its eigenvalues
        pt = _eig_fun(pt, lambda e: torch.clip(e, min=0., max=None))
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.m):
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
        n = pt_a.size()[0]
        # Apply shrinkage regularizer: down-weight all off-diagonal parts of each RSM by self._shrink.
        off_diag_n = 1.0 - torch.eye(n, device=pt_a.device, dtype=pt_a.dtype)
        pt_a = pt_a - self._shrink * off_diag_n * pt_a
        pt_b = pt_b - self._shrink * off_diag_n * pt_b
        if torch.linalg.matrix_rank(pt_a) < self.m or torch.linalg.matrix_rank(pt_b) < self.m:
            return torch.tensor([float('inf')])
        # TODO - do we need shrinkage and rank checks if we use a pseudo-inverse instead? Or will eigs be zero
        # and therefore dist --> infinity?
        inv_a_half = _inv_matrix_sqrt(pt_a)
        x_inv_y = inv_a_half @ pt_b @ inv_a_half
        log_eigs = torch.log(torch.linalg.eigvals(x_inv_y).real)
        return torch.sqrt(torch.sum(log_eigs**2))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        a_half = _matrix_sqrt(pt_a)
        inv_a_half = _inv_matrix_sqrt(pt_a)
        # See equations (3.12) and (3.13) in [1]. Here we combine them and simplify a bit algebraically.
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In Riemannian Geometric Statistics
        # in Medical Image Analysis. Elsevier Ltd. https://doi.org/10.1016/B978-0-12-814725-2.00010-8
        #
        # Long version:
        #   log_a_b = a_half @ _matrix_log(inv_a_half @ pt_b @ inv_a_half) @ a_half
        #   return a_half @ _matrix_exp(frac * inv_a_half @ log_a_b @ inv_a_half) @ a_half
        # Medium version:
        #   return a_half @ _matrix_exp(frac * _matrix_log(inv_a_half @ pt_b @ inv_a_half)) @ a_half
        # Short version:
        return a_half @ _matrix_pow(inv_a_half @ pt_b @ inv_a_half, frac) @ a_half

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        # Find the nearest matrix to vec_w that is symmetric
        return (vec_w + vec_w.T) / 2

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        inv_base_point = _inv_matrix(pt_a)
        return torch.einsum('ij,ji->', inv_base_point @ vec_w, inv_base_point @ vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # See equation (3.12) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In Riemannian Geometric Statistics
        a_half = _matrix_sqrt(pt_a)
        inv_a_half = _inv_matrix_sqrt(pt_a)
        return a_half @ _matrix_exp(inv_a_half @ vec_w @ inv_a_half) @ a_half

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # See equation (3.13) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In Riemannian Geometric Statistics
        a_half = _matrix_sqrt(pt_a)
        inv_a_half = _inv_matrix_sqrt(pt_a)
        return a_half @ _matrix_log(inv_a_half @ pt_b @ inv_a_half) @ a_half

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # See equation (3.15) in [1].
        # [1] Pennec, X. (2019). Manifold-valued image processing with SPD matrices. In Riemannian Geometric Statistics
        # vec_v = self.log_map(pt_a, pt_b)
        # return -1/2*(vec_v @ torch.linalg.solve(pt_a, vec_w) + vec_w @ torch.linalg.solve(pt_a, vec_v))
        a_half = _matrix_sqrt(pt_a)
        inv_a_half = _inv_matrix_sqrt(pt_a)
        e = a_half @ _matrix_sqrt(inv_a_half @ pt_b @ inv_a_half) @ inv_a_half
        # Include to_tangent() to be absolutely sure the output lands in the tangent space of pt_b despite numerical
        # imprecision
        return self.to_tangent(pt_b, e @ vec_w @ e.T)


def _eig_fun(hmat, fun):
    """Apply a function to the eigenvalues of a symmetric (hermitian) matrix
    """
    e, u = torch.linalg.eigh(hmat)
    return u @ torch.diag(fun(e)) @ u.T


def _inv_matrix(hmat):
    return _eig_fun(hmat, fun=lambda x: 1. / x)


def _matrix_sqrt(hmat):
    return _eig_fun(hmat, fun=torch.sqrt)


def _inv_matrix_sqrt(hmat):
    return _eig_fun(hmat, fun=lambda x: 1. / torch.sqrt(x))


def _matrix_log(hmat):
    return _eig_fun(hmat, fun=torch.log)


def _matrix_exp(hmat):
    return _eig_fun(hmat, fun=torch.exp)


def _matrix_pow(hmat, p):
    return _eig_fun(hmat, fun=lambda e: torch.pow(e, p))
