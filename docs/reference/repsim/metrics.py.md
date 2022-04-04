## *Class* `RepresentationMetricSpace`


Base mixin class for all representational similarity/representational distance comparisons. Subclasses will inherit from *both* RepresentationMetricSpace and *one of* SPDMatrix or DistMatrix.



### *Function* `__init__ (self, n: int, kernel=None): super(RepresentationMetricSpace, self).__init__(rows=n) self._kernel = kernel @property def metric_type(self) -> MetricType: raise NotImplementedError("type must be specified by a subclass") @property def compare_type(self) -> CompareType: raise NotImplementedError("compare_type must be specified by a subclass") def to_rdm(self, x: NeuralData) -> Point: """Convert (n,d) sized neural data into (n,n) pairwise comparison (representational distance) matrix, where the latter is a Point in the metric space. """ return pairwise.compare(x, kernel=self._kernel, type=self.compare_type) def representational_distance(self, x: torch.Tensor, y: torch.Tensor) -> Scalar: return self.length(self.to_rdm(x), self.to_rdm(y)) class AngularCKA(RepresentationMetricSpace, SPDMatrix)`


Compute the angular distance between two representations x and y using the arccos(CKA) method described in the supplement of Williams et al (2021).

Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural Representations. NeurIPS. http://arxiv.org/abs/2110.14739



### *Function* `metric_type (self) -> MetricType: return MetricType.ANGLE @property def compare_type(self) -> CompareType: return CompareType.INNER_PRODUCT def length(self, rdm_x: Point, rdm_y: Point) -> Scalar: # Note: use clipping in case of numerical imprecision. arccos(1.00000000001) will give NaN! return torch.arccos(torch.clip(cka(rdm_x, rdm_y), -1.0, 1.0)) class Stress(RepresentationMetricSpace, DistMatrix)`


Difference-in-pairwise-distance, AKA 'stress' from the MDS literature.



### *Function* `metric_type (self) -> MetricType: return MetricType.LENGTH @property def compare_type(self) -> CompareType: return CompareType.DISTANCE def length(self, rdm_x: Point, rdm_y: Point) -> Scalar: diff_in_dist = upper_triangle(rdm_x - rdm_y) return torch.sqrt(torch.mean(diff_in_dist**2)) class AffineInvariantRiemannian(RepresentationMetricSpace, SPDMatrix)`


Compute the 'affine-invariant Riemannian metric', as advocated for by [1].

> - **NOTE** (`None`: `None`): given (n,d) sized inputs, this involves inverting a (n,n)-sized matrix, which might be rank-deficient. The authors of [1] got around this by switching the inner-product to be across conditions, and compared (d,d)-sized matrices. However, this no longer suffices as a general RSA tool, since in general d_x will not equal d_y.

We get around this by regularizing the n by n matrix, shrinking it towards its diagonal (see Yatsenko et al (2015))

[1] Shahbazi, M., Shirali, A., Aghajan, H., & Nili, H. (2021). Using distance on the Riemannian manifold to compare representations in brain and in models. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118271


### *Function* `__init__(self, **kwargs):
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
) -> torch.Tensor ()`


Compute Hilbert-Schmidt Independence Criteron (HSIC)

* note that if unbiased=True, it is possible to get small values below 0.

Arguments;
> - **k_x** (`None`: `None`): n by n values of kernel applied to all pairs of x data
> - **k_y** (`None`: `None`): n by n values of kernel on y data
> - **centered** (`None`: `None`): whether or not at least one kernel is already centered
> - **unbiased** (`None`: `None`): if True, use unbiased HSIC estimator of Song et al (2007),
        else use original estimator of Gretton et al (2005)

### Returns
    scalar score in [0*, inf) measuring dependence of x and y



### *Function* `cka(
    k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True
) -> torch.Tensor ()`


Compute Centered Kernel Alignment (CKA).

### Arguments
> - **k_x** (`None`: `None`): n by n values of kernel applied to all pairs of x data
> - **k_y** (`None`: `None`): n by n values of kernel on y data
> - **centered** (`None`: `None`): whether or not at least one kernel is already centered
> - **unbiased** (`None`: `None`): if True, use unbiased HSIC estimator of Song et al (2007),
        else use original estimator of Gretton et al (2005)
### Returns
    scalar score in [0*, 1] measuring normalized dependence between x and y

* note that if unbiased=True, it is possible to get small values below 0.
