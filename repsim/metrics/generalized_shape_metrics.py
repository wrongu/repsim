import torch
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import GeodesicLengthSpace, Point, Scalar
from repsim.geometry.trig import slerp
from repsim.util import prod


class ShapeMetric(RepresentationMetricSpace, GeodesicLengthSpace):
    """Compute the basic shape-metric advocated by Williams et al (2021), using the angular (arc-length) definition.

    Implementation is paraphrased from https://github.com/ahwillia/netrep/

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS. https://arxiv.org/abs/2110.14739
    """

    SCORE_METHODS = ["euclidean", "angular"]

    def __init__(self, m, p, alpha=1.0, score_method="euclidean"):
        super().__init__(dim=m*p, shape=(m, p))
        self.m = m
        self.p = p
        self._alpha = alpha
        self._score_method = score_method
        assert score_method in ShapeMetric.SCORE_METHODS, \
            f"score_method must be one of {ShapeMetric.SCORE_METHODS} but was {score_method}"

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData, *, set_alpha=None) -> Point:
        """Convert size (m,d) neural data to a size (m,p) matrix. If d>p then we keep the top p dimensions by PCA. If
        d<p we pad with zeros.

        Essentially, this takes care of 'translation' and 'scaling' invariances, up to some regularization, so that
        length() only needs to solve the optimal rotation problem.
        """
        if x.shape[0] != self.m:
            raise ValueError(f"Expected x to be size ({self.m}, ?) but is size {x.shape}")

        # Center columns to handle translation-invariance
        x = x - torch.mean(x, dim=0)

        # Rescale and (partially) whiten to handle scale-invariance, using self._alpha or overriding it with set_alpha.
        # The reason to optionally override is that when 0 < self._alpha < 1, whiten(whiten(x)) != whiten(x). When
        # a user calls neural_data_to_point(x) for the first time, we'll use self._alpha. But in other cases such as
        # self.project(x), we'll assume that x was already converted to a 'Point' and not do any more whitening.
        x = _whiten(x, self._alpha if set_alpha is None else set_alpha)

        # Pad or truncate to p dimensions
        d = prod(x.shape) // self.m
        if d > self.p:
            # PCA to truncate
            u, s, v = torch.linalg.svd(x)
            x = u[:, :self.p] @ torch.diag(s[:self.p]) @ v[:, :self.p][:self.p, :].T
        elif d < self.p:
            # Pad zeros
            num_pad = self.p - d
            x = torch.hstack([x.view(self.m, d), x.new_zeros(self.m, num_pad)])

        return x

    def string_id(self) -> str:
        return f"ShapeMetric[{self._alpha:.2f}][{self.p}][{self._score_method}].{self.m}"

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _project_impl(self, pt: Point) -> Point:
        # Center and truncate but don't whiten because whiten(whiten(x)) != whiten(x) whenever alpha<1.
        return self.neural_data_to_point(pt, set_alpha=1.)

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.p):
            return False
        # Test centered
        if not torch.allclose(torch.mean(pt, dim=0), pt.new_zeros(pt.shape[1:]), atol=atol):
            return False
        # Note - NOT testing if whitened, since when alpha>0 this could trigger whiten(whiten(x)), which != whiten(x).
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # Step 1: find optimal rotation that aligns pt_a and pt_b. Assumes pt_a and pt_b are whitened already (hence
        # setting alpha=1.0 here). Since we assume pt_a and pt_b are already centered and scaled, this reduces to the
        new_a, new_b = _orthogonal_procrustes(pt_a, pt_b)

        # Step 2: score the result
        if self._score_method == "euclidean":
            diff_ab = new_a - new_b
            return torch.mean(torch.linalg.vector_norm(diff_ab, dim=-1))
        elif self._score_method == "angular":
            cos_ab = torch.sum(new_a * new_b) / torch.sqrt(torch.sum(new_a * new_a) * torch.sum(new_b * new_b))
            return torch.arccos(torch.clip(cos_ab, -1.0, +1.0))

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        # Step 1: find optimal rotation that aligns pt_a and pt_b. Assumes pt_a and pt_b are whitened already (hence
        # setting alpha=1.0 here). Since we assume pt_a and pt_b are already centered and scaled, this reduces to the
        new_a, new_b = _orthogonal_procrustes(pt_a, pt_b)

        # Step 2: interpolate 'new' points
        if self._score_method == "euclidean":
            return new_a * (1-frac) + new_b * frac
        elif self._score_method == "angular":
            return slerp(new_a, new_b, frac)



class EuclideanShapeMetric(ShapeMetric):
    """Specialization of ShapeMetric which automatically sets score_method='euclidean'
    """
    def __init__(self, *args, **kwargs):
        super(EuclideanShapeMetric, self).__init__(*args, score_method="euclidean", **kwargs)


class AngularShapeMetric(ShapeMetric):
    """Specialization of ShapeMetric which automatically sets score_method='angular'
    """
    def __init__(self, *args, **kwargs):
        super(AngularShapeMetric, self).__init__(*args, score_method="angular", **kwargs)


def _orthogonal_procrustes(a, b, anchor="middle"):
    """Provided a and b, each matrices of size (m, p) that are already centered and scaled, solve the orthogonal
    procrustest problem (rotate a and b into a common frame that minimizes distances).

    If anchor="middle" (default) then both a and b
    If anchor="a", then a is left unchanged and b is rotated towards it
    If anchor="b", then b is left unchanged and a is rotated towards it

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: new_a, new_b the rotated versions of a and b, minimizing element-wise squared differences
    """
    u, _, v = torch.linalg.svd(a.T @ b)
    # Helpful trick to see how these are related: u is the inverse of u.T, and likewise v is inverse of v.T. We get to
    # the anchor=a and anchor=b solutions by right-multiplying both return values by u.T or right-multiplying both
    # return values by v, respectively (if both return values are rotated in the same way, it preserves the shape).
    if anchor == "middle":
        return a @ u, b @ v.T
    elif anchor == "a":
        return a, b @ v.T @ u.T
    elif anchor == "b":
        return a @ u @ v, b
    else:
        raise ValueError(f"Invalid 'anchor' argument: {anchor} (must be 'middle', 'a', or 'b')")


def _whiten(x, alpha, clip_eigs=1e-9):
    """Compute (partial) whitening transform of x. When alpha=0 it is classic ZCA whitening and columns of x are totally
    decorrelated. When alpha=1, nothing happens.

    Assumes x is already centered.
    """
    e, v = torch.linalg.eigh(x.T @ x)
    e = torch.clip(e, min=clip_eigs, max=None)
    d = alpha + (1 - alpha) * (e ** -0.5)
    # From right to left, the transformation (1) projects x onto v, (2) divides by stdev in each direction, and (3)
    # rotates back to align with original directions in x-space (ZCA)
    z = v @ torch.diag(d) @ v.T
    # Think of this as (z @ x.T).T, but note z==z.T
    return x @ z
