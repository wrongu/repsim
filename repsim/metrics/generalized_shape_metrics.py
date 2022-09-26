import torch
import numpy as np
from torch.linalg import norm, vector_norm, svd, qr, eigh
from scipy.linalg import solve_sylvester
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.geometry.trig import slerp
from repsim.util import prod


class ShapeMetric(RepresentationMetricSpace, RiemannianSpace):
    """Compute the basic shape-metric advocated by Williams et al (2021), using the angular (arc-length) definition.

    Implementation is paraphrased from https://github.com/ahwillia/netrep/

    Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape Metrics on Neural
        Representations. NeurIPS. https://arxiv.org/abs/2110.14739
    """

    SCORE_METHODS = ["euclidean", "angular"]

    def __init__(self, m, p, alpha=1.0, score_method="euclidean"):
        super().__init__(dim=m*p, shape=(m, p))
        self.p = p
        self._alpha = alpha
        self._score_method = score_method
        assert score_method in ShapeMetric.SCORE_METHODS, \
            f"score_method must be one of {ShapeMetric.SCORE_METHODS} but was {score_method}"
        self._rotation_basis = None

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,p) matrix. If d>p then we keep the top p dimensions by PCA. If
        d<p we pad with zeros.

        Essentially, this takes care of 'translation' and 'scaling' invariances, up to some regularization, so that
        length() only needs to solve the optimal rotation problem.
        """
        if x.shape[0] != self.m:
            raise ValueError(f"Expected x to be size ({self.m}, ?) but is size {x.shape}")

        # Flatten all but first dimension. TODO optional conv reshape into (m*h*w,c) as done by Williams et al?
        x = torch.reshape(x, (self.m, -1))

        # Center columns to handle translation-invariance
        x = x - torch.mean(x, dim=0)

        # Pad or truncate to p dimensions
        d = prod(x.shape) // self.m
        if d > self.p:
            # PCA to truncate -- project onto top p principal axes (no rescaling)
            _, _, v = svd(x)
            x = x @ v[:, :self.p]
        elif d < self.p:
            # Pad zeros
            num_pad = self.p - d
            x = torch.hstack([x.view(self.m, d), x.new_zeros(self.m, num_pad)])

        # Rescale and (partially) whiten to handle scale-invariance, using self._alpha.
        x = _whiten(x, self._alpha)

        # In case of 'angular' metric only, make all point clouds unit-frobenius-norm
        if self._score_method == "angular":
            x = x / norm(x, ord="fro")

        return x

    def string_id(self) -> str:
        return f"ShapeMetric[{self._alpha:.2f}][{self.p}][{self._score_method}].{self.m}"

    @property
    def is_spherical(self) -> bool:
        return self._score_method == "angular"

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        # Assume that pt.shape == (m, p). The first operation is to center it:
        pt = pt - torch.mean(pt, dim=0)
        # If score_method is 'angular' then scale doesn't matter; normalize the point
        if self._score_method == "angular":
            pt = pt / torch.sqrt(torch.sum(pt ** 2))
        # Note – whitening is only done in neural_data_to_point since whitening is only idempotent if alpha==0.
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.p):
            return False
        # Test centered
        if not torch.allclose(torch.mean(pt, dim=0), pt.new_zeros(pt.shape[1:]), atol=atol):
            return False
        # Test unit norm (if angular)
        if self._score_method == "angular" and not torch.isclose(torch.sum(pt ** 2), pt.new_ones((1,))):
            return False
        # Note – whitening is not tested because we could only assert something about it if alpha==0.
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # Step 1: find optimal rotation that aligns pt_a and pt_b. Assumes pt_a and pt_b are whitened already (hence
        # setting alpha=1.0 here). Since we assume pt_a and pt_b are already centered and scaled, this reduces to the
        new_a, new_b = _orthogonal_procrustes(pt_a, pt_b, anchor="a")

        # Step 2: score the result
        if self._score_method == "euclidean":
            diff_ab = new_a - new_b
            return torch.mean(vector_norm(diff_ab, dim=-1))
        elif self._score_method == "angular":
            cos_ab = torch.sum(new_a * new_b) / torch.sqrt(torch.sum(new_a * new_a) * torch.sum(new_b * new_b))
            return torch.arccos(torch.clip(cos_ab, -1.0, +1.0))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        # Step 1: find optimal rotation that aligns pt_a and pt_b. Assumes pt_a and pt_b are whitened already (hence
        # setting alpha=1.0 here). Since we assume pt_a and pt_b are already centered and scaled, this reduces to the
        new_a, new_b = _orthogonal_procrustes(pt_a, pt_b, anchor="a")

        # Step 2: interpolate 'new' points
        if self._score_method == "euclidean":
            return new_a * (1-frac) + new_b * frac
        elif self._score_method == "angular":
            return slerp(new_a, new_b, frac)

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    @staticmethod
    def _rotation_tangent_space(pt_a: Point) -> torch.Tensor:
        m, p = pt_a.shape
        basis = []
        for i in range(p):
            for j in range(i+1, p):
                v = pt_a.new_zeros(m, p)
                v[:, i], v[:, j] = pt_a[:, j], -pt_a[:, i]
                basis.append(v)
        basis = torch.stack([torch.flatten(b) for b in basis], dim=-1)
        q, r = qr(basis)
        return q[:, :p-1] / norm(q[:, :p-1], dim=0, keepdims=True)

    def _pre_shape_tangent(self, pt_a: Point, vec_w: Vector):
        """Project to tangent in the pre-shape space. Pre-shapes are equivalent translation (and scale if using angular
        distance), but not rotation.

        :param pt_a: base point for the tangent vector
        :param vec_w: ambient space vector
        :return: tangent vector with mean-shifts removed, as well as scaling removed (if score_method=angular)
        """
        # Points must be 'centered', so subtract off component of vec that would affect the mean
        vec_w = vec_w - torch.mean(vec_w, dim=0)
        # In 'angular' case, subtract off component that would uniformly scale all points (component of the tangent
        # in the direction of pt_a)
        if self._score_method == "angular":
            vec_w = vec_w - pt_a * torch.sum(vec_w * pt_a) / torch.sum(pt_a * pt_a)
        return vec_w

    def _horizontal_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        """The 'horizontal' part of the tangent space is the part that is actually movement in the quotient space,
        i.e. across equivalence classes. For example, east/west movement where equivalence = lines of longitude.
        """
        vec_w = self._pre_shape_tangent(pt_a, vec_w)
        vert_part = self._vertical_tangent(pt_a, vec_w)
        return vec_w - vert_part * torch.sum(vec_w * vert_part) / torch.sum(vert_part * vert_part)

    def _vertical_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        """The 'vertical' part of the tangent space is the part that doesn't count as movement in the quotient space,
        i.e. within equivalence classes. For example, north/south movement where equivalence = lines of longitude.

        The space of 'vertical' tangents, after accounting for shifts and scales with _aux_to_tangent, is the set of
        rotations. We get these by looking at the span of all 2D rotations – one per pair of axes in our space.
        """
        vec_w = self._pre_shape_tangent(pt_a, vec_w)
        # See equation (2) in Nava-Yazdani et al (2020), but note that all of our equations are transposed from theirs
        xxT = pt_a.T @ pt_a
        wxT = vec_w.T @ pt_a
        mat_a = _solve_sylvester(xxT, xxT, wxT - wxT.T)
        return pt_a @ mat_a.T

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        # Return tangent in the PRE shape space; tangent in the quotient space is given by _horizontal_tangent. But
        # note that log_map returns a horizontal tangent vector. levi_civita transports both horizontal and vertical
        # parts, so this function is designed to retain vertical largely to be consistent with transport.
        return self._pre_shape_tangent(pt_a, vec_w)

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        # Ensure that we're only measuring the 'horizontal' part of each tangent vector.
        h_vec_w, h_vec_v = self._horizontal_tangent(pt_a, vec_w), self._horizontal_tangent(pt_a, vec_v)
        return torch.sum(h_vec_w * h_vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # Ensure that we're only applying the 'horizontal' part of the tangent vector.
        h_vec_w = self._horizontal_tangent(pt_a, vec_w)
        if self._score_method == "euclidean":
            return pt_a + h_vec_w
        elif self._score_method == "angular":
            # Identical to Hypersphere.exp_map
            # See https://math.stackexchange.com/a/1930880
            norm = torch.sqrt(torch.sum(h_vec_w * h_vec_w))
            c1 = torch.cos(norm)
            c2 = torch.sinc(norm / np.pi)
            return c1 * pt_a + c2 * h_vec_w

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # Align b to a, then get 'horizontal' vector pointing from a to new b
        _, new_b = _orthogonal_procrustes(pt_a, pt_b, anchor="a")
        if self._score_method == "euclidean":
            return new_b - pt_a
        elif self._score_method == "angular":
            # Identical to Hypersphere.log_map but in the direction of 'new_b'
            unscaled_w = self._pre_shape_tangent(pt_a, new_b)
            norm_w = unscaled_w / torch.clip(torch.sqrt(torch.sum(unscaled_w * unscaled_w)), 1e-7)
            return norm_w * self.length(pt_a, pt_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Both the horizontal and vertical parts of tangent vectors are equivariant after rotation (Lemma 1b of
        # Nava-Yazdani et al (2020)). This means we can start by aligning a to b as follows:
        r_a, _ = _orthogonal_procrustes_rotation(pt_a, pt_b, anchor="b")
        pt_a, vec_w = pt_a @ r_a, vec_w @ r_a
        # Next, we do the usual euclidean or spherical transports between aligned a and b:
        if self._score_method == "euclidean":
            return vec_w
        elif self._score_method == "angular":
            # Refer to Hypersphere.levi_civita
            vec_v = self.log_map(pt_a, pt_b)
            angle = self.length(pt_a, pt_b)
            unit_v = vec_v / torch.clip(angle, 1e-7)  # the length of tangent vector v *is* the length from a to b
            w_along_v = torch.sum(unit_v * vec_w)
            orth_part = vec_w - w_along_v * unit_v
            return orth_part + torch.cos(angle) * w_along_v * unit_v - torch.sin(angle) * w_along_v * pt_a


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


def _orthogonal_procrustes_rotation(a, b, anchor="middle"):
    """Provided a and b, each matrices of size (m, p) that are already centered and scaled, solve the orthogonal
    procrustest problem (rotate a and b into a common frame that minimizes distances).

    If anchor="middle" (default) then both a and b
    If anchor="a", then a is left unchanged and b is rotated towards it
    If anchor="b", then b is left unchanged and a is rotated towards it

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: r_a and r_b, which, when right-multiplied with a and b, gives the aligned coordinates, or None for each if
    no transform is required
    """
    u, _, v = svd(a.T @ b)
    # Helpful trick to see how these are related: u is the inverse of u.T, and likewise v is inverse of v.T. We get to
    # the anchor=a and anchor=b solutions by right-multiplying both return values by u.T or right-multiplying both
    # return values by v, respectively (if both return values are rotated in the same way, it preserves the shape).
    if anchor == "middle":
        return u, v.T
    elif anchor == "a":
        return None, v.T @ u.T
    elif anchor == "b":
        return u @ v, None
    else:
        raise ValueError(f"Invalid 'anchor' argument: {anchor} (must be 'middle', 'a', or 'b')")


def _orthogonal_procrustes(a, b, anchor="middle"):
    """Provided a and b, each matrices of size (m, p) that are already centered and scaled, solve the orthogonal
    procrustest problem (rotate a and b into a common frame that minimizes distances).

    If anchor="middle" (default) then both a and b
    If anchor="a", then a is left unchanged and b is rotated towards it
    If anchor="b", then b is left unchanged and a is rotated towards it

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: new_a, new_b the rotated versions of a and b, minimizing element-wise squared differences
    """
    r_a, r_b = _orthogonal_procrustes_rotation(a, b, anchor)
    return a @ r_a if r_a is not None else a, b @ r_b if r_b is not None else b


def _whiten(x, alpha, clip_eigs=1e-9):
    """Compute (partial) whitening transform of x. When alpha=0 it is classic ZCA whitening and columns of x are totally
    decorrelated. When alpha=1, nothing happens.

    Assumes x is already centered.
    """
    e, v = eigh(x.T @ x)
    e = torch.clip(e, min=clip_eigs, max=None)
    d = alpha + (1 - alpha) * (e ** -0.5)
    # From right to left, the transformation (1) projects x onto v, (2) divides by stdev in each direction, and (3)
    # rotates back to align with original directions in x-space (ZCA)
    z = v @ torch.diag(d) @ v.T
    # Think of this as (z @ x.T).T, but note z==z.T
    return x @ z


def _solve_sylvester(a, b, q):
    # TODO - implement natively in pytorch so we don't have to convert to numpy on CPU and back again
    a_np, b_np, q_np = a.detach().cpu().numpy(), b.detach().cpu().numpy(), q.detach().cpu().numpy()
    return torch.tensor(solve_sylvester(a_np, b_np, q_np), dtype=a.dtype, device=a.device)
