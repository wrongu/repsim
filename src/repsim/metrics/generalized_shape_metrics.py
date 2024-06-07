import torch
import numpy as np
from torch.linalg import norm, eigh
from scipy.linalg import solve_sylvester
from .representation_metric_space import RepresentationMetricSpace, NeuralData
from repsim.geometry import RiemannianSpace, Point, Scalar, Vector
from repsim.geometry.trig import slerp
from repsim.util import prod


class PreShapeMetric(RepresentationMetricSpace, RiemannianSpace):
    """The Pre Shape Metric is like the Shape Metric but without removing rotations (no alignment
    step)."""

    SCORE_METHODS = ["euclidean", "angular"]

    def __init__(self, m, p, alpha=1.0, score_method="euclidean"):
        super().__init__(dim=m * p, shape=(m, p))
        self.p = p
        self._alpha = alpha
        self._score_method = score_method
        assert (
            score_method in PreShapeMetric.SCORE_METHODS
        ), f"score_method must be one of {PreShapeMetric.SCORE_METHODS} but was {score_method}"

    ###############################################
    # Implement RepresentationMetricSpace methods #
    ###############################################

    # TODO - allow p=None in which case no reshaping is done and matrices are always cast to the
    #  larger dimension in comparisons
    def neural_data_to_point(self, x: NeuralData) -> Point:
        """Convert size (m,d) neural data to a size (m,p) matrix. If d>p then we keep the top p
        dimensions by PCA. If d<p we pad with zeros.

        Essentially, this takes care of 'translation' and 'scaling' invariances, up to some
        regularization.
        """
        if x.shape[0] != self.m:
            raise ValueError(
                f"Expected x to be size ({self.m}, ?) but is size {x.shape}"
            )

        # Flatten all but first dimension. TODO optional conv reshape into (m*h*w,c) as done by Williams et al?
        x = torch.reshape(x, (self.m, -1))

        # Center columns to handle translation-invariance
        x = x - torch.mean(x, dim=0, keepdim=True)

        # Pad or truncate to p dimensions. Important that this happens before whitening because whitening destroys
        # principal components (makes them all equivalent).
        d = prod(x.shape) // self.m
        if d > self.p:
            x = _dim_reduce(x, self.p)
        elif d < self.p:
            x = _pad_zeros(x, self.p)

        # Rescale and (partially) whiten to handle scale-invariance, using self._alpha.
        x = _whiten(x, self._alpha)

        # In case of 'angular' metric only, make all point clouds unit-frobenius-norm. Then, points live on the
        # hypesphere of centered and unit-norm m-by-p matrices.
        if self._score_method == "angular":
            x = x / norm(x, ord="fro")

        return x

    def string_id(self) -> str:
        return f"PreShapeMetric[{self._alpha:.2f}][{self.p}][{self._score_method}].{self.m}"

    @property
    def is_spherical(self) -> bool:
        # TODO - when we whiten data with alpha=0, we are effectively projecting onto the unit
        #  sphere. So we should consider that case 'spherical' too. But for 0<alpha<1 and
        #  score_method='euclidean' it is neither spherical nor not-spherical. We currently don't
        #  handle that case particularly well.
        return self._score_method == "angular"

    #################################
    # Implement LengthSpace methods #
    #################################

    def _project_impl(self, pt: Point) -> Point:
        # Assume that pt.shape == (m, p). The first operation is to center it:
        pt = pt - torch.mean(pt, dim=0)
        # If score_method is 'angular' then scale doesn't matter; normalize the point
        if self._score_method == "angular":
            pt = pt / norm(pt, ord="fro")
        # Note – whitening is only done in neural_data_to_point since whitening is only idempotent if alpha==0.
        return pt

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        # Test shape
        if pt.shape != (self.m, self.p):
            return False
        # Test centered
        if not torch.allclose(
            torch.mean(pt, dim=0), pt.new_zeros(pt.shape[1:]), atol=atol
        ):
            return False
        # Test unit norm (if angular)
        if self._score_method == "angular" and not torch.isclose(
            norm(pt, ord="fro"), pt.new_ones((1,))
        ):
            return False
        if self._alpha == 0.0:
            # Test whitened (if alpha==0)
            whitened_pt = _whiten(pt, 0.0)
            if self._score_method == "angular":
                if not torch.allclose(
                    pt,
                    whitened_pt / torch.linalg.norm(whitened_pt, ord="fro"),
                    atol=atol,
                ):
                    return False
            elif self._score_method == "euclidean":
                if not torch.allclose(pt, whitened_pt, atol=atol):
                    return False
        return True

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        if self._score_method == "euclidean":
            diff_ab = pt_a - pt_b
            return norm(diff_ab, ord="fro")
        elif self._score_method == "angular":
            cos_ab = torch.sum(pt_a * pt_b) / torch.sqrt(
                torch.sum(pt_a * pt_a) * torch.sum(pt_b * pt_b)
            )
            return torch.arccos(torch.clip(cos_ab, -1.0, +1.0))

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        if self._score_method == "euclidean":
            return pt_a * (1 - frac) + pt_b * frac
        elif self._score_method == "angular":
            return slerp(pt_a, pt_b, frac)

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def to_tangent(self, pt_a: Point, vec_w: Vector):
        """Project to tangent in the pre-shape space. Pre-shapes are equivalent translation (and
        scale if using angular distance), but not rotation.

        :param pt_a: base point for the tangent vector
        :param vec_w: ambient space vector
        :return: tangent vector with mean-shifts removed, as well as scaling removed (if
            score_method=angular)
        """
        # Points must be 'centered', so subtract off component of vec that would affect the mean
        vec_w = vec_w - torch.mean(vec_w, dim=0)
        # In 'angular' case, subtract off component that would uniformly scale all points (component of the tangent
        # in the direction of pt_a)
        if self._score_method == "angular":
            vec_w = vec_w - pt_a * torch.sum(vec_w * pt_a) / torch.sum(pt_a * pt_a)
        return vec_w

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        return torch.sum(vec_w * vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        if self._score_method == "euclidean":
            return pt_a + vec_w
        elif self._score_method == "angular":
            # Identical to Hypersphere.exp_map
            # See https://math.stackexchange.com/a/1930880
            norm = self.norm(pt_a, vec_w)
            c1 = torch.cos(norm)
            c2 = torch.sinc(norm / np.pi)
            return c1 * pt_a + c2 * vec_w

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        if self._score_method == "euclidean":
            return pt_b - pt_a
        elif self._score_method == "angular":
            # Identical to Hypersphere.log_map
            unscaled_w = self.to_tangent(pt_a, pt_b)
            norm_w = unscaled_w / torch.clip(self.norm(pt_a, unscaled_w), 1e-7)
            return norm_w * self.length(pt_a, pt_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        if self._score_method == "euclidean":
            return vec_w
        elif self._score_method == "angular":
            # Identical to Hypersphere.levi_civita
            vec_v = self.log_map(pt_a, pt_b)
            angle = self.length(pt_a, pt_b)
            unit_v = vec_v / torch.clip(
                angle, 1e-7
            )  # the length of tangent vector v *is* the length from a to b
            w_along_v = torch.sum(unit_v * vec_w)
            orth_part = vec_w - w_along_v * unit_v
            return orth_part + w_along_v * (
                torch.cos(angle) * unit_v - torch.sin(angle) * pt_a
            )


class ShapeMetric(PreShapeMetric):
    """Compute the generalized shape-metrics advocated by [1]

    Implementation is paraphrased from https://github.com/ahwillia/netrep/, but follows our
    interface rather than the scikit-learn interface. Inspiration is also taken from
    https://github.com/geomstats/.

    Note on the practical differences between PreShape and Shape:
    - Shape space decomposes the PreShape tangent space into vertical (within equivalence class) and
        horizontal (across equivalence class) parts.
    - Shape.to_tangent is not overridden, so Shape.to_tangent(pt, vec) will in general contain both
        horz and vert parts
    - Shape.exp_map and Shape.levi_civita both respect the vertical part
    - Shape.log_map returns *only* the horizontal part
    - Shape.inner_product only takes the horizontal part
    This means that exp_map and log_map are not exact inverses up to _equality_. However, they are
    inverses up to _equivalence_.

    Known bug (!) ShapeMetrics do not have consistent geometry with alpha<1.0 until we add some
    fancier geometry to handle the fact that (partial) whitening induces additional (partial)
    constraints on the space.

    [1] Williams, A. H., Kunz, E., Kornblith, S., & Linderman, S. W. (2021). Generalized Shape
        Metrics on Neural Representations. NeurIPS. https://arxiv.org/abs/2110.14739
    """

    def string_id(self) -> str:
        return (
            f"ShapeMetric[{self._alpha:.2f}][{self.p}][{self._score_method}].{self.m}"
        )

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        # Length in shape space = length in pre shape space after aligning points to each other
        return super(ShapeMetric, self)._length_impl(
            *_orthogonal_procrustes(pt_a, pt_b)
        )

    #########################################
    # Implement GeodesicLengthSpace methods #
    #########################################

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        # Choice of anchor here is largely arbitrary, but for local consistency with log_map we
        # set it to 'a'
        return super(ShapeMetric, self)._geodesic_impl(
            *_orthogonal_procrustes(pt_a, pt_b, anchor="a"), frac
        )

    #####################################
    # Implement RiemannianSpace methods #
    #####################################

    def _horizontal_tangent(
        self, pt_a: Point, vec_w: Vector, *, vert_part: Vector = None
    ) -> Vector:
        """The 'horizontal' part of the tangent space is the part that is actually movement in the
        quotient space, i.e. across equivalence classes.

        For example, east/west movement where equivalence = lines of longitude.
        """
        # Start by ensuring vec_w is a tangent vector in the pre-shape space
        vec_w = super(ShapeMetric, self).to_tangent(pt_a, vec_w)
        if vert_part is None:
            # Calculate vertical part
            vert_part = self._vertical_tangent(pt_a, vec_w)
        # The horizontal part is whatever is left after projecting away the vertical part
        square_vert_norm = torch.clip(torch.sum(vert_part * vert_part), 1e-7)
        horz_part = vec_w - vert_part * torch.sum(vec_w * vert_part) / square_vert_norm
        return horz_part

    def _solve_skew_symmetric_vertical_tangent(self, pt_a: Point, vec_w: Vector):
        """Find A such that x@A is the vertical part of vec_w at pt_a."""
        # Start by ensuring vec_w is a tangent vector in the pre-shape space
        vec_w = super(ShapeMetric, self).to_tangent(pt_a, vec_w)
        # See equation (2) in Nava-Yazdani et al (2020), but note that all of our equations are
        # transposed from theirs
        xxT = pt_a.T @ pt_a
        wxT = vec_w.T @ pt_a
        return _solve_sylvester(xxT, xxT, wxT - wxT.T)

    def _vertical_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        """The 'vertical' part of the tangent space is the part that doesn't count as movement in
        the quotient space, i.e. within equivalence classes. For example, north/south movement where
        equivalence = lines of longitude.

        The space of 'vertical' tangents, after accounting for shifts and scales with
        _aux_to_tangent, is the set of rotations. We get these by looking at the span of all 2D
        rotations – one per pair of axes in our space.
        """
        return pt_a @ self._solve_skew_symmetric_vertical_tangent(pt_a, vec_w)

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        # Ensure that we're only measuring the 'horizontal' part of each tangent vector. (We
        # expect distance between two points to be equal to square root norm of the logarithmic
        # map between them).
        h_vec_w, h_vec_v = self._horizontal_tangent(
            pt_a, vec_w
        ), self._horizontal_tangent(pt_a, vec_v)
        return super(ShapeMetric, self).inner_product(pt_a, h_vec_w, h_vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # Decompose into horizontal and vertical parts. The vertical part specifies a rotation in
        # the sense that Skew-Symmetric matrices are the tangent space of SO(p), and the vertical
        # part equals Ax for some skew-symmetric matrix A. We get from skew-symmetry to rotation
        # using the matrix exponential, i.e. rotation_matrix = matrix_exp(skew_symmetric_matrix)
        mat_a = self._solve_skew_symmetric_vertical_tangent(pt_a, vec_w)
        rotation = torch.matrix_exp(mat_a)
        horz_part = self._horizontal_tangent(pt_a, vec_w, vert_part=pt_a @ mat_a)
        # Apply vertical part, and note that rotation is equivariant with respect to horizontal
        # vectors, or horz_Rx(Rw)=Rhorz_x(w). This means that we (1) rotate pt_a to pt_a',
        # and (2) the new horizontal vector at pt_a' is equal to the rotation applied to the
        # original horizontal vector
        pt_a, horz_part = pt_a @ rotation, horz_part @ rotation
        # After applying the vertical part, delegate to the ambient PreShapeSpace for the
        # remaining horizontal part
        return super(ShapeMetric, self).exp_map(pt_a, horz_part)

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        # Only returns *horizontal* part of the tangent. Note that this means log_map and exp_map
        # are not inverses from the perspective of the PreShapeSpace, but they are in the
        # ShapeSpace. In other words, if c=exp_map(a,log_map(a,b)), then we'll have length(b,
        # c)=0 but not b==c Method: align b to a and get a-->b' horizontal part from the
        # PreShapeSpace's log_map
        _, new_b = _orthogonal_procrustes(pt_a, pt_b, anchor="a")
        return super(ShapeMetric, self).log_map(pt_a, new_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Both the horizontal and vertical parts of tangent vectors are equivariant after
        # rotation (Lemma 1b of Nava-Yazdani et al (2020)). This means we can start by aligning a
        # to b as follows to take care of the vertical part, then all that's left is to transport
        # the horizontal part:
        r_a, _ = _orthogonal_procrustes_rotation(pt_a, pt_b, anchor="b")
        new_pt_a, new_vec_w = pt_a @ r_a, vec_w @ r_a
        return super(ShapeMetric, self).levi_civita(new_pt_a, pt_b, new_vec_w)


class EuclideanShapeMetric(ShapeMetric):
    """Specialization of ShapeMetric which automatically sets score_method='euclidean'."""

    def __init__(self, *args, **kwargs):
        super(EuclideanShapeMetric, self).__init__(
            *args, score_method="euclidean", **kwargs
        )


class AngularShapeMetric(ShapeMetric):
    """Specialization of ShapeMetric which automatically sets score_method='angular'."""

    def __init__(self, *args, **kwargs):
        super(AngularShapeMetric, self).__init__(
            *args, score_method="angular", **kwargs
        )


def _orthogonal_procrustes_rotation(a, b, anchor="middle"):
    """Provided a and b, each matrix of size (m, p) that are already centered and scaled, solve the
    orthogonal procrustest problem (rotate a and b into a common frame that minimizes distances).

    If anchor="middle" (default) then both a and b If anchor="a", then a is left unchanged and b is
    rotated towards it If anchor="b", then b is left unchanged and a is rotated towards it

    See
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: r_a and r_b, which, when right-multiplied with a and b, gives the aligned coordinates,
        or None for each if no transform is required
    """
    with torch.no_grad():
        u, _, v = torch.linalg.svd(a.T @ b)
    # Helpful trick to see how these are related: u is the inverse of u.T, and likewise v is
    # inverse of v.T. We get to the anchor=a and anchor=b solutions by right-multiplying both
    # return values by u.T or right-multiplying both return values by v, respectively (if both
    # return values are rotated in the same way, it preserves the shape).
    if anchor == "middle":
        return u, v.T
    elif anchor == "a":
        return None, v.T @ u.T
    elif anchor == "b":
        return u @ v, None
    else:
        raise ValueError(
            f"Invalid 'anchor' argument: {anchor} (must be 'middle', 'a', or 'b')"
        )


def _orthogonal_procrustes(a, b, anchor="middle"):
    """Provided a and b, each matrix of size (m, p) that are already centered and scaled, solve the
    orthogonal procrustest problem (rotate a and b into a common frame that minimizes distances).

    If anchor="middle" (default) then both a and b If anchor="a", then a is left unchanged and b is
    rotated towards it If anchor="b", then b is left unchanged and a is rotated towards it

    See
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :return: new_a, new_b the rotated versions of a and b, minimizing element-wise squared
        differences
    """
    r_a, r_b = _orthogonal_procrustes_rotation(a, b, anchor)
    return a @ r_a if r_a is not None else a, b @ r_b if r_b is not None else b


def _whiten(x, alpha, clip_eigs=1e-9):
    """Compute (partial) whitening transform of x. When alpha=0 it is classic ZCA whitening and
    columns of x are totally decorrelated. When alpha=1, nothing happens.

    Assumes x is already centered.
    """
    e, v = eigh(x.T @ x / len(x))
    e = torch.clip(e, min=clip_eigs, max=None)
    d = alpha + (1 - alpha) * (e**-0.5)
    # From right to left, the transformation (1) projects x onto v, (2) divides by stdev in each
    # direction, and (3) rotates back to align with original directions in x-space (ZCA)
    z = v @ torch.diag(d) @ v.T
    # Think of this as (z @ x.T).T, but note z==z.T
    return x @ z


def _dim_reduce(x, p):
    # PCA to truncate -- project onto top p principal axes (no rescaling)
    with torch.no_grad():
        _, _, vT = torch.linalg.svd(x, full_matrices=False)
    # svd returns v.T, so the principal axes are in the *rows*. The following einsum is
    # equivalent to x @ vT.T[:, :p] but a bit faster because the transpose is not actually
    # performed.
    return torch.einsum("mn,pn->mp", x, vT[:p, :])


def _pad_zeros(x, p):
    m, d = x.size()
    num_pad = p - d
    return torch.hstack([x.view(m, d), x.new_zeros(m, num_pad)])


def _solve_sylvester(a, b, q):
    # TODO - implement natively in pytorch so we don't have to convert to numpy on CPU and back
    #  again
    a_np, b_np, q_np = (
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        q.detach().cpu().numpy(),
    )
    return torch.tensor(
        solve_sylvester(a_np, b_np, q_np), dtype=a.dtype, device=a.device
    )
