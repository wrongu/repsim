import torch
import numpy as np
from repsim.geometry import Point, Scalar, Vector, RiemannianSpace
from repsim.geometry.trig import slerp


class HyperSphere(RiemannianSpace):
    """Class for handling geometric operations on an n-dimensional hypersphere."""

    def __init__(self, dim):
        # a dim-dimensional sphere has points that live in dim+1-dimensional space
        super().__init__(dim=dim, shape=(dim + 1,))

    def _geodesic_impl(self, pt_a: Point, pt_b: Point, frac: float = 0.5) -> Point:
        return slerp(pt_a, pt_b, frac)

    def _project_impl(self, pt: Point) -> Point:
        return pt / torch.sqrt(torch.sum(pt * pt))

    def _contains_impl(self, pt: Point, atol: float = 1e-6) -> bool:
        radius = torch.sqrt(torch.sum(pt * pt, dim=-1))
        return torch.abs(radius - 1.0) <= atol

    def _length_impl(self, pt_a: Point, pt_b: Point) -> Scalar:
        dot_ab = torch.dot(pt_a, pt_b)
        len_a, len_b = torch.sqrt(torch.dot(pt_a, pt_a)), torch.sqrt(
            torch.dot(pt_b, pt_b)
        )
        cosine = dot_ab / len_a / len_b
        return torch.arccos(torch.clip(cosine, -1.0, +1.0))

    def to_tangent(self, pt_a: Point, vec_w: Vector) -> Vector:
        dot_a_w = torch.sum(pt_a * vec_w)
        return vec_w - dot_a_w * pt_a

    def inner_product(self, pt_a: Point, vec_w: Vector, vec_v: Vector):
        # Just the usual inner product in the ambient space
        return torch.sum(vec_w * vec_v)

    def exp_map(self, pt_a: Point, vec_w: Vector) -> Point:
        # See https://math.stackexchange.com/a/1930880
        vec_w = self.to_tangent(pt_a, vec_w)
        norm = torch.sqrt(torch.sum(vec_w * vec_w))
        c1 = torch.cos(norm)
        c2 = torch.sinc(norm / np.pi)
        return c1 * pt_a + c2 * vec_w

    def log_map(self, pt_a: Point, pt_b: Point) -> Vector:
        unscaled_w = self.to_tangent(pt_a, pt_b)
        norm_w = unscaled_w / torch.clip(
            torch.sqrt(torch.sum(unscaled_w * unscaled_w)), 1e-7
        )
        return norm_w * self.length(pt_a, pt_b)

    def levi_civita(self, pt_a: Point, pt_b: Point, vec_w: Vector) -> Vector:
        # Idea: decompose the tangent vector w into (i) a part that is orthogonal to the
        # transport direction, and (ii) a part along the transport direction. The orthogonal part
        # will be unchanged through the map, and the parallel part will be rotated in the plane
        # spanned by pt_a and the unit v. (thanks to the geomstats package for reference
        # implementation)
        vec_v = self.log_map(pt_a, pt_b)
        angle = self.length(pt_a, pt_b)
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
