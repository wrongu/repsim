import torch
from repsim.geometry.geodesic import midpoint
from repsim.geometry import LengthSpace, Point, Scalar


def _bisector_length(x: Scalar, y: Scalar, z: Scalar) -> Scalar:
    """Given a triangle ABC in Euclidean space with side lengths AB=x, BC=y, AC=z, returns the
    length of BD, where D is the midpoint of AC."""
    return torch.sqrt(x * x / 2 + y * y / 2 - z * z / 4)


def alexandrov(space: LengthSpace, pt_a: Point, pt_b: Point, pt_c: Point) -> Scalar:
    """Compute Alexandrov curvature from three points in the space.

    :param space: a LengthSpace that defines the metric and geodesics
    :param pt_a: a point in the space
    :param pt_b: a point in the space distinct from pt_a
    :param pt_c: a point in the space distinct from pt_a and pt_c
    :return: length of (pt_b, pt_d) minus expected length in a Euclidean space, where pt_d is
        midpoint of (pt_a, pt_c)
    """
    pt_d = midpoint(space, pt_a, pt_c)
    bisector_lenth = space.length(pt_b, pt_d)
    euclidean_bisector_length = _bisector_length(
        space.length(pt_a, pt_b), space.length(pt_b, pt_c), space.length(pt_a, pt_c)
    )
    return bisector_lenth - euclidean_bisector_length


__all__ = ["alexandrov"]


if __name__ == "__main__":
    # sanity-check triangle calculation in 2D
    a, b, c = torch.randn(2), torch.randn(2), torch.randn(2)
    d = (a + c) / 2
    x, y, z, d = (
        torch.linalg.norm(a - b),
        torch.linalg.norm(b - c),
        torch.linalg.norm(a - c),
        torch.linalg.norm(b - d),
    )
    print(d, _bisector_length(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])))
