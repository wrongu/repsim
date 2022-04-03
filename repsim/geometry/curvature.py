import torch
from repsim.geometry.manifold import Manifold, Point, Scalar
from repsim.geometry.geodesic import midpoint
import warnings


def _bisector_length(x: Scalar, y: Scalar, z: Scalar) -> Scalar:
    """Given a triangle ABC with side lengths AB=x, BC=y, AC=z, returns
    the length of BD, where D is the midpoint of AC
    """
    return torch.sqrt(x*x/2 + y*y/2 - z*z/4)


def alexandrov(pt_a: Point,
               pt_b: Point,
               pt_c: Point,
               space: Manifold) -> Scalar:
    midpt_ac, converged = midpoint(pt_a, pt_c, space)
    if not converged:
        warnings.warn("midoint() failed to converge. Curvature may be inaccurate.")
    bisector_lenth = space.length(pt_b, midpt_ac)
    euclidean_bisector_length = _bisector_length(space.length(pt_a, pt_b),
                                                 space.length(pt_b, pt_c),
                                                 space.length(pt_a, pt_c))
    return (bisector_lenth - euclidean_bisector_length) / euclidean_bisector_length


if __name__ == "__main__":
    # sanity-check triangle calculation in 2D
    a, b, c = torch.randn(2), torch.randn(2), torch.randn(2)
    d = (a+c)/2
    x, y, z, d = torch.linalg.norm(a-b), torch.linalg.norm(b-c), torch.linalg.norm(a-c), torch.linalg.norm(b-d)
    print(d, _bisector_length(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])))
