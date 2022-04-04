import torch
from .curvature import _bisector_length


def test_triangle_calculation_2d():
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
