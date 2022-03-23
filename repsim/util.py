import torch
from repsim.types import CorrType


def pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise distances (x-y)*(x-y)

    :param x: n by d matrix of d-dimensional values
    :param y: m by d matrix of d-dimensional values
    :return: n by m matrix of squared pairwise distances
    """
    n, d = x.size()
    if y.size()[-1] != d:
        raise ValueError("x and y must have same second dimension")
    xx = torch.sum(x*x, dim=1)
    yy = torch.sum(y*y, dim=1)
    xy = torch.einsum('nd,md->nm', x, y)
    # Using (x-y)*(x-y) = x*x + y+y - 2*x*y
    return xx[:, None] + yy[None, :] - 2 * xy


def upper_triangle(A: torch.Tensor, offset=1) -> torch.Tensor:
    """Get the upper-triangular elements of a square matrix.

    :param A: square matrix
    :param offset: number of diagonals to exclude, including the main diagonal. Deafult is 1.
    :return: a 1-dimensional torch Tensor containing upper-triangular values of A
    """
    if (A.ndim != 2) or (A.size()[0] != A.size()[1]):
        raise ValueError("A must be square")

    i, j = torch.triu_indices(*A.size(), offset=offset, device=A.device)
    return A[i, j]


def corrcoef(a: torch.Tensor, b: torch.Tensor, type: CorrType = CorrType.PEARSON) -> torch.Tensor:
    """Correlation coefficient between two vectors.

    :param a: a 1-dimensional torch.Tensor of values
    :param b: a 1-dimensional torch.Tensor of values
    :return: correlation between a and b
    """
    if type == CorrType.PEARSON:
        z_a = (a - a.mean()) / a.std(dim=-1)
        z_b = (b - b.mean()) / b.std(dim=-1)
        return torch.sum(z_a * z_b)
    elif type == CorrType.SPEARMAN:
        return corrcoef(torch.argsort(a), torch.argsort(b), type=CorrType.PEARSON)
