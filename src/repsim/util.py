import torch
import enum


class CompareType(enum.Enum):
    """Comparison type for repsim.compare and repsim.pairwise.compare.

    CompareType.INNER_PRODUCT: an inner product like x @ y.T. Large values = more similar.
    CompareType.ANGLE: values are 'distances' in [0, pi/2] CompareType.COSINE: values are cosine of
    ANGLE, i.e. inner-product of unit vectors CompareType.DISTANCE: a distance, like ||x-y||. Small
    values = more similar. CompareType.SQUARE_DISTANCE: squared distance.

    Note that INNER_PRODUCT has a different sign than the others, indicating that high inner-product
    means low distance and vice versa.
    """

    INNER_PRODUCT = -1
    ANGLE = 0
    COSINE = 1
    DISTANCE = 2
    SQUARE_DISTANCE = 3


def pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise distances (x-y)*(x-y)

    :param x: n by d matrix of d-dimensional values
    :param y: m by d matrix of d-dimensional values
    :return: n by m matrix of squared pairwise distances
    """
    _, dx = x.size()[0], x.size()[1:]
    _, dy = x.size()[0], x.size()[1:]
    if dx != dy:
        raise ValueError(
            f"x and y must have same second dimension but are {dx} and {dy}"
        )
    xx = torch.einsum("i...,i...->i", x, x)
    yy = torch.einsum("j...,j...->j", y, y)
    xy = torch.einsum("i...,j...->ij", x, y)
    # Using (x-y)*(x-y) = x*x + y+y - 2*x*y, and clipping in [0, inf) just in case of numerical
    # imprecision
    return torch.clip(xx[:, None] + yy[None, :] - 2 * xy, 0.0, None)


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


def prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def eig_fun(hmat, fun):
    """Apply a function to the eigenvalues of a symmetric (hermitian) matrix."""
    e, u = torch.linalg.eigh(hmat)
    return u @ torch.diag(fun(e)) @ u.T


def inv_matrix(hmat):
    return eig_fun(hmat, fun=lambda x: 1.0 / x)


def matrix_sqrt(hmat):
    return eig_fun(hmat, fun=torch.sqrt)


def inv_matrix_sqrt(hmat):
    return eig_fun(hmat, fun=lambda x: 1.0 / torch.sqrt(x))


def matrix_log(hmat):
    return eig_fun(hmat, fun=torch.log)


def matrix_exp(hmat):
    return eig_fun(hmat, fun=torch.exp)


def matrix_pow(hmat, p):
    return eig_fun(hmat, fun=lambda e: torch.pow(e, p))
