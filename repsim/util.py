import torch
import enum


class CompareType(enum.Enum):
    """Comparison type for repsim.compare and repsim.pairwise.compare.

    CompareType.INNER_PRODUCT: an inner product like x @ y.T. Large values = more similar.
    CompareType.ANGLE: values are 'distances' in [0, pi/2]
    CompareType.DISTANCE: a distance, like ||x-y||. Small values = more similar.
    CompareType.SQUARE_DISTANCE: squared distance.

    Note that INNER_PRODUCT has a different sign than the others, indicating that high inner-product means low distance
    and vice versa.
    """

    INNER_PRODUCT = -1
    ANGLE = 0
    DISTANCE = 1
    SQUARE_DISTANCE = 2


class MetricType(enum.Enum):
    """Different levels of strictness for measuring distance from x to y.

    MetricType.CORR: the result isn't a metric but a correlation in [-1, 1]. Large values indicate low 'distance', sort of.
    MetricType.PRE_METRIC: a function d(x, y) that satisfies d(x,x)=0 and x(d,y)>=0
    MetricType.METRIC: a pre-metric that also satisfies the triangle inequality: d(x,y)+d(y,z)>=d(x,z)
    MetricType.LENGTH: a metric that is equal to the length of a shortest-path
    MetricType.RIEMANN: a length along a Riemannian manifold
    MetricType.ANGLE: angle between vectors, which is by definition an arc length (along a sphere surface)

    Note that each of these specializes the ones above it, which is why each of the Enum values is constructed as a bit
    mask: in binary, PRE_METRIC is 00001, METRIC is 00011, LENGTH is 00111, ANGLE is 10111, and RIEMANN is 01111
    """

    CORR = 0x00
    PRE_METRIC = 0x01
    METRIC = 0x03
    LENGTH = 0x07
    RIEMANN = 0x0F
    ANGLE = 0x17


class CorrType(enum.Enum):
    PEARSON = 1
    SPEARMAN = 2


def pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise distances (x-y)*(x-y)

    :param x: n by d matrix of d-dimensional values
    :param y: m by d matrix of d-dimensional values
    :return: n by m matrix of squared pairwise distances
    """
    n, d = x.size()
    if y.size()[-1] != d:
        raise ValueError("x and y must have same second dimension")
    xx = torch.sum(x * x, dim=1)
    yy = torch.sum(y * y, dim=1)
    xy = torch.einsum("nd,md->nm", x, y)
    # Using (x-y)*(x-y) = x*x + y+y - 2*x*y, and clipping in [0, inf) just in case of numerical imprecision
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


def corrcoef(
    a: torch.Tensor, b: torch.Tensor, type: CorrType = CorrType.PEARSON
) -> torch.Tensor:
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
        return corrcoef(
            torch.argsort(a).float(), torch.argsort(b).float(), type=CorrType.PEARSON
        )
