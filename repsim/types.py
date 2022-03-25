import enum


class CompareType(enum.Enum):
    """Comparison type for repsim.compare and repsim.pairwise.compare.

    CompareType.INNER_PRODUCT: an inner product like dot(x,y). Large values = more similar.
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
    RIEMANN = 0x0f
    ANGLE = 0x17


class CorrType(enum.Enum):
    PEARSON = 1
    SPEARMAN = 2
