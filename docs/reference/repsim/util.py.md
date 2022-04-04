## *Class* `CompareType`


Comparison type for repsim.compare and repsim.pairwise.compare.

> - **CompareType.INNER_PRODUCT** (`None`: `None`): an inner product like x @ y.T. Large values = more similar.
> - **CompareType.ANGLE** (`None`: `None`): values are 'distances' in [0, pi/2]
> - **CompareType.DISTANCE** (`None`: `None`): a distance, like ||x-y||. Small values = more similar.
> - **CompareType.SQUARE_DISTANCE** (`None`: `None`): squared distance.

Note that INNER_PRODUCT has a different sign than the others, indicating that high inner-product means low distance and vice versa.


## *Class* `MetricType`


Different levels of strictness for measuring distance from x to y.

> - **MetricType.CORR** (`None`: `None`): the result isn't a metric but a correlation in [-1, 1]. Large values indicate low 'distance', sort of.
> - **MetricType.PRE_METRIC** (`None`: `None`): a function d(x, y) that satisfies d(x,x)=0 and x(d,y)>=0
> - **MetricType.METRIC** (`None`: `None`): a pre-metric that also satisfies the triangle inequality: d(x,y)+d(y,z)>=d(x,z)
> - **MetricType.LENGTH** (`None`: `None`): a metric that is equal to the length of a shortest-path
> - **MetricType.RIEMANN** (`None`: `None`): a length along a Riemannian manifold
> - **MetricType.ANGLE** (`None`: `None`): angle between vectors, which is by definition an arc length (along a sphere surface)

Note that each of these specializes the ones above it, which is why each of the Enum values is constructed as a bit
> - **mask** (`None`: `None`): in binary, PRE_METRIC is 00001, METRIC is 00011, LENGTH is 00111, ANGLE is 10111, and RIEMANN is 01111


## *Class* `CorrType(enum.Enum):
    PEARSON = 1
    SPEARMAN = 2


def pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor`


Compute squared pairwise distances (x-y)*(x-y)

> - **x** (`None`: `None`): n by d matrix of d-dimensional values
> - **y** (`None`: `None`): m by d matrix of d-dimensional values
> - **return** (`None`: `None`): n by m matrix of squared pairwise distances


### *Function* `upper_triangle(A: torch.Tensor, offset=1) -> torch.Tensor ()`


Get the upper-triangular elements of a square matrix.

> - **A** (`None`: `None`): square matrix
> - **offset** (`None`: `None`): number of diagonals to exclude, including the main diagonal. Deafult is 1.
> - **return** (`None`: `None`): a 1-dimensional torch Tensor containing upper-triangular values of A


### *Function* `corrcoef(
    a: torch.Tensor, b: torch.Tensor, type: CorrType = CorrType.PEARSON
) -> torch.Tensor ()`


Correlation coefficient between two vectors.

> - **a** (`None`: `None`): a 1-dimensional torch.Tensor of values
> - **b** (`None`: `None`): a 1-dimensional torch.Tensor of values
> - **return** (`None`: `None`): correlation between a and b
