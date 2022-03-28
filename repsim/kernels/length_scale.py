import tensorly as tl
from repsim.util import pdist2, upper_triangle


def median_euclidean(x: tl.tensor, y=None, fraction=1.0):
    if y is None:
        # Compare rows of x to other rows of x
        return fraction * tl.sqrt(tl.median(upper_triangle(pdist2(x, x))))
    else:
        # Compare x to y
        return fraction * tl.sqrt(tl.median(pdist2(x, y)))
