import numpy as np
from repsim.metrics.generalized_shape_metrics import ShapeMetric, _orthogonal_procrustes
from tests.constants import atol, rtol
import pytest


def test_shape_metric_alpha(metric, data_x, data_y):
    if not isinstance(metric, ShapeMetric):
        pytest.skip()

    # Idea of this test is that if we begin by aligning points (x and y), then we ought to get the same result
    # whether we interpolate-then-embed or if we embed-then-interpolate. The way that this can fail is if the
    # embedding step includes some curvature not accounted for by the length() function. As an analogy, imagine that
    # our space is R^3 and that the embedding function is f(x)=a*x + (1-a)*x/||x||. In other words, a=0 means f(x)
    # embeds on a sphere and a=1 means euclidean. for intermediate values of 'a', we would like the geodesics to not
    # take straight path 'shortcuts'. So, we are asserting that length(f(x),f(x')) is to equal length(f(x),
    # f((x+x')/2))+length(f((x+x')/2),f(x'))
    x, y = _orthogonal_procrustes(data_x, data_y, anchor="a")
    mid = (x + y) / 2
    pt_x, pt_y, pt_mid = metric.neural_data_to_point(x), metric.neural_data_to_point(y), metric.neural_data_to_point(mid)
    piecewise_length = metric.length(pt_x, pt_mid) + metric.length(pt_mid, pt_y)
    xy_length = metric.length(pt_x, pt_y)
    assert np.isclose(piecewise_length, xy_length, atol=atol, rtol=rtol), \
        f"With z=(x+y)/2, expected length(f(x),f(z))+length(f(z),f(y)) = length(f(x),f(y))"
