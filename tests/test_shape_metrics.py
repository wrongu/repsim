import numpy as np
from repsim.metrics.generalized_shape_metrics import ShapeMetric, _dim_reduce, _pad_zeros, _orthogonal_procrustes
from tests.constants import atol, rtol
import pytest


def test_shape_metric_alpha(metric, data_x, data_y):
    if not isinstance(metric, ShapeMetric):
        pytest.skip()
    elif metric._alpha < 1.0:
        pytest.xfail(reason="known bug: shape metrics with partial whitening (with alpha < 1.0) do not properly handle "
                            "length/geodesics in a way that respects homotopy, which is what this test asserts")

    # Do some initial standardization to get x and y to be aligned in their top p principal components.
    _, d = data_x.size()
    if d <= metric.p:
        x, y = _pad_zeros(data_x, metric.p), _pad_zeros(data_y, metric.p)
    else:
        x, y = _dim_reduce(data_x, metric.p), _dim_reduce(data_y, metric.p)
    x, y = _orthogonal_procrustes(x, y, anchor="a")

    # Idea of this test is that, given the standardization of the previous few lines, we ought to get the same result
    # whether we interpolate-then-embed or if we embed-then-interpolate. The way that this can fail is if the
    # embedding step includes some curvature not accounted for by the length() function. As an analogy, imagine that
    # our space is R^3 and that the embedding function is f(x)=a*x + (1-a)*x/||x||. In other words, a=0 means f(x)
    # embeds on a sphere and a=1 means euclidean. for intermediate values of 'a', we would like the geodesics to not
    # take straight path 'shortcuts'. So, we are asserting that length(f(x),f(x')) is to equal length(f(x),
    # f((x+x')/2))+length(f((x+x')/2),f(x')))
    # Visualization of the problem: https://colab.research.google.com/drive/13-kd_RQYcpiuemHKXCX4_l1iQCt1vImE
    mid = (x + y) / 2
    pt_x, pt_y, pt_mid = metric.neural_data_to_point(x), metric.neural_data_to_point(y), metric.neural_data_to_point(mid)
    piecewise_length = metric.length(pt_x, pt_mid) + metric.length(pt_mid, pt_y)
    xy_length = metric.length(pt_x, pt_y)
    assert np.isclose(piecewise_length, xy_length, atol=atol, rtol=rtol), \
        f"With z=(x+y)/2, expected length(f(x),f(z))+length(f(z),f(y)) = length(f(x),f(y))"
