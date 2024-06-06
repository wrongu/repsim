from repsim import (
    AngularCKA,
    AffineInvariantRiemannian,
    Stress,
    AngularShapeMetric,
    EuclideanShapeMetric,
)
from repsim.kernels import SquaredExponential


def test_name(metric):
    # Note: the expected value of each test_name is defined in tests/metrics.py
    assert metric.string_id() == metric.test_name
