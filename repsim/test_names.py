from repsim import AngularCKA, AffineInvariantRiemannian, Stress, AngularShapeMetric, EuclideanShapeMetric
from repsim.kernels import SquaredExponential


def test_names():
    # BASICS - DEFAULT KERNELS
    metric = AngularCKA(m=1000)
    assert metric.string_id() == "AngularCKA.Linear.1000"

    metric = AffineInvariantRiemannian(m=1000)
    assert metric.string_id() == "AffineInvariantRiemannian[gram].Linear.1000"

    metric = AffineInvariantRiemannian(m=1000, eps=0.05)
    assert metric.string_id() == "AffineInvariantRiemannian[gram][0.050].Linear.1000"

    metric = AffineInvariantRiemannian(m=1000, eps=0.05, mode="cov", p=100)
    assert metric.string_id() == "AffineInvariantRiemannian[cov][0.050][100].1000"

    metric = Stress(m=1000)
    assert metric.string_id() == "Stress.Linear.1000"

    # KERNEL WITH AUTO SCALE
    metric = AngularCKA(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AngularCKA.SqExp[auto].1000"

    metric = AffineInvariantRiemannian(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AffineInvariantRiemannian[gram].SqExp[auto].1000"

    metric = Stress(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "Stress.SqExp[auto].1000"

    # KERNEL WITH LENGTH SCALE
    metric = AngularCKA(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AngularCKA.SqExp[0.300].100"

    metric = AffineInvariantRiemannian(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AffineInvariantRiemannian[gram].SqExp[0.300].100"

    metric = Stress(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "Stress.SqExp[0.300].100"

    # SHAPE METRICS
    metric = AngularShapeMetric(m=1000, p=100)
    assert metric.string_id() == "ShapeMetric[1.00][100][angular].1000"

    metric = EuclideanShapeMetric(m=1000, p=100)
    assert metric.string_id() == "ShapeMetric[1.00][100][euclidean].1000"

    metric = AngularShapeMetric(m=100, p=40, alpha=0.5)
    assert metric.string_id() == "ShapeMetric[0.50][40][angular].100"

    metric = EuclideanShapeMetric(m=100, p=40, alpha=0.5)
    assert metric.string_id() == "ShapeMetric[0.50][40][euclidean].100"
