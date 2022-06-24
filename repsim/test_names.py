from repsim import AngularCKA, AffineInvariantRiemannian, Stress
from repsim.kernels import SquaredExponential


def test_names():
    # BASICS - DEFAULT KERNELS
    metric = AngularCKA(m=1000)
    assert metric.string_id() == "AngularCKA.Linear.1000"

    metric = AffineInvariantRiemannian(m=1000)
    assert metric.string_id() == "AffineInvariantRiemannian.Linear.1000"

    metric = AffineInvariantRiemannian(m=1000, shrinkage=0.1)
    assert metric.string_id() == "AffineInvariantRiemannian[0.100].Linear.1000"

    metric = Stress(m=1000)
    assert metric.string_id() == "Stress.Linear.1000"

    # KERNEL WITH AUTO SCALE
    metric = AngularCKA(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AngularCKA.SqExp[auto].1000"

    metric = AffineInvariantRiemannian(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AffineInvariantRiemannian.SqExp[auto].1000"

    metric = Stress(m=1000, kernel=SquaredExponential())
    assert metric.string_id() == "Stress.SqExp[auto].1000"

    # KERNEL WITH LENGTH SCALE
    metric = AngularCKA(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AngularCKA.SqExp[0.300].100"

    metric = AffineInvariantRiemannian(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AffineInvariantRiemannian.SqExp[0.300].100"

    metric = Stress(m=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "Stress.SqExp[0.300].100"
