from repsim import AngularCKA, AffineInvariantRiemannian, Stress, ScaleInvariantStress
from repsim.kernels import Linear, Laplace, SquaredExponential


def test_names():
    # BASICS - DEFAULT KERNELS
    metric = AngularCKA(n=1000)
    assert metric.string_id() == "AngularCKA.Linear.1000"

    metric = AffineInvariantRiemannian(n=1000)
    assert metric.string_id() == "AffineInvariantRiemannian.Linear.1000"

    metric = Stress(n=1000)
    assert metric.string_id() == "Stress.Linear.1000"

    metric = ScaleInvariantStress(n=1000)
    assert metric.string_id() == "ScaleInvariantStress.Linear.1000"

    # KERNEL WITH AUTO SCALE
    metric = AngularCKA(n=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AngularCKA.SqExp[auto].1000"

    metric = AffineInvariantRiemannian(n=1000, kernel=SquaredExponential())
    assert metric.string_id() == "AffineInvariantRiemannian.SqExp[auto].1000"

    metric = Stress(n=1000, kernel=SquaredExponential())
    assert metric.string_id() == "Stress.SqExp[auto].1000"

    metric = ScaleInvariantStress(n=1000, kernel=SquaredExponential())
    assert metric.string_id() == "ScaleInvariantStress.SqExp[auto].1000"

    # KERNEL WITH LENGTH SCALE
    metric = AngularCKA(n=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AngularCKA.SqExp[0.300].100"

    metric = AffineInvariantRiemannian(n=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "AffineInvariantRiemannian.SqExp[0.300].100"

    metric = Stress(n=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "Stress.SqExp[0.300].100"

    metric = ScaleInvariantStress(n=100, kernel=SquaredExponential(length_scale=0.3))
    assert metric.string_id() == "ScaleInvariantStress.SqExp[0.300].100"