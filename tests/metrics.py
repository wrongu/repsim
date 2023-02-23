"""tests/metrics.py

The role of this file is to define a list of metrics. Every metric defined here will have every test run on it, if that
test 'requests' it by declaring a 'metric' parameter.
"""
import pytest
from repsim.kernels import SquaredExponential, Laplace, Linear
from repsim import AngularCKA, Stress, AngularShapeMetric, EuclideanShapeMetric, AffineInvariantRiemannian
from tests.constants import size_m, size_n

_list_of_metrics = [
    # Angular CKA with a linear kernel is not affine invariant
    {"metric": AngularCKA(size_m),
     "name": f"AngularCKA.Linear.{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Next two tests: assert that scale invariance is due to adaptive length scale in the kernel when using sqexp
    {"metric": AngularCKA(size_m, kernel=SquaredExponential(length_scale='median')),
     "name": f"AngularCKA.SqExp[median].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularCKA(size_m, kernel=SquaredExponential(length_scale=0.5)),
     "name": f"AngularCKA.SqExp[0.500].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    # Next two tests: Stress with a linear kernel is scale-invariant only if rescale=True is set
    {"metric": Stress(size_m, kernel=Linear()),
     "name": f"Stress.Linear.{size_m}",
     "expected-curvature": "zero",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    {"metric": Stress(size_m, kernel=Linear(), rescale=True),
     "name": f"Stress.Linear.scaled.{size_m}",
     "expected-curvature": "zero",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Next two tests: Stress with an auto-length-scale kernel is scale invariant
    {"metric": Stress(size_m, kernel=SquaredExponential()),
     "name": f"Stress.SqExp[auto].{size_m}",
     "expected-curvature": "zero",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": Stress(size_m, kernel=Laplace()),
     "name": f"Stress.Laplace[auto].{size_m}",
     "expected-curvature": "zero",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # The 'affine invariant' metric is not, in fact, affine-invariant when using a kernel
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=SquaredExponential()),
     "name": f"AffineInvariantRiemannian[gram].SqExp[auto].{size_m}",
     "expected-curvature": "negative",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Laplace()),
     "name": f"AffineInvariantRiemannian[gram].Laplace[auto].{size_m}",
     "expected-curvature": "negative",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # The 'affine invariant' metric *is* affine-invariant using a linear kernel, but only works if n > size_m
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Linear()),
     "name": f"AffineInvariantRiemannian[gram].Linear.{size_m}",
     "expected-curvature": "negative",
     "high-rank-data": True,  # (!!) AffineInvariantRiemannian with a Linear kernel requires n > m
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    # ...If we instead use a Linear kernel with eps regularization, we lose the scale and affine invariance again
    {"metric": AffineInvariantRiemannian(m=size_m, kernel=Linear(), eps=0.01),
     "name": f"AffineInvariantRiemannian[gram][0.010].Linear.{size_m}",
     "expected-curvature": "negative",
     "high-rank-data": True,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": False,
     "affine-invariant": False},
    # Shape metrics with alpha=0 (full whitening) are invariant to everything...
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=0.0),
     "name": f"ShapeMetric[0.00][{size_n}][euclidean].{size_m}",
     "expected-curvature": "positive",  # (!!) EuclideanShapeMetric curvature ≥ 0 when alpha < 1
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    # ...unless p<n, in which case they will in general lose affine invariance because top-p PCs changed
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n//2, alpha=0.0),
     "name": f"ShapeMetric[0.00][{size_n//2}][euclidean].{size_m}",
     "expected-curvature": "positive",  # (!!) EuclideanShapeMetric curvature ≥ 0 when alpha < 1
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # Shape metrics are also not affine-invariant when doing partial (alpha=0.5) or no (alpha=1.0) whitening
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=0.5),
     "name": f"ShapeMetric[0.50][{size_n}][euclidean].{size_m}",
     "expected-curvature": "positive",  # (!!) EuclideanShapeMetric curvature ≥ 0 when alpha < 1
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": EuclideanShapeMetric(m=size_m, p=size_n, alpha=1.0),
     "name": f"ShapeMetric[1.00][{size_n}][euclidean].{size_m}",
     "expected-curvature": "zero",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    # ...repeat the last 4 tests for the AngularShapeMetric
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=0.0),
     "name": f"ShapeMetric[0.00][{size_n}][angular].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": True},
    {"metric": AngularShapeMetric(m=size_m, p=size_n//2, alpha=0.0),
     "name": f"ShapeMetric[0.00][{size_n//2}][angular].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=0.5),
     "name": f"ShapeMetric[0.50][{size_n}][angular].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
    {"metric": AngularShapeMetric(m=size_m, p=size_n, alpha=1.0),
     "name": f"ShapeMetric[1.00][{size_n}][angular].{size_m}",
     "expected-curvature": "positive",
     "high-rank-data": False,
     "rotation-invariant": True,
     "translation-invariant": True,
     "scale-invariant": True,
     "affine-invariant": False},
]

@pytest.fixture(params=_list_of_metrics, ids=lambda p: p["metric"].string_id())
def metric(request):
    m = request.param["metric"]
    # Add other properties listed in _list_of_metrics as instance variables on the metric object, prefixed with
    # "test_" to avoid naming conflicts. In other words, this loop adds properties like m.test_rotation_invariant and
    # m.test_high_rank_data based on key, value pairs in the list at the top of this file.
    d = m.__dict__
    for k, v in request.param.items():
        if k == "metric":
            continue
        k = "test_" + k.replace("-", "_")
        d[k] = v
    return m
