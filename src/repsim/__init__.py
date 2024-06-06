import torch
from repsim.metrics import (
    RepresentationMetricSpace,
    Stress,
    AngularCKA,
    AffineInvariantRiemannian,
    EuclideanShapeMetric,
    AngularShapeMetric,
)
from typing import Union


def compare(
    x: torch.Tensor,
    y: torch.Tensor,
    method: Union[RepresentationMetricSpace, str] = "stress",
    **kwargs,
) -> torch.Tensor:
    metric_lookup = {
        "stress": Stress,
        "angular_cka": AngularCKA,
        "affine_invariant_riemannian": AffineInvariantRiemannian,
        "euclidean_shape_metric": EuclideanShapeMetric,
        "angular_shape_metric": AngularShapeMetric,
    }

    if isinstance(method, str):
        if method.lower() not in metric_lookup:
            raise ValueError(
                f'Unrecognized Representational Similarity Method "{method}". '
                f"Options are: {metric_lookup.keys()}"
            )
        method = metric_lookup[method.lower()](m=x.size()[0], **kwargs)
    elif not isinstance(method, RepresentationMetricSpace):
        raise ValueError(
            f"Method must be string or RepresentationMetricSpace instance, but was {type(method)}"
        )

    return method.length(method.neural_data_to_point(x), method.neural_data_to_point(y))


__all__ = [
    "compare",
    "RepresentationMetricSpace",
    "Stress",
    "AngularCKA",
    "AffineInvariantRiemannian",
    "EuclideanShapeMetric",
    "AngularShapeMetric",
]
