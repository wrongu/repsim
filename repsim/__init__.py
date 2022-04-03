import torch
from repsim.util import CorrType
from repsim.compare_impl import (
    RepresentationMetricSpace,
    Stress,
    GeneralizedShapeMetric,
    AffineInvariantRiemannian,
)
from typing import Union


def compare(
    x: torch.Tensor,
    y: torch.Tensor,
    method: Union[RepresentationMetricSpace, str] = "stress",
    **kwargs
) -> torch.Tensor:
    metric_lookup = {
        "stress": Stress,
        "generalized_shape_metric": GeneralizedShapeMetric,
        "riemannian": AffineInvariantRiemannian,
    }

    if isinstance(method, str):
        if method.lower() not in metric_lookup:
            raise ValueError(
                f'Unrecognized Representational Similarity Method "{method}". Options are: {metric_lookup.keys()}'
            )
        method = metric_lookup[method.lower()](n=x.size()[0], **kwargs)
    elif not isinstance(method, RepresentationMetricSpace):
        raise ValueError(f"Method must be string or RepresentationMetricSpace instance, but was {type(method)}")

    return method.length(method.to_rdm(x), method.to_rdm(y))


__all__ = ["compare", "RepresentationMetricSpace", "Stress", "GeneralizedShapeMetric", "AffineInvariantRiemannian"]
