import torch
from repsim.util import CorrType
from repsim.compare import (
    BaseRepSim,
    Stress,
    GeneralizedShapeMetric,
    AffineInvariantRiemannian,
    Corr,
)
from typing import Union


def compare(
    x: torch.Tensor,
    y: torch.Tensor,
    method: Union[BaseRepSim, str] = "stress",
    **kwargs,
) -> torch.Tensor:
    method_lookup = {
        "stress": Stress(),
        "generalized_shape_metric": GeneralizedShapeMetric(),
        "riemannian": AffineInvariantRiemannian(),
        "spearman": Corr(corr_type=CorrType.SPEARMAN),
        "pearson": Corr(corr_type=CorrType.PEARSON),
    }

    if isinstance(method, str):
        if method.lower() not in method_lookup:
            raise ValueError(
                f'Unrecognized Representational Similarity Method "{method}". Options are: {method_lookup.keys()}'
            )
        method = method_lookup[method.lower()]

    return method.compare(x, y, **kwargs)
