import tensorly as tl
from repsim.util import CorrType
from repsim.compare_impl import (
    BaseRepSim,
    Stress,
    GeneralizedShapeMetric,
    AffineInvariantRiemannian,
    Corr,
)
from typing import Union


def compare(
    x: tl.tensor,
    y: tl.tensor,
    method: Union[BaseRepSim, str] = "stress",
    **kwargs,
) -> tl.tensor:
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
