import torch
from repsim.util import CorrType
from repsim.compare import Stress, GeneralizedShapeMetric, AffineInvariantRiemannian, Corr


def compare(x: torch.Tensor, y: torch.Tensor, method: str = 'stress', **kwargs) -> torch.Tensor:
    meths = {
        'stress': Stress(),
        'generalized_shape_metric': GeneralizedShapeMetric(),
        'riemannian': AffineInvariantRiemannian(),
        'spearman': Corr(corr_type=CorrType.SPEARMAN),
        'pearson': Corr(corr_type=CorrType.PEARSON),
    }

    if method.lower() not in meths:
        raise ValueError(f'Unrecognized Representational Similarity Method "{method}". Options are: {meths.keys()}')

    return meths[method.lower()].compare(x, y, **kwargs)
