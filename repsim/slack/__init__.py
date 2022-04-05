from typing import List, Literal, Union
from scipy.stats import normaltest
import torch
import numpy as np

_SLACK_METHODS = Literal["angle", "clumping_normal", "min_geodesic", "min_euclidean"]


def slack_angle(
    embeddings: Union[torch.Tensor, List[torch.Tensor]], destination: torch.Tensor
) -> torch.Tensor:
    """
    Compute the slack of each node in a representation path.

    Uses the angle offset between each predecessor and successor nodes.

    Arguments:
        embeddings (torch.Tensor or list): embeddings to calculate slack for
        destination (torch.Tensor): destination embedding
    """


def slack_clumping_normality(
    embeddings,
    destination: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the slack of each node in a representation path.

    First computes the distances between each node, and then computes the
    Shapiro-Wilk test for normality.

    Arguments:
        embeddings (torch.Tensor or list): embeddings to calculate slack for
        destination (torch.Tensor): destination embedding
    """
    embeddings = embeddings if destination is None else [*embeddings, destination]
    distances = [np.linalg.norm(i - j) for i, j in zip(embeddings, embeddings[1:])]
    return normaltest(distances)


def slack_min_euclidean(embeddings, destination: torch.Tensor = None):
    """
    Compute the slack of each node in a representation path.

    Computes the total path distance and returns a ratio (greater than 1) of
    the total path distance to the minimum distance between the start and
    destination nodes.

    Arguments:
        embeddings (torch.Tensor or list): embeddings to calculate slack for
        destination (torch.Tensor): destination embedding
    """
    distances = [np.linalg.norm(i - j) for i, j in zip(embeddings, embeddings[1:])]
    dest = destination or embeddings[-1]
    return np.sum(distances) / np.linalg.norm(dest - embeddings[0])


def slack_min_geodesic(
    embeddings: Union[torch.Tensor, List[torch.Tensor]], destination: torch.Tensor
) -> torch.Tensor:
    """
    Compute the slack of each node in a representation path.

    TODO: Combine with slack_min_euclidean?

    Computes the total path distance and returns a ratio (greater than 1) of
    the total path distance to the minimum distance between the start and
    destination nodes.

    Arguments:
        embeddings (torch.Tensor or list): embeddings to calculate slack for
        destination (torch.Tensor): destination embedding
    """
    raise NotImplementedError()


def slack(
    path: Union[torch.Tensor, List[torch.Tensor]],
    destination: torch.Tensor,
    method: _SLACK_METHODS = "angle",
):
    """
    Calculate slack for a path.

    Arguments:
        path (torch.Tensor or list): path to calculate slack for
        method (str): method to use for slack calculation
    """
    methods = {
        "angle": slack_angle,
        "clumping_normal": slack_clumping_normality,
        "min_geodesic": slack_min_geodesic,
        "min_euclidean": slack_min_euclidean,
    }
    if method not in methods:
        raise ValueError(
            f"Invalid slack method: {method}. Must be one of {list(methods.keys())}"
        )
    return methods[method](path, destination)
