"""tests/kernels.py.

The role of this file is to define a list of kernels to parametrize kernel tests
"""

import pytest
import numpy as np
from repsim.kernels import SquaredExponential, Laplace, Linear

_list_of_kernels = [
    {
        "kernel": SquaredExponential(length_scale="auto"),
        "name": "SqExp[auto]",
        "rank": lambda p: np.inf,
    },
    {
        "kernel": SquaredExponential(length_scale=5.0),
        "name": "SqExp[5.000]",
        "rank": lambda p: np.inf,
    },
    {
        "kernel": SquaredExponential(length_scale="median/2.0"),
        "name": "SqExp[median/2.0]",
        "rank": lambda p: np.inf,
    },
    {
        "kernel": Laplace(length_scale="auto"),
        "name": "Laplace[auto]",
        "rank": lambda p: np.inf,
    },
    {
        "kernel": Laplace(length_scale=5.0),
        "name": "Laplace[5.000]",
        "rank": lambda p: np.inf,
    },
    {
        "kernel": Laplace(length_scale="median/2.000"),
        "name": "Laplace[median/2.000]",
        "rank": lambda p: np.inf,
    },
    {"kernel": Linear(), "name": "Linear", "rank": lambda p: p},
]


@pytest.fixture(params=_list_of_kernels, ids=lambda p: p["kernel"].string_id())
def kernel(request):
    m = request.param["kernel"]
    # Add other properties listed in _list_of_kernels as instance variables on the metric object,
    # prefixed with "test_" to avoid naming conflicts. In other words, this loop adds properties
    # like m.test_rotation_invariant and m.test_high_rank_data based on key, value pairs in the
    # list at the top of this file.
    d = m.__dict__
    for k, v in request.param.items():
        if k == "kernel":
            continue
        k = "test_" + k.replace("-", "_")
        d[k] = v
    return m
