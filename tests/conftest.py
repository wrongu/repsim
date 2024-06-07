import pytest
import torch
import numpy as np
from tests.constants import size_m, size_n, size_n_high_rank, device, dtype
from tests.metrics import metric
from tests.kernels import kernel


@pytest.fixture(autouse=True)
def set_global_test_seed():
    # Seed chosen by keyboard mashing. Per the torch docs, this sets seeds on all devices. Per
    # the pytest docs, this being an 'autouse' fixture means it will be run before every test.
    torch.manual_seed(348628013)
    np.random.seed(5387290)


@pytest.fixture
def data_x():
    return torch.randn(size_m, size_n, device=device, dtype=dtype)


@pytest.fixture
def data_y(data_x):
    return (
        data_x
        + torch.randn(size_m, size_n, device=device, dtype=dtype) / np.sqrt(size_n) * 2
    )


@pytest.fixture
def data_z(data_x):
    return (
        data_x
        + torch.randn(size_m, size_n, device=device, dtype=dtype) / np.sqrt(size_n) * 2
    )


@pytest.fixture
def high_rank_x():
    return torch.randn(size_m, size_n_high_rank, device=device, dtype=dtype)


@pytest.fixture
def high_rank_y(high_rank_x):
    return (
        high_rank_x
        + torch.randn(size_m, size_n_high_rank, device=device, dtype=dtype)
        / np.sqrt(size_n_high_rank)
        * 2
    )


@pytest.fixture
def high_rank_z(high_rank_x):
    return (
        high_rank_x
        + torch.randn(size_m, size_n_high_rank, device=device, dtype=dtype)
        / np.sqrt(size_n_high_rank)
        * 2
    )


@pytest.fixture
def data_labels():
    # Create dummy one-hot labels for 4 classes
    labels = torch.zeros(size_m, 4, device=device, dtype=dtype)
    labels[torch.arange(size_m), torch.randint(0, 4, (size_m,))] = 1
    return labels


__all__ = [
    "data_x",
    "data_y",
    "data_z",
    "high_rank_x",
    "high_rank_y",
    "data_labels",
    "metric",
]
