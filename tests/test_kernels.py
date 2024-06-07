import torch
from tests.constants import size_m, atol, rtol
from repsim.kernels import center, is_centered


def test_kernel_name(kernel):
    assert kernel.test_name == kernel.string_id()


def test_kernel_shape(kernel, data_x, data_y):
    k_xx = kernel(data_x)
    assert k_xx.shape == (size_m, size_m)
    k_xx_2 = kernel(data_x, data_x)
    assert k_xx_2.shape == (size_m, size_m)
    k_xy = kernel(data_x, data_y)
    assert k_xy.shape == (size_m, size_m)


def test_centering(kernel, data_x):
    k_xx = kernel(data_x)
    assert is_centered(center(k_xx), atol=atol, rtol=rtol)


def test_centering_is_idempotent_eye(data_x):
    k_xx = torch.eye(size_m, dtype=data_x.dtype, device=data_x.device)
    assert torch.allclose(center(k_xx), center(center(k_xx)), atol=atol, rtol=rtol)


def test_centering_is_idempotent_data(kernel, data_x):
    k_xx = kernel(data_x)
    assert torch.allclose(center(k_xx), center(center(k_xx)), atol=atol, rtol=rtol)


def test_kernel_rank(kernel, data_x):
    for p in [5, 10, 20, 30]:
        # NOTE: single-precision runs into some numerical precision issues, but we can assert
        # that the logic makes sense using doubles.
        k_xx = kernel(data_x[:, :p].double())
        assert torch.linalg.matrix_rank(k_xx, hermitian=True) == min(
            size_m, kernel.test_rank(p)
        )
