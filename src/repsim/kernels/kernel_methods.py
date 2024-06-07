import torch


def center(k: torch.Tensor) -> torch.Tensor:
    """Center features of a kernel by pre- and post-multiplying by the centering matrix H.

    In other words, if k_ij is dot(x_i, x_j), the result will be dot(x_i - mu_x, x_j - mu_x).

    :param k: a n by n Gram matrix of inner products between xs
    :return: a n by n centered matrix
    """
    n = k.size()[0]
    if k.size() != (n, n):
        raise ValueError(
            f"Expected k to be nxn square matrix, but it has size {k.size()}"
        )
    H = (
        torch.eye(n, device=k.device, dtype=k.dtype)
        - torch.ones((n, n), device=k.device, dtype=k.dtype) / n
    )
    return H @ k @ H


def is_centered(k: torch.Tensor, **kwargs) -> bool:
    # Centering is essentially subtracting the mean. We can test for centeredness by testing if
    # the row- and col-means are both zero
    row_mean, col_mean = torch.mean(k, dim=0), torch.mean(k, dim=1)
    return torch.allclose(
        row_mean, torch.zeros_like(row_mean), **kwargs
    ) and torch.allclose(col_mean, torch.zeros_like(col_mean), **kwargs)


def hsic(
    k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True
) -> torch.Tensor:
    """Compute Hilbert-Schmidt Independence Criteron (HSIC)

    :param k_x: n by n values of kernel applied to all pairs of x data
    :param k_y: n by n values of kernel on y data
    :param centered: whether or not at least one kernel is already centered
    :param unbiased: if True, use unbiased HSIC estimator of Song et al (2007), else use original
        estimator of Gretton et al (2005)
    :return: scalar score in [0*, inf) measuring dependence of x and y

    * note that if unbiased=True, it is possible to get small values below 0.
    """
    if k_x.size() != k_y.size():
        raise ValueError(
            "RDMs must have the same size, but got {} and {}".format(
                k_x.size(), k_y.size()
            )
        )
    n = k_x.size()[0]

    if not centered:
        k_y = center(k_y)

    if unbiased:
        # Remove the diagonal
        k_x = k_x * (1 - torch.eye(n, device=k_x.device, dtype=k_x.dtype))
        k_y = k_y * (1 - torch.eye(n, device=k_y.device, dtype=k_y.dtype))
        # Equation (4) from Song et al (2007)
        return (
            (k_x * k_y).sum()
            - 2 * (k_x.sum(dim=0) * k_y.sum(dim=0)).sum() / (n - 2)
            + k_x.sum() * k_y.sum() / ((n - 1) * (n - 2))
        ) / (n * (n - 3))
    else:
        # The original estimator from Gretton et al (2005)
        return torch.sum(k_x * k_y) / (n - 1) ** 2


def cka(
    k_x: torch.Tensor, k_y: torch.Tensor, centered: bool = False, unbiased: bool = True
) -> torch.Tensor:
    """Compute Centered Kernel Alignment (CKA).

    :param k_x: n by n values of kernel applied to all pairs of x data
    :param k_y: n by n values of kernel on y data
    :param centered: whether or not at least one kernel is already centered
    :param unbiased: if True, use unbiased HSIC estimator of Song et al (2007), else use original
        estimator of Gretton et al (2005)
    :return: scalar score in [0*, 1] measuring normalized dependence between x and y.

    * note that if unbiased=True, it is possible to get small values below 0.
    """
    hsic_xy = hsic(k_x, k_y, centered, unbiased)
    hsic_xx = hsic(k_x, k_x, centered, unbiased)
    hsic_yy = hsic(k_y, k_y, centered, unbiased)
    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
