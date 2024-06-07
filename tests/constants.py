import os
import numpy as np
import torch.cuda

if os.getenv("BIG_M"):
    size_m = 1000
    size_n = 300
    size_n_high_rank = 1500
else:
    size_m = 100
    size_n = 30
    size_n_high_rank = 150

if os.getenv("USE_CUDA") and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if os.getenv("DTYPE") == "float64":
    dtype = torch.float64
else:
    dtype = torch.float32

num_repeats = 2
rtol = 1e-3
atol = 1e-3


# Adjust tolerance for angles; we exect dot products to be as close as possible (e.g. x'*x =
# 0.99999) but small dot- product errors are amplified in angle calculations (e.g. acos(0.99999)
# = 4.5e-3). The size of the spherical tolerance is adjusted to the location on the arccos curve.
def spherical_atol(dot_value):
    dot_value = np.clip(dot_value, -1.0, 1.0)
    # The derivative of arccos is -1 / sqrt(1 - x^2), so the error tolerance we allow in angles
    # is abs(d angle / d dot) times the error tolerance in the dot product value (which is atol).
    # Since the max of |da/dd| is 1, this means that spherical_atol will range between atol and
    # arccos(1-atol).
    max_atol = np.arccos(1 - atol)
    return min(max_atol, atol / np.sqrt(1 - dot_value**2))


__all__ = [
    "size_m",
    "size_n",
    "size_n_high_rank",
    "device",
    "dtype",
    "num_repeats",
    "rtol",
    "atol",
    "spherical_atol",
]
