import os
import numpy as np


if os.getenv("BIG_M"):
    size_m = 1000
    size_n = 300
    size_n_high_rank = 1500
else:
    size_m = 100
    size_n = 30
    size_n_high_rank = 150

num_repeats = 2
rtol = 1e-3
atol = 3e-3
# Adjust tolerance for angles; we exect dot products to be as close as possible (e.g. x'*x = 0.99999) but small dot-
# product errors are amplified in angle calculations (e.g. acos(0.99999) = 4.5e-3)
spherical_atol = np.arccos(1 - atol)