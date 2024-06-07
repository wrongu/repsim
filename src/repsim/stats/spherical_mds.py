import torch
import numpy as np
from repsim.geometry.hypersphere import HyperSphere
from repsim.geometry.stats import IterativeFrechetMean
from sklearn.base import BaseEstimator
from sklearn.manifold import MDS
from sklearn.utils import check_random_state
from joblib import Parallel, delayed, effective_n_jobs
import warnings


def _spherical_mds_single(
    distances,
    *,
    dim=2,
    init=None,
    max_iter=300,
    max_inner_loop=None,
    eps=1e-3,
    random_state=None
):
    """Run a single instance of Spherical MDS, solving positions of n points on the dim-dimensional
    sphere (embedded in dim+1-dimensional space), using the algorithm from [1].

    [1] Agarwal, A., Phillips, J. M., & Venkatasubramanian, S. (2010). Universal multi-dimensional
    scaling. Proceedings     of the ACM SIGKDD International Conference on Knowledge Discovery and
    Data Mining, 1149–1158.
    https://doi.org/10.1145/1835804.1835948
    """
    random_state = check_random_state(random_state)
    n_samples = distances.shape[0]

    def _project(z_):
        return z_ / torch.sqrt(torch.sum(z_ * z_, dim=-1, keepdim=True))

    # Initialize with Euclidean MDS, then project onto sphere
    if init is None:
        euclidean_mds = MDS(
            n_components=dim + 1,
            n_init=1,
            dissimilarity="precomputed",
            random_state=random_state,
        )
        z = torch.tensor(euclidean_mds.fit_transform(distances))
        z = _project(z - torch.mean(z, dim=0))
    else:
        assert init.shape == (n_samples, dim + 1)
        z = _project(init - torch.mean(init, dim=0))

    def _angles(z_):
        return torch.arccos(torch.clip(torch.einsum("ia,ja->ij", z_, z_), -1.0, +1.0))

    def _squared_stress(z_):
        return torch.sum((torch.triu(distances) - torch.triu(_angles(z_))) ** 2)

    # Iteratively adjust one point at a time using Agarwal et al' algorithm
    sphere = HyperSphere(dim=dim)
    sq_stress = _squared_stress(z)
    # Each z[i] gets its own 'updater', which is an object that will help nudge z[i] closer to
    # other z[j]s
    updaters = [IterativeFrechetMean(sphere) for _ in z]
    for itr in range(max_iter):
        # Reset each 'updater'; when resetting to n=0, it takes big steps, and when resetting to
        # n=big, it takes smaller steps. We want to take smaller and smaller steps as 'itr' gets
        # bigger, so reset with n=itr.
        for u in updaters:
            u.reset(n=itr)
        # For each z[i], loop over z[j]s and nudge the value of z[i] towards those z[j]s. Use
        # randperm so we get a different order of z[i]s in each loop
        for i in torch.randperm(n_samples):
            # For each z_j, draw an arc from z_j to z_i and find the point along that line that
            # is the 'correct' distance away. Call it hat_z_j
            idx_j = (
                torch.randperm(n_samples)
                if max_inner_loop is None
                else torch.randperm(n_samples)[:max_inner_loop]
            )
            for j in idx_j:
                if i == j:
                    continue
                # Ratio of distance-we-want to distance-we-have
                ratio = distances[i, j] / sphere.length(z[i], z[j])
                # Find the point along the j-->i geodesic that is at the correct distance
                hat_z_j = sphere.exp_map(z[j], sphere.log_map(z[j], z[i]) * ratio)
                # Average together all hat_z_j's
                updaters[i].update(hat_z_j)
            z[i] = updaters[i].mean
        new_sq_stress = _squared_stress(z)

        if sq_stress - new_sq_stress < eps:
            # Converged!
            return z, torch.sqrt(new_sq_stress), itr
        else:
            sq_stress = new_sq_stress
    return z, torch.sqrt(sq_stress), max_iter - 1


def spherical_mds(
    distances,
    *,
    dim=2,
    init=None,
    n_init=4,
    n_jobs=None,
    max_iter=300,
    max_inner_loop=None,
    eps=1e-3,
    random_state=None,
    verbose=0,
    return_n_iter=False
):
    random_state = check_random_state(random_state)

    best_z, best_stress, best_iter = None, None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _spherical_mds_single(
                distances,
                dim=dim,
                init=init,
                max_iter=max_iter,
                max_inner_loop=max_inner_loop,
                eps=eps,
                random_state=random_state,
            )
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_z = pos.clone()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_spherical_mds_single)(
                distances,
                dim=dim,
                init=init,
                max_iter=max_iter,
                max_inner_loop=max_inner_loop,
                eps=eps,
                random_state=seed,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_z = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_z, best_stress, best_iter
    else:
        return best_z, best_stress


class SphericalMDS(BaseEstimator):
    """MultiDimensional Scaling on a sphere, in the style of sklearn.manifold.MDS.

    Note: backend is torch rather than numpy.
    """

    def __init__(
        self,
        dim=2,
        *,
        n_init=4,
        dissimilarity="arc length",
        center=False,
        max_iter=300,
        max_inner_loop=None,
        verbose=0,
        eps=1e-3,
        n_jobs=None,
        random_state=None
    ):
        self.dim = dim
        self.n_init = n_init
        self.dissimilarity = dissimilarity.lower()
        self.center = center
        self.max_iter = max_iter
        self.max_inner_loop = max_inner_loop
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None, init=None):
        self.fit_transform(X, y, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        if y is not None:
            warnings.warn("SphericalMDS does not use 'y' argument")

        if self.dissimilarity == "precomputed":
            if not _is_arc_length_matrix(X):
                raise ValueError(
                    "With dissimilarity='precomputed', X must be a valid matrix "
                    "of pairwise arc-distances."
                )
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "arc length":
            self.dissimilarity_matrix_ = pairwise_arc_lengths(X, self.center)
        else:
            raise ValueError(
                "'dissimilarity' argument must be one of 'arc length' or 'precomputed'"
            )

        self.embedding_, self.stress_, self.n_iter_ = spherical_mds(
            self.dissimilarity_matrix_,
            dim=self.dim,
            init=init,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            max_inner_loop=self.max_inner_loop,
            eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
        )
        return self.embedding_


def pairwise_arc_lengths(X, center):
    # TODO - implement batch-wise and pair-wise operations natively in HyperSphere (and other
    #  spaces)
    if center:
        X = X - torch.mean(X, 0)
    dot_ij = X @ X.T
    len_ii = torch.sqrt(torch.diag(dot_ij))
    cosine = dot_ij / len_ii[:, None] / len_ii[None, :]
    return torch.arccos(torch.clip(cosine, -1.0, +1.0))


def _is_arc_length_matrix(X):
    tolerance, soft_tolerance = 1e-4, 1e-3
    if X.ndim != 2:
        return False
    if X.shape[0] != X.shape[1]:
        return False
    if not torch.all(X >= 0.0) or not torch.all(X <= np.pi):
        return False
    # The diagonal often comes from arccos(dot(a,b)), and we generally care more about precision
    # in the dot() part. Note that arccos(0.999) = .045, which is still 'far from zero'. So
    # instead of asserting that the diagonal is zero, we'll assert that it's as close to zero as
    # can be expected based on 'tolerance' error in the dot() part.
    if not torch.all(X.diag().abs() < np.arccos(1.0 - tolerance)):
        if torch.all(X.diag().abs() < np.arccos(1.0 - soft_tolerance)):
            warnings.warn(
                "Diagonal of arc-length matrix is not zero, but is close to zero."
            )
        else:
            return False
    return True
