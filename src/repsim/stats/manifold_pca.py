import torch
from repsim.geometry import RiemannianSpace
from repsim.geometry.stats import optimize_frechet_mean
from sklearn.base import BaseEstimator
import warnings
from typing import Any


class ManifoldPCA(BaseEstimator):
    """Principal Component Analysis on a Riemannian manifold, in the style of
    sklearn.decomposition.PCA.

    Note: backend is torch rather than numpy.
    """

    def __init__(self, space: RiemannianSpace, *, n_components=2):
        super(ManifoldPCA, self).__init__()
        self.space = space
        self.n_components = n_components

    def _validate_data(
        self,
        X: Any = "no_validation",
        y: Any = "no_validation",
        reset: Any = True,
        validate_separately: Any = False,
        **check_params: Any
    ) -> Any:
        if X == "no_validation":
            return X
        # Override BaseEstimator._validate_data; just ensure that all points in X are on the
        # manifold
        return torch.stack([self.space.project(x) for x in X], dim=0)

    def fit(self, X, y=None, init=None):
        self.fit_transform(X, y, init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """Fit and transform each point into coordinate space of top n_components PCs.

        To convert back from PC-coordinates to points on the manifold, use
        ManifoldPCA.inverse_transform.

        :param X: Iterable of points in the space
        :param y: unused
        :param init: unused
        :return: n by n_components matrix of coordinates
        """
        points = self._validate_data(X)

        if y is not None or init is not None:
            warnings.warn("ManifoldPCA does not use 'y' nor 'init' arguments")

        # Estimate the Frechet mean, using the iterative mean method for initialization
        self.frechet_mean_ = optimize_frechet_mean(
            self.space, points, init_method="iterative"
        )

        # Get the coordinates of every point in the tangent space of the mean
        tangent_vectors = torch.stack(
            [self.space.log_map(self.frechet_mean_, pt) for pt in points], dim=0
        )

        # Now we do PCA in the linear-looking tangent space
        _, s, vT = torch.linalg.svd(
            tangent_vectors.view(len(tangent_vectors), -1), full_matrices=False
        )
        self.singular_values_ = s
        self.components_ = vT[: self.n_components, :].T.reshape(
            self.space.shape + (self.n_components,)
        )
        self.scales_ = torch.sqrt(s[: self.n_components])

        return self._transform(tangent_vectors)

    def transform(self, X, y=None):
        if y is not None:
            warnings.warn("ManifoldPCA does not use 'y' argument")

        points = self._validate_data(X)
        tangent_vectors = torch.stack(
            [self.space.log_map(self.frechet_mean_, pt) for pt in points], dim=0
        )
        return self._transform(tangent_vectors)

    def inverse_transform(self, coordinates):
        """Coordinates back to points on the manifold."""
        tangent_vectors = torch.einsum("nj,...j->n...", coordinates, self.components_)
        return torch.stack(
            [self.space.exp_map(self.frechet_mean_, vec) for vec in tangent_vectors],
            dim=0,
        )

    def _transform(self, tangent_vectors):
        """Project tangent vectors into top PCs."""
        return torch.einsum("n...,...j->nj", tangent_vectors, self.components_)
