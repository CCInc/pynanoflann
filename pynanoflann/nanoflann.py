"Sklearn interface to the native nanoflann module"
import copyreg
import warnings
from typing import Optional

import nanoflann_ext
import numpy as np
from sklearn.neighbors.base import (KNeighborsMixin, NeighborsBase,
                                    RadiusNeighborsMixin, UnsupervisedMixin)
from sklearn.utils.validation import check_is_fitted

SUPPORTED_TYPES = [np.float32, np.float64]


def pickler(c):
    X = c._fit_X if hasattr(c, '_fit_X') else None
    return unpickler, (c.n_neighbors, c.radius, c.leaf_size, c.metric, X)


def unpickler(n_neighbors, radius, leaf_size, metric, X):
    # Recreate an kd-tree instance
    tree = KDTree(n_neighbors, radius, leaf_size, metric)
    # Unpickling of the fitted instance
    if X is not None:
        tree.fit(X)
    return tree


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError('Supported types: [{}]'.format(SUPPORTED_TYPES))
    if len(points.shape) != 2:
        raise ValueError(f'Incorrect shape {len(points.shape)} != 2')


class KDTree(NeighborsBase, KNeighborsMixin,
             RadiusNeighborsMixin, UnsupervisedMixin):

    def __init__(self, X: np.ndarray, n_neighbors=5, leaf_size=10):

        super().__init__(
            n_neighbors=n_neighbors, leaf_size=leaf_size)

        self.fit(X)

    def fit(self, X: np.ndarray, index_path: Optional[str] = None):
        """
        Args:
            X: np.ndarray data to use
            index_path: str Path to a previously built index. Allows you to not rebuild index.
                NOTE: Must use the same data on which the index was built.
        """
        if X.shape[1] != 3:
            raise ValueError('Dimension not supported')

        _check_arg(X)
        if X.dtype == np.float32:
            self.index = nanoflann_ext.KDTree32(self.n_neighbors, self.leaf_size)
        else:
            self.index = nanoflann_ext.KDTree64(self.n_neighbors, self.leaf_size)

        self.index.fit(X, index_path if index_path is not None else "")
        self._fit_X = X

    def query(self, X, k=None):
        n_neighbors = k
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if self._fit_X.shape[0] < n_neighbors:
            raise ValueError(f"Expected n_neighbors <= n_samples,\
                 but n_samples = {self._fit_X.shape[0]}, n_neighbors = {n_neighbors}")

        dists, idxs = self.index.kneighbors(X, n_neighbors)
        dists = np.sqrt(dists)

        return dists, idxs

    def query_radius(self, X, r=None, return_distance=False):
        radius = r
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        radius = radius ** 2 #nanoflann uses square distances

        dists, idxs = self.index.radius_neighbors(X, radius)
        idxs = np.array([np.array(x) for x in idxs])

        if return_distance:
            dists = np.array([np.sqrt(np.array(x)).squeeze() for x in dists])
            return dists, idxs
        else:
            return idxs

    def get_data(self, copy: bool = True) -> np.ndarray:
        """Returns underlying data points. If copy is `False` then no modifications should be applied to the returned data.

        Args:
            copy: whether to make a copy.
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)

        if copy:
            return self._fit_X.copy()
        else:
            return self._fit_X

    def save_index(self, path: str) -> int:
        "Save index to the binary file. NOTE: Data points are NOT stored."
        return self.index.save_index(path)


# Register pickling of non-trivial types
copyreg.pickle(KDTree, pickler, unpickler)
