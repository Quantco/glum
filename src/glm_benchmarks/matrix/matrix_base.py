from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MatrixBase(ABC):
    """
    Base class for all matrix classes. Cannot be instantiated.
    """

    ndim = 2
    shape: Tuple[int, int]
    dtype: np.dtype

    @abstractmethod
    def dot(self, other):
        """ Matrix multiplication. """
        pass

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    def __rmatmul__(self, other):
        """
        other @ self = (self.T @ other.T).T
        """
        return (self.T @ other.T).T

    @abstractmethod
    def getcol(self, i: int):
        pass

    @abstractmethod
    def sandwich(self, d: np.ndarray) -> np.ndarray:
        pass

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    @abstractmethod
    def toarray(self) -> np.ndarray:
        pass

    @abstractmethod
    def transpose(self):
        pass

    @property
    def T(self):
        return self.transpose()

    @abstractmethod
    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        pass

    @abstractmethod
    def _get_col_means(self, weights: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        pass

    def standardize(self, weights: np.ndarray, scale_predictors: bool):
        """
        Returns a ColScaledMat, col_means, and col_stds
        """
        from .scaled_mat import ColScaledMat

        col_means = self._get_col_means(weights)
        if scale_predictors:
            col_stds = self._get_col_stds(weights, col_means)
        else:
            col_stds = np.ones(self.shape[1])
        return ColScaledMat(self, -col_means), col_means, col_stds

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11
