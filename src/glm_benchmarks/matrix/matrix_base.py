from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from .standardize import one_over_var_inf_to_zero


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
    def transpose_dot(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        pass

    def __rmatmul__(self, other: Union[np.ndarray, List]) -> np.ndarray:
        """
        other @ X = (X.T @ other.T).T = X.transpose_dot(other.T).T

        Parameters
        ----------
        other: array-like

        Returns
        -------
        array

        """
        if not hasattr(other, "T"):
            other = np.asarray(other)
        return self.transpose_dot(other.T).T  # type: ignore

    @abstractmethod
    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        pass

    def _get_col_means(self, weights: np.ndarray) -> np.ndarray:
        return self.transpose_dot(weights)

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
            col_stds = np.ones(self.shape[1], self.dtype)

        one_over_col_sds = one_over_var_inf_to_zero(col_stds)

        self.scale_cols_inplace(one_over_col_sds)
        return ColScaledMat(self, -col_means * one_over_col_sds), col_means, col_stds

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        pass

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11
