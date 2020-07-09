from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class MatrixBase(ABC):
    """
    Base class for all matrix classes. Cannot be instantiated.
    """

    ndim = 2
    shape: Tuple[int, int]
    dtype: np.dtype

    @abstractmethod
    def dot(self, other, cols: np.ndarray = None):
        """
        Perform: self[:, cols] @ other

        The cols parameter allows restricting to a subset of the
        matrix without making a copy.
        """
        pass

    @abstractmethod
    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        """
        Perform: self[rows, cols].T @ vec

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.
        """
        pass

    @abstractmethod
    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        """
        Perform a sandwich product: (self[rows, cols].T * d[rows]) @ self[rows, cols]

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.
        """
        pass

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    @abstractmethod
    def getcol(self, i: int):
        pass

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    @abstractmethod
    def toarray(self) -> np.ndarray:
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

    def get_col_means(self, weights: np.ndarray) -> np.ndarray:
        return self.transpose_dot(weights)

    @abstractmethod
    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        pass

    def standardize(
        self, weights: np.ndarray, center_predictors: bool, scale_predictors: bool
    ) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
        """
        Returns a StandardizedMat, col_means, and col_stds

        If center_predictors is False, col_means will be zeros

        If scale_predictors is False, col_stds will be None
        """
        from .standardized_mat import StandardizedMat

        col_means = self.get_col_means(weights)
        if scale_predictors:
            col_stds = self.get_col_stds(weights, col_means)
            mult = one_over_var_inf_to_val(col_stds, 1.0)
            if center_predictors:
                shifter = -col_means * mult
                out_means = col_means
            else:
                shifter = np.zeros_like(col_means)
                out_means = shifter
        else:
            col_stds = None
            if center_predictors:
                shifter = -col_means
                out_means = col_means
            else:
                shifter = np.zeros_like(col_means)
                out_means = shifter
            mult = None

        return StandardizedMat(self, shifter, mult), out_means, col_stds

    @abstractmethod
    def __getitem__(self, item):
        pass

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11


def one_over_var_inf_to_val(arr: np.ndarray, val: float) -> np.ndarray:
    zeros = np.where(arr == 0)
    with np.errstate(divide="ignore"):
        one_over = 1 / arr
    one_over[zeros] = val
    return one_over
