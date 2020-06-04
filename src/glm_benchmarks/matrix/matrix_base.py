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
    def sandwich(self, d: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def limited_rmatvec(
        self, v: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def limited_matvec(
        self, v: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
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

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11
