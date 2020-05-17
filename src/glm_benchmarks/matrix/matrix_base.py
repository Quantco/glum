from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MatrixBase(ABC):
    """
    Base class for all matrix classes. Cannot be instantiated.
    """

    skip_sklearn_check = True

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # apparently allowed! satisfies mypy.
        self.shape: Tuple[int, int]
        self.dtype: np.dtype

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

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11
