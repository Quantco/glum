from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Tuple, Union

import numpy as np


class MatrixBase(ABC):
    """
    Base class for all matrix classes. Cannot be instantiated.
    """

    skip_sklearn_check = True

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.shape: Tuple = ()

    @abstractmethod
    def toarray(self) -> np.ndarray:
        pass

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    @abstractmethod
    def multiply(self, other):
        """
        Element-wise multiplication, as in np.multiply or
        scipy.sparse.csr_matrix.multiply
        """
        pass

    def __mul__(self, other):
        """ Defines the behavior of "*", element-wise multiplication. """
        return self.multiply(other)

    @abstractmethod
    def transpose(self):
        pass

    @property
    def T(self):
        return self.transpose()

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
    def sum(self, axis: Optional[int]) -> Union[np.ndarray, float]:
        pass

    def mean(self, axis: Optional[int]) -> Union[np.ndarray, float]:
        if axis is None:
            denominator = reduce(lambda x, y: x * y, self.shape)
        else:
            denominator = self.shape[axis]
        return self.sum(axis) / denominator

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of this class
    __array_priority__ = 11
