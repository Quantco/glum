import numpy as np
import pytest

from quantcore.glm.matrix import DenseGLMDataMatrix


@pytest.mark.parametrize("scale_predictors", [False, True])
def test_standardize(scale_predictors):
    arr = DenseGLMDataMatrix(np.eye(2))
    arr.standardize(weights=[1, 1], scale_predictors=scale_predictors)
