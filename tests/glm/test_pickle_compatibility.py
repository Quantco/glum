"""Tests for pickle compatibility with models from older glum versions."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV


@pytest.fixture
def test_data():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": pd.Categorical(["x", "y", "x", "y", "x"]),
            "y": [1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )
    return df[["num", "cat"]], df["y"]


def test_missing_max_inner_iter(test_data, tmp_path):
    """Test loading models pickled before max_inner_iter was added (< v3.1.0)."""
    # Create a model and fit it
    X, y = test_data
    model = GeneralizedLinearRegressor(alpha=0.1)
    model.fit(X, y)

    # Simulate an old model by removing max_inner_iter
    if hasattr(model, "max_inner_iter"):
        del model.max_inner_iter

    # Pickle to file
    pickle_file = tmp_path / "old_model.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(model, f)

    # Load from file
    with open(pickle_file, "rb") as f:
        old_model = pickle.load(f)

    # Verify max_inner_iter was set to default
    assert hasattr(old_model, "max_inner_iter")
    assert old_model.max_inner_iter == 100000

    # Verify predictions still work
    predictions = old_model.predict(X)
    assert predictions.shape == y.shape


def test_feature_dtypes_to_categorical_levels_migration(test_data, tmp_path):
    """Test migration from feature_dtypes_ to _categorical_levels_."""
    # Create a model with categorical data
    X, y = test_data
    model = GeneralizedLinearRegressor(alpha=0.1)
    model.fit(X, y)

    # Simulate old model: old-style feature_dtypes_
    if hasattr(model, "_categorical_levels_"):
        categorical_levels = model._categorical_levels_
        delattr(model, "_categorical_levels_")

        model.__dict__["feature_dtypes_"] = {}
        for col in X.columns:
            if col in categorical_levels:
                model.__dict__["feature_dtypes_"][col] = pd.CategoricalDtype(
                    categories=categorical_levels[col]
                )
            else:
                model.__dict__["feature_dtypes_"][col] = X[col].dtype

    if hasattr(model, "_feature_dtypes_"):
        delattr(model, "_feature_dtypes_")

    # Pickle to file
    pickle_file = tmp_path / "old_model.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(model, f)

    # Load from file
    with open(pickle_file, "rb") as f:
        old_model = pickle.load(f)

    # Verify that _categorical_levels_ was created
    assert hasattr(old_model, "_categorical_levels_")

    # Check that categorical columns were migrated correctly
    for col, levels in categorical_levels.items():
        assert col in old_model._categorical_levels_
        assert old_model._categorical_levels_[col] == levels

    # Verify that _feature_dtypes_ was created for backward compat
    assert hasattr(old_model, "_feature_dtypes_")

    # Verify that categorical_levels_ property works
    levels = old_model.categorical_levels_
    for col in categorical_levels:
        assert col in levels
        assert levels[col] == categorical_levels[col]

    # Test prediction works
    predictions = old_model.predict(X)
    assert predictions.shape == y.shape


@pytest.mark.parametrize(
    "model_class,file_name,params",
    [
        (GeneralizedLinearRegressor, "glum_v3_0_model.pkl", {"alpha": 0.1}),
        (GeneralizedLinearRegressorCV, "glum_v3_0_cv_model.pkl", {}),
    ],
)
def test_glum_v3_0_pickle_compatibility(test_data, model_class, file_name, params):
    """Test loading a model pickled with glum v3.0."""
    X, y = test_data
    pickle_path = Path(__file__).parent / "pickles" / file_name

    # Load the pickled model from v3.0
    with open(pickle_path, "rb") as f:
        old_model = pickle.load(f)

    new_model = model_class(**params)
    new_model.fit(X, y)

    # Verify predictions match
    old_predictions = old_model.predict(X)
    new_predictions = new_model.predict(X)
    np.testing.assert_array_almost_equal(old_predictions, new_predictions)

    # Verify all expected attributes are present
    assert getattr(old_model, "max_inner_iter") == 100000

    # Verify categorical_levels_ and feature_dtypes_ properties match between models
    assert old_model.categorical_levels_ == new_model.categorical_levels_

    # Check that feature_dtypes_ property works and contains categorical info
    # Have to do it this way as there might be a str/object mismatch in categories_dtype
    if old_model.categorical_levels_:
        old_feature_dtypes = old_model.feature_dtypes_
        new_feature_dtypes = new_model.feature_dtypes_
        for col in old_model.categorical_levels_:
            assert col in old_feature_dtypes
            assert col in new_feature_dtypes
            assert isinstance(old_feature_dtypes[col], pd.CategoricalDtype)
            assert isinstance(new_feature_dtypes[col], pd.CategoricalDtype)
            assert (
                old_feature_dtypes[col].categories.tolist()
                == new_feature_dtypes[col].categories.tolist()
            )
