# These classes and functions are heavily based on code from the dask_ml package:
# https://github.com/dask/dask-ml
#
# License for dask_ml:
# BSD 3-Clause "New" or "Revised" License
#
# Copyright (c) 2017, Anaconda, Inc. and contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# Neither the name of Anaconda nor the names of any contributors may be used to
# endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


from typing import Iterable, Optional, Union

import numpy as np
import packaging.version
import pandas as pd
import sklearn.compose
from pandas.api.types import is_categorical_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.utils.validation import check_is_fitted

_PANDAS_VERSION = packaging.version.parse(pd.__version__)
_HAS_CTD = _PANDAS_VERSION >= packaging.version.parse("0.21.0")


class Categorizer(BaseEstimator, TransformerMixin):
    """Transform columns of a DataFrame to categorical dtype.

    This is a useful pre-processing step for dummy, one-hot, or
    categorical encoding.

    Notes
    -----
    This transformer only applies to
    ``pandas.DataFrame``. All object-type columns are converted to
    categoricals. The set of categories will be the values present in the
    column and the categoricals will be unordered.

    All other columns are included in the transformed output untouched.

    Attributes
    ----------
    columns_ : pandas.Index
        The columns that were categorized. Useful when ``categories`` is None,
        and we detect the categorical and object columns

    categories_ : dict
        A dictionary mapping column names to dtypes. For pandas>=0.21.0, the
        values are instances of ``pandas.api.types.CategoricalDtype``. For
        older pandas, the values are tuples of ``(categories, ordered)``.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ['a', 'a', 'b']})
    >>> ce = Categorizer()
    >>> ce.fit_transform(df).dtypes
    A       int64
    B    category
    dtype: object

    >>> ce.categories_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    Using CategoricalDtypes for specifying the categories:

    >>> from pandas.api.types import CategoricalDtype
    >>> ce = Categorizer(categories={"B": CategoricalDtype(['a', 'b', 'c'])})
    >>> ce.fit_transform(df).B.dtype
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)
    """

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "Categorizer":
        """Find the categorical columns.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        columns = X.columns
        categories = {}
        for name in columns:
            col = X[name]
            if not is_categorical_dtype(col):
                col = pd.Series(col, index=X.index).astype("category")

            if _HAS_CTD:
                categories[name] = col.dtype
            else:
                categories[name] = (col.cat.categories, col.cat.ordered)

        self.columns_ = columns
        self.categories_ = categories
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Transform the columns in ``X`` according to ``self.categories_``.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        X_trn : pandas.DataFrame
            Same type as the input. The columns in ``self.categories_`` will
            be converted to categorical dtype.
        """
        check_is_fitted(self, "categories_")
        categories = self.categories_

        for k, dtype in categories.items():
            if _HAS_CTD:
                if not isinstance(dtype, pd.api.types.CategoricalDtype):
                    dtype = pd.api.types.CategoricalDtype(*dtype)
                X[k] = X[k].astype(dtype)
            else:
                cat, ordered = dtype
                X[k] = X[k].astype("category").cat.set_categories(cat, ordered)

        return X


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Dummy (one-hot) encode categorical columns.

    Parameters
    ----------
    drop_first : bool, default False
        Whether to drop the first category in each column.

    Attributes
    ----------
    columns_ : Index
        The columns in the training data before dummy encoding

    transformed_columns_ : Index
        The columns in the training data after dummy encoding

    categorical_columns_ : Index
        The categorical columns in the training data

    noncategorical_columns_ : Index
        The rest of the columns in the training data

    categorical_blocks_ : dict
        Mapping from column names to slice objects. The slices
        represent the positions in the transformed array that the
        categorical column ends up at

    dtypes_ : dict
        Dictionary mapping column name to either

        * instances of CategoricalDtype (pandas >= 0.21.0)
        * tuples of (categories, ordered)

    Notes
    -----
    This transformer only applies to pandas DataFrames.

    The inverse transformation can be used on a dataframe or array.

    Examples
    --------
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4],
    ...                      "B": pd.Categorical(['a', 'a', 'a', 'b'])})
    >>> de = DummyEncoder()
    >>> trn = de.fit_transform(data)
    >>> trn
    A  B_a  B_b
    0  1    1    0
    1  2    1    0
    2  3    1    0
    3  4    0    1

    >>> de.columns_
    Index(['A', 'B'], dtype='object')

    >>> de.non_categorical_columns_
    Index(['A'], dtype='object')

    >>> de.categorical_columns_
    Index(['B'], dtype='object')

    >>> de.dtypes_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    >>> de.categorical_blocks_
    {'B': slice(1, 3, None)}
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "DummyEncoder":
        """Determine the categorical columns to be dummy encoded.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        self.categorical_columns_ = X.select_dtypes(include=["category"]).columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)

        if _HAS_CTD:
            self.dtypes_ = {col: X[col].dtype for col in self.categorical_columns_}
        else:
            self.dtypes_ = {
                col: (X[col].cat.categories, X[col].cat.ordered)
                for col in self.categorical_columns_
            }

        left = len(self.non_categorical_columns_)
        self.categorical_blocks_ = {}
        for col in self.categorical_columns_:
            right = left + len(X[col].cat.categories)
            if self.drop_first:
                right -= 1
            self.categorical_blocks_[col], left = slice(left, right), right

        sample = X.iloc[:1]
        self.transformed_columns_ = pd.get_dummies(
            sample, drop_first=self.drop_first
        ).columns
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns_)
            )
        return pd.get_dummies(X, drop_first=self.drop_first)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal (integer) encode categorical columns.

    Attributes
    ----------
    columns_ : Index
        The columns in the training data before/after encoding

    categorical_columns_ : Index
        The categorical columns in the training data

    noncategorical_columns_ : Index
        The rest of the columns in the training data

    dtypes_ : dict
        Dictionary mapping column name to either

        * instances of CategoricalDtype (pandas >= 0.21.0)
        * tuples of (categories, ordered)

    Notes
    -----
    This transformer only applies to pandas DataFrames.

    Examples
    --------
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4],
    ...                      "B": pd.Categorical(['a', 'a', 'a', 'b'])})
    >>> enc = OrdinalEncoder()
    >>> trn = enc.fit_transform(data)
    >>> trn
       A  B
    0  1  0
    1  2  0
    2  3  0
    3  4  1

    >>> enc.columns_
    Index(['A', 'B'], dtype='object')

    >>> enc.non_categorical_columns_
    Index(['A'], dtype='object')

    >>> enc.categorical_columns_
    Index(['B'], dtype='object')

    >>> enc.dtypes_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}
    """

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "OrdinalEncoder":
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        self.categorical_columns_ = X.select_dtypes(include=["category"]).columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)

        if _HAS_CTD:
            self.dtypes_ = {col: X[col].dtype for col in self.categorical_columns_}
        else:
            self.dtypes_ = {
                col: (X[col].cat.categories, X[col].cat.ordered)
                for col in self.categorical_columns_
            }

        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Ordinal encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns_)
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Unexpected type {}".format(type(X)))

        X = X.copy()
        for col in self.categorical_columns_:
            X[col] = X[col].cat.codes
        return X


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    """Applies transformers to columns of a pandas DataFrame.
    Returns a `pandas.DataFrame`, but otherwise behaves like
    `sklearn.compose.ColumnTransformer`.
    See the `sklearn.compose.ColumnTransformer` documentation for more information.
    """

    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
        transformer_weights=None,
    ):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )

    def _hstack(self, Xs: Iterable[Union[pd.Series, pd.DataFrame]]):
        """
        Stacks X horizontally.

        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        return pd.concat(Xs, axis="columns")


def make_column_transformer(*transformers, remainder: str = "drop"):  # noqa: D103
    # This is identical to scikit-learn's. We're just using our
    # ColumnTransformer instead.
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(
        transformer_list,
        remainder=remainder,
    )


make_column_transformer.__doc__ = getattr(  # noqa: B009
    sklearn.compose.make_column_transformer, "__doc__"
)
