from typing import Any, Optional, Sequence, Union

import numpy as np
import packaging.version
import pandas as pd
import sklearn.compose
from pandas.api.types import is_categorical_dtype
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.utils.validation import check_is_fitted

_PANDAS_VERSION = packaging.version.parse(pd.__version__)
_HAS_CTD = _PANDAS_VERSION >= packaging.version.parse("0.21.0")


class Categorizer(BaseEstimator, TransformerMixin):
    """Transform columns of a DataFrame to categorical dtype.

    This is a useful pre-processing step for dummy, one-hot, or
    categorical encoding.

    Parameters
    ----------
    categories : mapping, optional

        A dictionary mapping column name to instances of
        ``pandas.api.types.CategoricalDtype``. Alternatively, a
        mapping of column name to ``(categories, ordered)`` tuples.

    columns : sequence, optional

        A sequence of column names to limit the categorization to.
        This argument is ignored when ``categories`` is specified.

    Notes
    -----
    This transformer only applies to ``dask.DataFrame`` and
    ``pandas.DataFrame``. By default, all object-type columns are converted to
    categoricals. The set of categories will be the values present in the
    column and the categoricals will be unordered. Pass ``dtypes`` to control
    this behavior.

    All other columns are included in the transformed output untouched.

    For ``dask.DataFrame``, any unknown categoricals will become known.

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

    def __init__(
        self, categories: Optional[dict] = None, columns: Optional[pd.Index] = None
    ):
        self.categories = categories
        self.columns = columns

    def _check_array(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO: refactor to check_array
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected a pandas or dask DataFrame, got " "{} instead".format(type(X))
            )
        return X

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "Categorizer":
        """Find the categorical columns.

        Parameters
        ----------
        X : pandas.DataFrame or dask.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        X = self._check_array(X)

        if self.categories is not None:
            # some basic validation
            columns = pd.Index(self.categories)
            categories = self.categories

        columns, categories = self._fit(X)

        self.columns_ = columns
        self.categories_ = categories
        return self

    def _fit(self, X: pd.DataFrame):
        if self.columns is None:
            columns = X.select_dtypes(include=["object", "category"]).columns
        else:
            columns = self.columns
        categories = {}
        for name in columns:
            col = X[name]
            if not is_categorical_dtype(col):
                # This shouldn't ever be hit on a dask.array, since
                # the object columns would have been converted to known cats
                # already
                col = pd.Series(col, index=X.index).astype("category")

            if _HAS_CTD:
                categories[name] = col.dtype
            else:
                categories[name] = (col.cat.categories, col.cat.ordered)

        return columns, categories

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Transform the columns in ``X`` according to ``self.categories_``.

        Parameters
        ----------
        X : pandas.DataFrame or dask.DataFrame
        y : ignored

        Returns
        -------
        X_trn : pandas.DataFrame or dask.DataFrame
            Same type as the input. The columns in ``self.categories_`` will
            be converted to categorical dtype.
        """
        check_is_fitted(self, "categories_")
        X = self._check_array(X).copy()
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
    columns : sequence, optional
        The columns to dummy encode. Must be categorical dtype.
        Dummy encodes all categorical dtype columns by default.
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
    This transformer only applies to dask and pandas DataFrames. For dask
    DataFrames, all of your categoricals should be known.

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

    >>> de.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                    A    B_a    B_b
    npartitions=2
    0              int64  uint8  uint8
    2                ...    ...    ...
    3                ...    ...    ...
    Dask Name: get_dummies, 4 tasks
    """

    def __init__(
        self, columns: Optional[Sequence[Any]] = None, drop_first: bool = False
    ):
        self.columns = columns
        self.drop_first = drop_first

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "DummyEncoder":
        """Determine the categorical columns to be dummy encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        columns = self.columns
        if columns is None:
            columns = X.select_dtypes(include=["category"]).columns
        else:
            for column in columns:
                assert is_categorical_dtype(X[column]), "Must be categorical"

        self.categorical_columns_ = columns
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

        if isinstance(X, pd.DataFrame):
            sample = X.iloc[:1]
        else:
            sample = X._meta_nonempty

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
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns_)
            )
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X, drop_first=self.drop_first, columns=self.columns)
        else:
            raise TypeError("Unexpected type {}".format(type(X)))

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Inverse dummy-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.transformed_columns_)

        non_cat = X[list(self.non_categorical_columns_)]

        cats = []
        for col in self.categorical_columns_:
            slice_ = self.categorical_blocks_[col]
            if _HAS_CTD:
                dtype = self.dtypes_[col]
                categories, ordered = dtype.categories, dtype.ordered  # type: ignore
            else:
                categories, ordered = self.dtypes_[col]

            # use .values to avoid warning from pandas
            cols_slice = list(X.columns[slice_])
            inds = X[cols_slice].values
            codes = inds.argmax(1)

            if self.drop_first:
                codes += 1
                codes[(inds == 0).all(1)] = 0

            series = pd.Series(
                pd.Categorical.from_codes(codes, categories, ordered=ordered),
                name=col,
            )

            cats.append(series)
        df = pd.concat([non_cat] + cats, axis=1)[self.columns_]
        return df


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal (integer) encode categorical columns.

    Parameters
    ----------
    columns : sequence, optional
        The columns to encode. Must be categorical dtype.
        Encodes all categorical dtype columns by default.

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
    This transformer only applies to dask and pandas DataFrames. For dask
    DataFrames, all of your categoricals should be known.

    The inverse transformation can be used on a dataframe or array.

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

    >>> enc.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                       A     B
    npartitions=2
    0              int64  int8
    2                ...   ...
    3                ...   ...
    Dask Name: assign, 8 tasks

    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "OrdinalEncoder":
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        columns = self.columns
        if columns is None:
            columns = X.select_dtypes(include=["category"]).columns
        else:
            for column in columns:
                assert is_categorical_dtype(X[column]), "Must be categorical"

        self.categorical_columns_ = columns
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
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns)
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Unexpected type {}".format(type(X)))

        X = X.copy()
        for col in self.categorical_columns_:
            X[col] = X[col].cat.codes
        return X

    def inverse_transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse ordinal-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)

        X = X.copy()
        for col in self.categorical_columns_:
            if _HAS_CTD:
                dtype = self.dtypes_[col]
                categories, ordered = dtype.categories, dtype.ordered  # type: ignore
            else:
                categories, ordered = self.dtypes_[col]

            # use .values to avoid warning from pandas
            codes = X[col].values

            series = pd.Series(
                pd.Categorical.from_codes(codes, categories, ordered=ordered),
                name=col,
            )

            X[col] = series

        return X


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    """Applies transformers to columns of an array or pandas DataFrame.

    EXPERIMENTAL: some behaviors may change between releases without
    deprecation.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the results combined into a single
    feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Read more in the :ref:`User Guide <column_transformer>`.

    .. versionadded:: 0.9.0

    .. note::

       This requires scikit-learn 0.20.0 or newer.

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, column(s)) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : string
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : estimator or {'passthrough', 'drop'}
            Estimator must support `fit` and `transform`. Special-cased
            strings 'drop' and 'passthrough' are accepted as well, to
            indicate to drop the columns or to pass them through untransformed,
            respectively.
        column(s) : string or int, array-like of string or int, slice, \
boolean mask array or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above.

    remainder : {'drop', 'passthrough'} or estimator, default 'drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support `fit` and `transform`.

    sparse_threshold : float, default = 0.3
        If the transformed output consists of a mix of sparse and dense data,
        it will be stacked as a sparse matrix if the density is lower than this
        value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this
        keyword will be ignored.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    preserve_dataframe : bool, (default=True)
        Whether to preserve preserve pandas DataFrames when concatenating
        the results.

        .. warning::

           The default behavior of keeping DataFrames differs from
           scikit-learn's current behavior. Set ``preserve_dataframe=False``
           if you need to ensure that the output matches scikit-learn's
           ColumnTransformer.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column). `fitted_transformer` can be an
        estimator, 'drop', or 'passthrough'. If there are remaining columns,
        the final element is a tuple of the form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : Bunch object, a dictionary with attribute access
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    sparse_output_ : boolean
        Boolean flag indicating whether the output of ``transform`` is a
        sparse matrix or a dense numpy array, which depends on the output
        of the individual transformers and the `sparse_threshold` keyword.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    See Also
    --------
    dask_ml.compose.make_column_transformer : convenience function for
        combining the outputs of multiple transformer objects applied to
        column subsets of the original feature space.

    Examples
    --------
    >>> from dask_ml.compose import ColumnTransformer
    >>> from sklearn.preprocessing import Normalizer
    >>> ct = ColumnTransformer(
    ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
    ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = np.array([[0., 1., 2., 2.],
    ...               [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of X to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)    # doctest: +NORMALIZE_WHITESPACE
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])

    """

    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
        transformer_weights=None,
        preserve_dataframe=True,
    ):
        self.preserve_dataframe = preserve_dataframe
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )

    def _hstack(self, Xs):
        """
        Stacks X horizontally.

        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        types = {type(X) for X in Xs}

        if self.sparse_output_:
            return sparse.hstack(Xs).tocsr()
        elif self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis="columns")
        else:
            return np.hstack(Xs)


def make_column_transformer(*transformers, **kwargs):  # noqa: D103
    # This is identical to scikit-learn's. We're just using our
    # ColumnTransformer instead.
    n_jobs = kwargs.pop("n_jobs", 1)
    remainder = kwargs.pop("remainder", "drop")
    preserve_dataframe = kwargs.pop("preserve_dataframe", True)
    if kwargs:
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(
        transformer_list,
        n_jobs=n_jobs,
        remainder=remainder,
        preserve_dataframe=preserve_dataframe,
    )


make_column_transformer.__doc__ = getattr(  # noqa: B009
    sklearn.compose.make_column_transformer, "__doc__"
)
