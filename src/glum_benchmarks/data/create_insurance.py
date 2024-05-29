import os
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import sklearn.compose
from git_root import git_root
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from ..util import exposure_and_offset_to_weights

# taken from https://github.com/lorentzenchr/Tutorial_freMTPL2/blob/master/glm_freMTPL2_example.ipynb  # noqa: E501
# Modified to generate data sets of different sizes


def create_insurance_raw_data(verbose=False) -> None:
    """Do some basic processing on the data that we will later transform into our \
    benchmark data sets."""
    # load the datasets
    # first row (=column names) uses "", all other rows use ''
    # use '' as quotechar as it is easier to change column names
    df = pd.read_csv(
        "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff", quotechar="'"
    )

    # rename column names '"name"' => 'name'
    df.rename(lambda x: x.replace('"', ""), axis="columns", inplace=True)
    df["IDpol"] = df["IDpol"].astype(np.int64)
    df.set_index("IDpol", inplace=True)

    df_sev = pd.read_csv(
        "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff", index_col=0
    )

    # join ClaimAmount from df_sev to df:
    #   1. cut ClaimAmount at 100_000
    #   2. aggregate ClaimAmount per IDpol
    #   3. join by IDpol
    df_sev["ClaimAmountCut"] = df_sev["ClaimAmount"].clip(upper=100_000)
    df = df.join(df_sev.groupby(level=0).sum(), how="left")
    df.fillna(value={"ClaimAmount": 0, "ClaimAmountCut": 0}, inplace=True)

    # Check if there are IDpol in df_sev that do not match any IDPol in df.
    df2 = pd.merge(
        df_sev,
        df.loc[:, ["ClaimNb"]],
        left_index=True,
        right_index=True,
        how="outer",
        indicator=True,
    )
    if verbose:
        print(
            "There are {} rows in freMTPL2sev that do not have a matching IDpol in "
            "freMTPL2freq. They have a ClaimAmountCut of {}.".format(
                df2[df2._merge == "left_only"].shape[0],
                df2.ClaimAmountCut[df2._merge == "left_only"].sum(),
            )
        )

    round(df_sev.ClaimAmountCut.sum() - df.ClaimAmountCut.sum(), 2)

    if verbose:
        print(
            "Number or rows with ClaimAmountCut > 0 and ClaimNb == 0: "
            f"{df[(df.ClaimAmountCut > 0) & (df.ClaimNb == 0)].shape[0]}"
        )

    # 9116 zero claims
    if verbose:
        print(
            "Number or rows with ClaimAmountCut = 0 and ClaimNb >= 1: "
            f"{df[(df.ClaimAmountCut == 0) & (df.ClaimNb >= 1)].shape[0]}"
        )

    # Note: Zero claims must be ignored in severity models, because the support is
    # (0, inf) not [0, inf). Therefore, we define the number of claims with positive
    # claim amount for later use.
    df["ClaimNb_pos"] = df["ClaimNb"]
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb_pos"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["ClaimNb_pos"] = df["ClaimNb_pos"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    out_path = git_root("data/insurance.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)


class Categorizer(BaseEstimator, TransformerMixin):
    """Transform columns of a DataFrame to categorical dtype."""

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "Categorizer":
        """Find the categorical columns."""
        columns = X.columns
        categories = {}
        for name in columns:
            col = X[name]
            if str(col.dtype) != "category":  # type: ignore
                col = pd.Series(col, index=X.index).astype("category")
            categories[name] = col.dtype

        self.columns_ = columns
        self.categories_ = categories
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Transform the columns in ``X`` according to ``self.categories_``."""
        check_is_fitted(self, "categories_")
        categories = self.categories_

        for k, dtype in categories.items():
            if not isinstance(dtype, pd.api.types.CategoricalDtype):
                dtype = pd.api.types.CategoricalDtype(*dtype)
            X[k] = X[k].astype(dtype)

        return X


def get_categorizer(col_name: str, name="cat") -> tuple[str, Categorizer]:
    """Get a Categorizer."""
    return name, Categorizer()


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal (integer) encode categorical columns."""

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "OrdinalEncoder":
        """Determine the categorical columns to be encoded."""
        self.categorical_columns_ = X.select_dtypes(include=["category"]).columns
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.DataFrame:
        """Ordinal encode the categorical columns in X."""
        check_is_fitted(self, "categorical_columns_")
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

    def _hstack(self, Xs: Iterable[Union[pd.Series, pd.DataFrame]], *, n_samples=None):
        """Stacks X horizontally."""
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


def func_returns_df(
    fn: Callable[[pd.DataFrame], np.ndarray],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Take a function that takes a dataframe and returns a Numpy array, and return a \
    function that takes a dataframe and returns a dataframe.

    fn: Function that takes a dataframe and returns a numpy array
    Returns: Function that takes a dataframe and returns a dataframe with the values
             determined by the original function, and the index and columns of the
             original dataframe.
    """
    return lambda x: x.assign(**{x.columns[0]: fn(x)})


def gen_col_trans() -> tuple[Any, list[str]]:
    """Generate a ColumnTransformer and list of names.

    The transformer corresponds to the GLM of the case study paper.

    Encodes k categories with k binary features (redundant).
    """
    column_trans = make_column_transformer(
        # VehPower 4, 5, 6, 7, 8, 9, drop=4
        (
            Pipeline(
                [
                    (
                        "cut_9",
                        FunctionTransformer(lambda x: np.minimum(x, 9), validate=False),
                    ),
                    get_categorizer("VehPower"),
                ]
            ),
            ["VehPower"],
        ),
        # VehAge intervals [0,1), [1, 10], (10, inf), drop=[1,10]
        (
            Pipeline(
                [
                    (
                        "bin",
                        FunctionTransformer(
                            func_returns_df(
                                lambda x: np.digitize(
                                    np.where(x == 10, 9, x), bins=[1, 10]
                                )
                            ),
                            validate=False,
                        ),
                    ),
                    get_categorizer("VehAge"),
                ]
            ),
            ["VehAge"],
        ),
        # DrivAge intervals [18,21), [21,26), [26,31), [31,41), [41,51), [51,71),[71,∞),
        # drop=[41,51)
        (
            Pipeline(
                [
                    (
                        "bin",
                        FunctionTransformer(
                            func_returns_df(
                                lambda x: np.digitize(x, bins=[21, 26, 31, 41, 51, 71])
                            ),
                            validate=False,
                        ),
                    ),
                    get_categorizer("DrivAge"),
                ]
            ),
            ["DrivAge"],
        ),
        (
            Pipeline(
                [
                    (
                        "cutat150",
                        FunctionTransformer(
                            lambda x: np.minimum(x, 150), validate=False
                        ),
                    )
                ]
            ),
            ["BonusMalus"],
        ),
        (Pipeline([get_categorizer("VehBrand")]), ["VehBrand"]),
        (Pipeline([get_categorizer("VehGas")]), ["VehGas"]),
        (FunctionTransformer(np.log, validate=False), ["Density"]),
        (Pipeline([get_categorizer("Region")]), ["Region"]),
        (
            Pipeline(
                [
                    get_categorizer("Area"),
                    ("OE", OrdinalEncoder()),
                    (
                        "plus_1",
                        FunctionTransformer(lambda x: x + 1, validate=False),
                    ),
                ]
            ),
            ["Area"],
        ),
        remainder="drop",
    )
    column_trans_names = [
        "VehPower_4",
        "VehPower_5",
        "VehPower_6",
        "VehPower_7",
        "VehPower_8",
        "VehPower_9",
        "VehAge_[0,1)",
        "VehAge_[1, 10]",
        "VehAge_(10,inf)",
        "DrivAge_[18,21)",
        "DrivAge_[21,26)",
        "DrivAge_[26,31)",
        "DrivAge_[31,41)",
        "DrivAge_[41,51)",
        "DrivAge_[51,71)",
        "DrivAge_[71,inf)",
        "BonusMalus",
        "VehBrand_B10",
        "VehBrand_B11",
        "VehBrand_B12",
        "VehBrand_B13",
        "VehBrand_B14",
        "VehBrand_B1",
        "VehBrand_B2",
        "VehBrand_B3",
        "VehBrand_B4",
        "VehBrand_B5",
        "VehBrand_B6",
        "VehGas_Diesel",
        "VehGas_Regular",
        "Density_log",
        "Region_R11",
        "Region_R21",
        "Region_R22",
        "Region_R23",
        "Region_R24",
        "Region_R25",
        "Region_R26",
        "Region_R31",
        "Region_R41",
        "Region_R42",
        "Region_R43",
        "Region_R52",
        "Region_R53",
        "Region_R54",
        "Region_R72",
        "Region_R73",
        "Region_R74",
        "Region_R82",
        "Region_R83",
        "Region_R91",
        "Region_R93",
        "Region_R94",
        "Area_ord",
    ]
    column_trans_names = [
        i
        for i in column_trans_names
        if i
        not in [
            "VehPower_4",
            "VehAge_[1, 10]",
            "DrivAge_[41,51)",
            "VehBrand_B1",
            "VehGas_Diesel",
            "Region_R24",
        ]
    ]
    return column_trans, column_trans_names


def add_noise(df: pd.DataFrame, noise: float) -> pd.DataFrame:
    """Add noise by swapping out data points."""
    np.random.seed(43212)
    for col in df.columns:
        if col in ["ClaimNb", "Exposure", "ClaimAmountCut", "ClaimNb_pos"]:
            continue
        swap = np.random.uniform(size=len(df)) < noise
        shuffle = np.random.choice(df[col], size=len(df))
        df.loc[swap, col] = shuffle[swap]

    return df


def compute_y_exposure(df, distribution):
    """Compute y and exposure depending on distribution."""
    if distribution == "poisson":
        exposure = df["Exposure"].values
        y = df["ClaimNb"].values / exposure
    elif distribution in ["gamma", "gaussian"]:
        exposure = df["ClaimNb_pos"].values
        y = df["ClaimAmountCut"].values / exposure
    elif "tweedie" in distribution:
        exposure = df["Exposure"].values
        y = df["ClaimAmountCut"].values / exposure
    elif distribution == "binomial":
        exposure = df["Exposure"].values
        df["HasClaim"] = (df["ClaimNb"] > 0).astype(np.int32)
        y = df["HasClaim"].values
    else:
        raise ValueError(
            "distribution must be one of ['poisson', 'gamma', 'tweedie', 'gaussian', "
            f"'binomial'] not {distribution}."
        )

    return y, exposure


def _read_insurance_data(
    num_rows: Optional[int], noise: Optional[float], distribution: str
) -> pd.DataFrame:
    path = git_root("data/insurance.parquet")
    if not os.path.exists(path):
        create_insurance_raw_data()
    df = pd.read_parquet(path)

    if distribution in ["gamma", "gaussian"]:
        df = df.query("ClaimAmountCut > 0")

    if num_rows is not None:
        # if we're oversampling, set default value for noise to 0.05
        # can be turned off by setting noise to zero
        if noise is None and num_rows > len(df):
            noise = 0.05
        df = df.sample(n=num_rows, replace=True, random_state=12345)

    if noise is not None:
        df = add_noise(df, noise=noise)
    return df


def generate_narrow_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the tutorial data set from the sklearn fork and save it to disk."""
    df = _read_insurance_data(num_rows, noise, distribution)

    col_trans_GLM1, _ = gen_col_trans()
    y, exposure = compute_y_exposure(df, distribution)

    return col_trans_GLM1.fit_transform(df), y, exposure


def generate_real_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real insurance data set."""
    df = pd.read_parquet(git_root("data", "outcomes.parquet"))
    X = pd.read_parquet(git_root("data", "X.parquet"))

    if distribution != "poisson":
        raise NotImplementedError("distribution must be poisson")

    # restrict X and df to train set
    train_set = df["sample"] == "train"
    X = X.loc[train_set].reset_index(drop=True)
    df = df.loc[train_set].reset_index(drop=True)

    # subsample
    if num_rows is not None:
        idx = df.sample(n=num_rows).index
        df = df.loc[idx].reset_index(drop=True)
        X = X.loc[idx].reset_index(drop=True)

    # account for exposure and offsets
    y, weights = exposure_and_offset_to_weights(
        power=1,
        y=df["sanzkh02"],
        exposure=df["je"],
        offset=df["offset_kh_sach_frequenz"],
    )

    return X, y, weights


def generate_wide_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a version of the tutorial data set with many features."""
    df = _read_insurance_data(num_rows, noise, distribution)
    cat_cols = [
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Region",
    ]

    transformer = make_column_transformer(
        (
            FunctionTransformer(),
            lambda x: [
                elmt
                for elmt in x.select_dtypes(["number"]).columns
                if elmt not in cat_cols
            ],
        ),
        (
            Pipeline([get_categorizer(col, "cat_" + col) for col in cat_cols]),
            cat_cols,
        ),
        remainder="drop",
    )
    y, exposure = compute_y_exposure(df, distribution)

    return transformer.fit_transform(df), y, exposure


def generate_intermediate_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the tutorial data set from the sklearn fork and save it to disk."""
    df = _read_insurance_data(num_rows, noise, distribution)
    df["BonusMalusClipped"] = df["BonusMalus"].clip(50, 100)

    col_trans_GLM1, _ = gen_col_trans()
    col_trans_GLM1.transformers.append(
        (
            "BonusMalusClipped",
            Pipeline([get_categorizer("BonusMalusClipped")]),
            ["BonusMalusClipped"],
        )
    )
    y, exposure = compute_y_exposure(df, distribution)

    return col_trans_GLM1.fit_transform(df), y, exposure
