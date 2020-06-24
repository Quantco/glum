from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dask_ml.compose import ColumnTransformer
from dask_ml.preprocessing import DummyEncoder
from git_root import git_root
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from ..util import exposure_and_offset_to_weights

# taken from https://github.com/lorentzenchr/Tutorial_freMTPL2/blob/master/glm_freMTPL2_example.ipynb
# Modified to generate data sets of different sizes


def create_raw_data() -> None:
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
    print(
        "There are {} rows in freMTPL2sev that do not have a matching IDpol in freMTPL2freq.\n"
        "They have a ClaimAmountCut of {}.".format(
            df2[df2._merge == "left_only"].shape[0],
            df2.ClaimAmountCut[df2._merge == "left_only"].sum(),
        )
    )

    round(df_sev.ClaimAmountCut.sum() - df.ClaimAmountCut.sum(), 2)

    print(
        "Number or rows with ClaimAmountCut > 0 and ClaimNb == 0: {}".format(
            df[(df.ClaimAmountCut > 0) & (df.ClaimNb == 0)].shape[0]
        )
    )

    # 9116 zero claims
    print(
        "Number or rows with ClaimAmountCut = 0 and ClaimNb >= 1: {}".format(
            df[(df.ClaimAmountCut == 0) & (df.ClaimNb >= 1)].shape[0]
        )
    )

    # Note: Zero claims must be ignored in severity models, because the support is (0, inf) not [0, inf).
    # Therefore, we define the number of claims with positive claim amount for later use.
    df["ClaimNb_pos"] = df["ClaimNb"]
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb_pos"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["ClaimNb_pos"] = df["ClaimNb_pos"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    df.to_parquet(git_root("data/insurance.parquet"))


def get_to_df_trans(columns: List[str]) -> Tuple[str, FunctionTransformer]:
    return "to_df", FunctionTransformer(lambda x: pd.DataFrame(x, columns=columns))


categorical_transformer = (
    "to_cat",
    FunctionTransformer(lambda x: x.astype("category"), validate=False),
)
reset_index_transformer = (
    "reset_index",
    FunctionTransformer(lambda x: x.reset_index(drop=True), validate=False),
)

categorical_trans_list = [
    categorical_transformer,
    ("OHE", DummyEncoder()),
    reset_index_transformer,
]


def gen_col_trans() -> Tuple[ColumnTransformer, List[str]]:
    """Generate a ColumnTransformer and list of names.

    With drop=False and standardize=False, the transformer corresponds to the GLM of the case study paper.

    drop = False does encode k categories with k binary features (redundant).
    standardize = True standardizes numerical features.
    """
    column_trans = ColumnTransformer(
        [
            # VehPower 4, 5, 6, 7, 8, 9, drop=4
            (
                "VehPower_cat",
                Pipeline(
                    [
                        (
                            "cut_9",
                            FunctionTransformer(
                                lambda x: np.minimum(x, 9), validate=False
                            ),
                        ),
                    ]
                    + categorical_trans_list
                ),
                ["VehPower"],
            ),
            # VehAge intervals [0,1), [1, 10], (10, inf), drop=[1,10]
            (
                "VehAge_cat",
                Pipeline(
                    [
                        (
                            "bin",
                            FunctionTransformer(
                                lambda x: np.digitize(
                                    np.where(x == 10, 9, x), bins=[1, 10]
                                ),
                                validate=False,
                            ),
                        ),
                        get_to_df_trans(["VehAge_cat"]),
                    ]
                    + categorical_trans_list
                ),
                ["VehAge"],
            ),
            # DrivAge intervals [18,21), [21,26), [26,31), [31,41), [41,51), [51,71),[71,∞), drop=[41,51)
            (
                "DrivAge_cat",
                Pipeline(
                    [
                        (
                            "bin",
                            FunctionTransformer(
                                lambda x: np.digitize(x, bins=[21, 26, 31, 41, 51, 71]),
                                validate=False,
                            ),
                        ),
                        get_to_df_trans(["DrivAge_cat"]),
                    ]
                    + categorical_trans_list
                ),
                ["DrivAge"],
            ),
            (
                "BonusMalus",
                Pipeline(
                    [
                        (
                            "cutat150",
                            FunctionTransformer(
                                lambda x: np.minimum(x, 150), validate=False
                            ),
                        ),
                        reset_index_transformer,
                    ]
                ),
                ["BonusMalus"],
            ),
            ("VehBrand_cat", Pipeline(categorical_trans_list), ["VehBrand"]),
            ("VehGas_Regular", Pipeline(categorical_trans_list), ["VehGas"]),
            (
                "Density_log",
                Pipeline(
                    [
                        (
                            "log",
                            FunctionTransformer(lambda x: np.log(x), validate=False),
                        ),
                        get_to_df_trans(["Density"]),
                        reset_index_transformer,
                    ],
                ),
                ["Density"],
            ),
            ("Region_cat", Pipeline(categorical_trans_list), ["Region"]),
            (
                "Area_ord",
                Pipeline(
                    [
                        ("OE", OrdinalEncoder()),
                        (
                            "plus_1",
                            FunctionTransformer(lambda x: x + 1, validate=False),
                        ),
                        get_to_df_trans(["Area"]),
                    ]
                ),
                ["Area"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
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
            "distribution must be one of ['poisson', 'gamma', 'tweedie', 'gaussian', 'binomial'] "
            f"not {distribution}."
        )

    return y, exposure


def read_insurance_data(
    num_rows: Optional[int], noise: Optional[float], distribution: str
) -> pd.DataFrame:
    df = pd.read_parquet(git_root("data/insurance.parquet"))

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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the tutorial data set from the sklearn fork and save it to disk."""

    df = read_insurance_data(num_rows, noise, distribution)

    col_trans_GLM1, _ = gen_col_trans()
    y, exposure = compute_y_exposure(df, distribution)

    return col_trans_GLM1.fit_transform(df), y, exposure


def generate_real_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a version of the tutorial data set with many features."""
    df = read_insurance_data(num_rows, noise, distribution)

    transformer = ColumnTransformer(
        [
            (
                "numerics",
                Pipeline([("id", FunctionTransformer()), reset_index_transformer]),
                lambda x: x.select_dtypes(["number"]).columns,
            ),
            (
                "one_hot_encode",
                Pipeline(categorical_trans_list),
                [
                    "Area",
                    "VehPower",
                    "VehAge",
                    "DrivAge",
                    "BonusMalus",
                    "VehBrand",
                    "VehGas",
                    "Region",
                ],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    y, exposure = compute_y_exposure(df, distribution)
    return transformer.fit_transform(df), y, exposure


def generate_intermediate_insurance_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the tutorial data set from the sklearn fork and save it to disk."""

    df = read_insurance_data(num_rows, noise, distribution)
    df["BonusMalusClipped"] = df["BonusMalus"].clip(50, 100)

    col_trans_GLM1, _ = gen_col_trans()
    col_trans_GLM1.transformers.append(
        ("BonusMalusClipped", Pipeline(categorical_trans_list), ["BonusMalusClipped"],)
    )
    y, exposure = compute_y_exposure(df, distribution)

    return col_trans_GLM1.fit_transform(df), y, exposure
