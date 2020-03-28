import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# taken from https://github.com/lorentzenchr/Tutorial_freMTPL2/blob/master/glm_freMTPL2_example.ipynb


def gen_col_trans(drop=True, standardize=False):
    """Generate a ColumnTransformer and list of names.

    With drop=False and standardize=False, the transformer corresponds to the GLM of the case study paper.

    drop = False does encode k categories with k binary features (redundant).
    standardize = True standardizes numerical features.
    """
    # drop dictionary
    dd = {
        "VehPower": [4],
        "VehAge": [1],
        "DrivAge": [4],
        "VehBrand": ["B1"],
        "VehGas": ["Diesel"],
        "Region": ["R24"],
    }
    if drop is False:
        for key, value in dd.items():
            dd[key] = None
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
                        (
                            "OHE",
                            OneHotEncoder(
                                categories="auto", drop=dd["VehPower"], sparse=False
                            ),
                        ),
                    ]
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
                        (
                            "OHE",
                            OneHotEncoder(
                                categories="auto", drop=dd["VehAge"], sparse=False
                            ),
                        ),
                    ]
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
                        (
                            "OHE",
                            OneHotEncoder(
                                categories="auto", drop=dd["DrivAge"], sparse=False
                            ),
                        ),
                    ]
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
                        )
                    ]
                    + ([("norm", StandardScaler())] if standardize else [])
                ),
                ["BonusMalus"],
            ),
            (
                "VehBrand_cat",
                OneHotEncoder(drop=dd["VehBrand"], sparse=False),
                ["VehBrand"],
            ),
            (
                "VehGas_Regular",
                OneHotEncoder(drop=dd["VehGas"], sparse=False),
                ["VehGas"],
            ),
            (
                "Density_log",
                Pipeline(
                    [("log", FunctionTransformer(lambda x: np.log(x), validate=False))]
                    + ([("norm", StandardScaler())] if standardize else [])
                ),
                ["Density"],
            ),
            ("Region_cat", OneHotEncoder(drop=dd["Region"]), ["Region"]),
            (
                "Area_ord",
                Pipeline(
                    [
                        ("OE", OrdinalEncoder()),
                        (
                            "plus_1",
                            FunctionTransformer(lambda x: x + 1, validate=False),
                        ),
                    ]
                    + ([("norm", StandardScaler())] if standardize else [])
                ),
                ["Area"],
            ),
        ],
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
    if drop:
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

col_trans_GLM1, col_trans_GLM1_names = gen_col_trans(drop=True, standardize=False)
X = col_trans_GLM1.fit_transform(df)
z = df["ClaimNb"].values
exposure = df["Exposure"].values
# claims frequency
y = z / exposure

import ipdb
ipdb.set_trace()
# save to disk
X = pd.DataFrame(X)
X.columns = col_trans_GLM1_names
X.assign(y=y, exposure=exposure).to_parquet(git_root("data/data.parquet"))
