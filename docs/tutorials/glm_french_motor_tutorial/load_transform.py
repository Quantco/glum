import numpy as np
import pandas as pd


def load_transform():
    """Load and transform data from openml."""
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

    # Note: Zero claims must be ignored in severity models,
    # because the support is (0, inf) not [0, inf).
    # Thus, we define the number of claims with positive claim amount for later use.
    df["ClaimNb_pos"] = df["ClaimNb"]
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb_pos"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["ClaimNb_pos"] = df["ClaimNb_pos"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    df["VehPower"] = np.minimum(df["VehPower"], 9)
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )
    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

    return df
