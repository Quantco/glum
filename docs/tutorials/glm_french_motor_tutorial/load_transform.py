import numpy as np
import pandas as pd


def load_transform():
    """Load and transform data from OpenML.

    Summary of transformations:

    1. We cut the number of claims to a maximum of 4, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for details).
    2. We cut the exposure to a maximum of 1, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for details).
    3. We define ``'ClaimAmountCut'`` as the the claim amount cut at 100'000 per single claim
       (before aggregation per policy). Reason: For large claims, extreme value theory
       might apply. 100'000 is the 0.9984 quantile, any claims larger account for 25% of
       the overall claim amount. This is a well known phenomenon for third-party liability.
    4. We aggregate the total claim amounts per policy ID and join them to ``freMTPL2freq``.
    5. We fix ``'ClaimNb'`` as the claim number with claim amount greater zero.
    6. ``'VehPower'``, ``'VehAge'``, and ``'DrivAge'`` are clipped and/or digitized into bins so
       they can be used as categoricals later on.
    """
    # load the datasets
    # first row (=column names) uses "", all other rows use ''
    # use '' as quotechar as it is easier to change column names
    df = pd.read_csv(
        "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff", quotechar="'"
    )

    # rename column names '"name"' => 'name'
    df = df.rename(lambda x: x.replace('"', ""), axis="columns")
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
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    # Clip and/or digitize predictors into bins
    df["VehPower"] = np.minimum(df["VehPower"], 9)
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )
    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

    return df
