import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__file__)


def process_data():
    """Process and save raw rossman data."""
    if not Path("processed_data").exists():
        os.mkdir("processed_data")
    for filename_stub in ["train", "test"]:
        print(f"Processing {filename_stub}")
        df = pd.read_csv(f"raw_data/{filename_stub}.csv")

        # fix data type (ints and strings)
        df["StateHoliday"] = df["StateHoliday"].astype(str)
        df["Open"] = df["Open"].astype(bool)

        df["Date"] = pd.to_datetime(df["Date"])
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month

        # make everything lower case
        df = df.rename(columns=lambda x: x.lower())

        # rename to snake case
        df = df.rename(
            columns={
                "dayofweek": "day_of_week",
                "stateholiday": "state_holiday",
                "schoolholiday": "school_holiday",
            }
        )

        # read store data state
        df_store = pd.read_csv("raw_data/store.csv")

        # rename columns
        df_store = df_store.rename(
            columns={
                "Store": "store",
                "StoreType": "store_type",
                "Assortment": "assortment",
                "CompetitionDistance": "competition_distance",
                "CompetitionOpenSinceMonth": "competition_open_since_month",
                "CompetitionOpenSinceYear": "competition_open_since_year",
                "Promo2": "promo2",
                "Promo2SinceWeek": "promo2_since_week",
                "Promo2SinceYear": "promo2_since_year",
                "PromoInterval": "promo_interval",
            }
        )

        index = df.index
        df = df.merge(df_store, how="left", on="store", indicator="_merge")
        df.index = index

        # check that we're matching everything
        df["_merge"].eq("both").all()
        df = df.drop(columns=["_merge"])

        df.to_parquet(f"processed_data/{filename_stub}.parquet")


def load_train():
    """Load training data set."""
    return pd.read_parquet("processed_data/train.parquet").sort_values(
        ["store", "date"]
    )


def load_test():
    """Load test data set."""
    return pd.read_parquet("processed_data/test.parquet").sort_values(["store", "date"])
