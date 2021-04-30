import geopandas as geopd
import numpy as np
import openml


def download_and_transform():
    """Download data from openml and apply basic transformations."""
    dataset = openml.datasets.get_dataset(42092)
    df, y, _, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df["price"] = y

    df = df[(df["price"] < 1.5e6) & (df["price"] > 1e5)]
    df["price"] = np.log(df["price"])

    drop_cols = [
        "date",
        "sqft_lot",
        "grade",
        "sqft_above",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]
    df = df.drop(columns=drop_cols)
    return df


def get_map_data(df):
    """Load in map data and merge price info."""
    gdf_map = geopd.read_file("Zip_Codes/Zip_Codes.shp")
    gdf_map["ZIP"] = gdf_map["ZIP"].astype(str)
    # include certain regions that have no data to prevent "holes" in map
    gdf_map = gdf_map[
        gdf_map["ZIP"].isin(list(df.zipcode.unique()) + ["98051", "98158", "98057"])
    ]
    # dissolve boundary between shared zip codes
    gdf_map = gdf_map.dissolve(by="ZIP", aggfunc="sum").reset_index()  #
    return gdf_map.merge(
        df.groupby(["zipcode"])["price"].mean(),
        left_on="ZIP",
        right_on="zipcode",
        how="outer",
    )
