"""
Helpers for loading map data.

Some functions adapted from:
https://towardsdatascience.com/
mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac
"""

import numpy as np
import pandas as pd
import shapefile as shp
import shapely


def read_shapefile(shp_path):
    """Create dataframe from shapefile."""
    sf = shp.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]  # get headings from shp file
    records = [list(i) for i in sf.records()]  # get records from shp file
    shps = [
        _get_shapely_shape(s) for s in sf.shapes()
    ]  # get shapely polygons from shp file shapes
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(geometry=shps)
    return df


def _get_shapely_shape(s):
    if len(s.parts) == 1:
        return shapely.geometry.Polygon(s.points, [])
    return shapely.geometry.multipolygon.MultiPolygon(
        [[ps] for ps in np.split(s.points, s.parts[1:])], context_type="geojson"
    )


def _get_polygon_coords(x):
    if type(x) is shapely.geometry.multipolygon.MultiPolygon:
        return [list(x[j].exterior.coords) for j in range(len(x.geoms))]
    else:
        return [list(x.exterior.coords)]


def create_kings_county_map(df, df_shapefile):
    """Merge sales data and map data."""
    df_shapefile["ZIP"] = df_shapefile["ZIP"].astype(str)

    # include certain regions that have no data to prevent "holes" in map
    df_shapefile = df_shapefile[
        df_shapefile["ZIP"].isin(
            list(df.zipcode.unique()) + ["98051", "98158", "98057"]
        )
    ]
    # dissolve boundary between shared zip codes
    df_shapefile = (
        df_shapefile.groupby(by="ZIP")
        .agg({"geometry": shapely.ops.unary_union, "ZIPCODE": "first"})
        .reset_index()
    )
    df_shapefile["coords_avg"] = df_shapefile["geometry"].apply(
        lambda x: (x.centroid.coords[0][0] - 0.015, x.centroid.coords[0][1])
    )
    df_shapefile["coords"] = df_shapefile["geometry"].apply(
        lambda x: _get_polygon_coords(x)
    )
    return df_shapefile.merge(
        df.groupby(["zipcode"])["price"].mean(),
        left_on="ZIP",
        right_on="zipcode",
        how="outer",
    )
