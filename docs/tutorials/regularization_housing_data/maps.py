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
    """Create data frame from shapefile.

    Parameters
    ----------
    shp_path : string
        Path to ``.shp`` file for map.

    Returns
    -------
    df : pandas.DataFrame
        Data frame with postcodes and corresponding geometric features.
    """
    sf = shp.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = [list(i) for i in sf.records()]
    shps = [_get_shapely_shape(s) for s in sf.shapes()]
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


def create_kings_county_map_df(df, df_shapefile):
    """Create data frame with sales and geometric information.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with housing sales information per postcode.
    df_shapefile: pandas.DataFrame
        Data frame with map geometry information per postcode.

    Returns
    -------
    df : pandas.DataFrame
        Data frame with sales and geometric information. Includes coordinates
        in array form and points for region centroids (for labeling).
    """
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
    df_shapefile["centroids"] = df_shapefile["geometry"].apply(
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


def _calc_color(data, vmin, vmax):
    color_sq = [
        "#F5EEF8",
        "#EBDEF0",
        "#D7BDE2",
        "#C39BD3",
        "#AF7AC5",
        "#9B59B6",
        "#884EA0",
        "#76448A",
        "#633974",
    ]
    new_data = data.copy()
    if (vmin is not None) or (vmax is not None):
        new_data = np.clip(new_data, vmin, vmax)
    new_data, _ = pd.cut(new_data, 9, retbins=True, labels=list(range(9)))
    return [color_sq[val] if not pd.isna(val) else None for val in new_data]


def plot_heatmap(df, label_col, data_col, ax, vmin=None, vmax=None):
    """Plot geographic heatmap on provided axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with geometric and sales information (or any other
        attributes to determine heat map intensity).
    label_col: str
        The column of the data frame to use as labels on the heatmap.
    data_col: string
        The column of the data frame with which to determine heatmap intensity.
    ax: matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes to plot on.
    vmin, vmax: float, optional (default=None))
        The range covered by the heatmap. With default (``None``),
        complete range of data covered.
    """
    df["color_ton"] = _calc_color(df[data_col], vmin, vmax)
    for point_sets in df.coords:
        for points in point_sets:
            x = [i[0] for i in points[:]]
            y = [i[1] for i in points[:]]
            ax.plot(x, y, "k")
    for _, row in df.iterrows():
        for dim in range(len(row.coords)):
            x_lon = np.zeros((len(row.coords[dim]), 1))
            y_lat = np.zeros((len(row.coords[dim]), 1))
            for ip in range(len(row.coords[dim])):
                x_lon[ip] = row.coords[dim][ip][0]
                y_lat[ip] = row.coords[dim][ip][1]
            if row.color_ton is None:
                ax.fill(x_lon, y_lat, "#FFFFFF", hatch="///", edgecolor="#747474")
            else:
                ax.fill(x_lon, y_lat, row.color_ton)
        x0 = row["centroids"][0]
        y0 = row["centroids"][1]
        ax.text(x0, y0, row[label_col], fontsize=10)
