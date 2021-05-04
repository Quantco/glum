"""
Helpers for plotting maps.

Adapted from:
https://towardsdatascience.com/
mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac
"""

import numpy as np
import pandas as pd


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
    """Plot heatmap from dataframe coordinates."""
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
        x0 = row.coords_avg[0]
        y0 = row.coords_avg[1]
        ax.text(x0, y0, row[label_col], fontsize=10)
