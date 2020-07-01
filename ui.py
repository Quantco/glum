# -*- coding: utf-8 -*-
"""

"""
# import copy
import math

import ipywidgets as widgets
import numpy as np
import pandas as pd
from base import GLMBase
from bokeh.io import output_notebook, show  # export_png, output_file
from bokeh.models import (  # ColumnDataSource,
    HoverTool,
    LinearAxis,
    NumeralTickFormatter,
    Range1d,
    Title,
)
from bokeh.plotting import figure
from ipywidgets import fixed, interactive

# from bokeh.layouts import gridplot
# from bokeh.palettes import Spectral9  # pylint: disable=no-name-in-module


class GLM(GLMBase):
    def view(self, data=None):
        """The main UI of the GLMUtility"""

        output_notebook()

        def view_one_way(var, transform, obs, fitted, model, ci, data):
            if data is None:
                temp = pd.pivot_table(
                    data=self.transformed_data,
                    index=[var],
                    values=[self.dependent, self.weight, "Fitted Avg"],
                    aggfunc=np.sum,
                )
            else:
                temp = pd.pivot_table(
                    data=self.predict(data),
                    index=[var],
                    values=[self.dependent, self.weight, "Fitted Avg"],
                    aggfunc=np.sum,
                )
            temp["Observed"] = temp[self.dependent] / temp[self.weight]
            temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
            temp = temp.merge(
                self.PDP[var][["Model", "CI_U", "CI_L"]],
                how="inner",
                left_index=True,
                right_index=True,
            )
            if transform == "Predicted Value":
                for item in ["Model", "CI_U", "CI_L"]:
                    temp[item] = self._link_transform(temp[item], "predicted value")
            else:
                for item in ["Observed", "Fitted"]:
                    temp[item] = self._link_transform(temp[item], "linear predictor")
            y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
            hover = HoverTool(
                tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
            )  # 'vline')
            if type(temp.index) == pd.core.indexes.base.Index:  # Needed for categorical
                p = figure(
                    plot_width=800,
                    y_range=y_range,
                    x_range=list(temp.index),
                    toolbar_location="right",
                    toolbar_sticky=False,
                )
            else:
                p = figure(
                    plot_width=800,
                    y_range=y_range,
                    toolbar_location="right",
                    toolbar_sticky=False,
                )

            # setting bar values
            p.add_tools(hover)
            p.add_layout(
                Title(text=var, text_font_size="12pt", align="center"), "above"
            )
            p.yaxis[0].axis_label = self.weight
            p.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
            p.add_layout(
                LinearAxis(
                    y_range_name="foo", axis_label=self.dependent + "/" + self.weight
                ),
                "right",
            )
            h = np.array(temp[self.weight])
            # Correcting the bottom position of the bars to be on the 0 line.
            adj_h = h / 2
            # add bar renderer
            p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
            # add line to secondondary axis
            p.extra_y_ranges = {
                "foo": Range1d(
                    start=min(temp["Observed"].min(), temp["Model"].min()) / 1.1,
                    end=max(temp["Observed"].max(), temp["Model"].max()) * 1.1,
                )
            }
            # p.add_layout(LinearAxis(y_range_name="foo"), 'right')
            # Observed Average line values
            if obs:
                p.line(
                    temp.index,
                    temp["Observed"],
                    line_width=2,
                    color="#ff69b4",
                    y_range_name="foo",
                )
            if fitted:
                p.line(
                    temp.index,
                    temp["Fitted"],
                    line_width=2,
                    color="#006400",
                    y_range_name="foo",
                )
            if model:
                p.line(
                    temp.index,
                    temp["Model"],
                    line_width=2,
                    color="#00FF00",
                    y_range_name="foo",
                )
            if ci:
                p.line(
                    temp.index,
                    temp["CI_U"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
                p.line(
                    temp.index,
                    temp["CI_L"],
                    line_width=2,
                    color="#db4437",
                    y_range_name="foo",
                )
            p.xaxis.major_label_orientation = math.pi / 4
            show(p)

        var = widgets.Dropdown(
            options=self.independent, description="Field:", value=self.independent[0]
        )
        transform = widgets.ToggleButtons(
            options=["Linear Predictor", "Predicted Value"],
            button_style="",
            value="Predicted Value",
            description="Transform:",
        )
        obs = widgets.ToggleButton(
            value=False, description="Observed Value", button_style="info"
        )
        fitted = widgets.ToggleButton(
            value=False, description="Fitted Value", button_style="info"
        )
        model = widgets.ToggleButton(
            value=False, description="Model Value", button_style="warning"
        )
        ci = widgets.ToggleButton(
            value=False, description="Conf. Interval", button_style="warning"
        )
        vw = interactive(
            view_one_way,
            var=var,
            transform=transform,
            obs=obs,
            fitted=fitted,
            model=model,
            ci=ci,
            data=fixed(data),
        )
        return widgets.VBox(
            (
                widgets.HBox((var, transform)),
                widgets.HBox((obs, fitted, model, ci)),
                vw.children[-1],
            )
        )

    # def lift_chart(
    #     self,
    #     data=None,
    #     deciles=10,
    #     title="",
    #     table=False,
    #     dont_show_give=False,
    #     file_name=None,
    # ):
    #     """ n-Decile lift chart
    #     """
    #     if data is None:
    #         data = self.transformed_data
    #     else:
    #         data = self.predict(data)
    #         data["Fitted Avg"] = data["Fitted Avg"] * data[self.weight]
    #     # data = data.reset_index()
    #     temp = data[[self.weight, self.dependent, "Fitted Avg"]]
    #     temp = copy.deepcopy(temp)
    #     temp["sort"] = temp["Fitted Avg"] / temp[self.weight]
    #     temp = temp.sort_values("sort", kind="mergesort")
    #     # temp['decile'] = (temp[self.weight].cumsum()/((sum(temp[self.weight])*1.00001)/10)+1).apply(np.floor)
    #     temp["decile_initial"] = (
    #         temp[self.weight].cumsum()
    #         / ((sum(temp[self.weight]) * 1.00001) / (deciles * 2))
    #         + 1
    #     ).apply(np.floor)
    #     decile_map = {
    #         item + 1: np.ceil((item + 1) / 2 + 0.01) for item in range(deciles * 2)
    #     }
    #     # decile_map = {1:1,2:2,3:2,4:3,5:3,6:4,7:4,8:5,9:5,10:6,11:6,12:7,13:7,14:8,15:8,16:9,17:9,18:10,19:10,20:11}
    #     temp["decile"] = temp["decile_initial"].map(decile_map)
    #     temp = pd.pivot_table(
    #         data=temp,
    #         index=["decile"],
    #         values=[self.dependent, self.weight, "Fitted Avg"],
    #         aggfunc="sum",
    #     )
    #     temp["Observed"] = temp[self.dependent] / temp[self.weight]
    #     temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
    #     y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
    #     hover = HoverTool(
    #         tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
    #     )  # 'vline')
    #     p = figure(
    #         plot_width=600,
    #         plot_height=375,
    #         y_range=y_range,
    #         title="Lift Chart",
    #         toolbar_sticky=False,
    #     )  # , x_range=list(temp.index)
    #     p.add_tools(hover)
    #     h = np.array(temp[self.weight])
    #     # Correcting the bottom position of the bars to be on the 0 line.
    #     adj_h = h / 2
    #     # add bar renderer
    #     p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
    #     # add line to secondondary axis
    #     p.extra_y_ranges = {
    #         "foo": Range1d(
    #             start=min(temp["Observed"].min(), temp["Fitted"].min()) / 1.1,
    #             end=max(temp["Observed"].max(), temp["Fitted"].max()) * 1.1,
    #         )
    #     }
    #     if title != "":
    #         p.add_layout(
    #             Title(text=title, text_font_size="12pt", align="center"), "above"
    #         )
    #     p.add_layout(LinearAxis(y_range_name="foo"), "right")
    #     # Observed Average line values
    #     p.line(
    #         temp.index,
    #         temp["Observed"],
    #         line_width=2,
    #         color="#ff69b4",
    #         y_range_name="foo",
    #     )
    #     p.line(
    #         temp.index,
    #         temp["Fitted"],
    #         line_width=2,
    #         color="#006400",
    #         y_range_name="foo",
    #     )
    #     if table:
    #         return temp
    #     elif file_name is None:
    #         show(p)
    #     else:
    #         export_png(p, filename=file_name)

    # def head_to_head(self, challenger, data=None, table=False):
    #     """Two way lift chart that is sorted by difference between Predicted
    #     scores.  Still bucketed to 10 levels with the same approximate weight
    #     """
    #     if data is None:
    #         data1 = self.transformed_data
    #         data2 = challenger.predict(self.data)
    #         data2["Fitted Avg"] = data2["Fitted Avg"] * data2[self.weight]
    #     else:
    #         data1 = self.predict(data)
    #         data1["Fitted Avg"] = data1["Fitted Avg"] * data1[self.weight]
    #         data2 = challenger.predict(data)
    #         data2["Fitted Avg"] = data2["Fitted Avg"] * data2[self.weight]
    #     temp = data1[[self.weight, self.dependent, "Fitted Avg"]]
    #     data2["Fitted Avg Challenger"] = data2["Fitted Avg"]
    #     data2 = data2[["Fitted Avg Challenger"]]
    #     temp = copy.deepcopy(temp)
    #     temp = temp.merge(data2, how="inner", left_index=True, right_index=True)

    #     temp["sort"] = temp["Fitted Avg"] / temp["Fitted Avg Challenger"]
    #     temp = temp.sort_values("sort")
    #     temp["decile"] = (
    #         temp[self.weight].cumsum() / ((sum(temp[self.weight]) * 1.00001) / 10) + 1
    #     ).apply(np.floor)
    #     temp = pd.pivot_table(
    #         data=temp,
    #         index=["decile"],
    #         values=[self.dependent, self.weight, "Fitted Avg", "Fitted Avg Challenger"],
    #         aggfunc="sum",
    #     )
    #     temp["Observed"] = temp[self.dependent] / temp[self.weight]
    #     temp["Fitted1"] = temp["Fitted Avg"] / temp[self.weight]
    #     temp["Fitted2"] = temp["Fitted Avg Challenger"] / temp[self.weight]
    #     y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
    #     hover = HoverTool(
    #         tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
    #     )  # 'vline')
    #     p = figure(
    #         plot_width=700,
    #         plot_height=400,
    #         y_range=y_range,
    #         title="Head to Head",
    #         toolbar_sticky=False,
    #     )  # , x_range=list(temp.index)
    #     p.add_tools(hover)
    #     h = np.array(temp[self.weight])
    #     # Correcting the bottom position of the bars to be on the 0 line.
    #     adj_h = h / 2
    #     # add bar renderer
    #     p.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")
    #     # add line to secondondary axis
    #     p.extra_y_ranges = {
    #         "foo": Range1d(
    #             start=min(
    #                 temp["Observed"].min(), temp["Fitted1"].min(), temp["Fitted2"].min()
    #             )
    #             / 1.1,
    #             end=max(
    #                 temp["Observed"].max(), temp["Fitted1"].max(), temp["Fitted2"].max()
    #             )
    #             * 1.1,
    #         )
    #     }
    #     p.add_layout(LinearAxis(y_range_name="foo"), "right")
    #     # Observed Average line values
    #     p.line(
    #         temp.index,
    #         temp["Observed"],
    #         line_width=2,
    #         color="#ff69b4",
    #         y_range_name="foo",
    #     )
    #     p.line(
    #         temp.index,
    #         temp["Fitted1"],
    #         line_width=2,
    #         color="#006400",
    #         y_range_name="foo",
    #     )
    #     p.line(
    #         temp.index,
    #         temp["Fitted2"],
    #         line_width=2,
    #         color="#146195",
    #         y_range_name="foo",
    #     )
    #     p.legend.location = "top_left"
    #     if not table:
    #         show(p)
    #     else:
    #         return temp

    # def two_way(self, x1, x2, pdp=False, table=False, file_name=None):
    #     """ Two way (two features from independent list) view of data

    #     """
    #     data = self.transformed_data
    #     a = (
    #         pd.pivot_table(
    #             data,
    #             index=x1,
    #             columns=x2,
    #             values=[self.weight, self.dependent, "Fitted Avg"],
    #             aggfunc="sum",
    #         )
    #         .fillna(0)
    #         .reset_index()
    #     )
    #     # print(a.head())
    #     response_list = [
    #         self.dependent + " " + str(item).strip() for item in (data[x2].unique())
    #     ]
    #     fitted_list = [
    #         "Fitted Avg " + str(item).strip() for item in (data[x2].unique())
    #     ]
    #     a.columns = [
    #         " ".join([str(i) for i in col]).strip() for col in a.columns.values
    #     ]
    #     a = a.fillna(0)
    #     a[x1] = a[x1].astype(str)
    #     weight_list = [
    #         self.weight + " " + str(item).strip() for item in data[x2].unique()
    #     ]
    #     source = ColumnDataSource(a)
    #     hover = HoverTool(
    #         tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
    #     )  # 'vline')
    #     p = figure(
    #         plot_width=800,
    #         x_range=list(a[x1]),
    #         toolbar_location="right",
    #         toolbar_sticky=False,
    #     )
    #     p.add_tools(hover)
    #     p.vbar_stack(
    #         stackers=weight_list,
    #         x=x1,
    #         source=source,
    #         width=0.9,
    #         alpha=[0.5] * len(weight_list),
    #         color=(Spectral9 * 100)[: len(weight_list)],
    #         legend=[str(item) for item in list(data[x2].unique())],
    #     )
    #     p.y_range = Range1d(0, max(np.sum(a[weight_list], axis=1)) * 1.8)
    #     p.xaxis[0].axis_label = x1
    #     p.xgrid.grid_line_color = None
    #     p.outline_line_color = None
    #     outcome = pd.DataFrame(
    #         np.divide(
    #             np.array(a[response_list]),
    #             np.array(a[weight_list]),
    #             where=np.array(a[weight_list]) > 0,
    #         ),
    #         columns=["Outcome " + str(item) for item in list(data[x2].unique())],
    #     )  # add line to secondondary axis
    #     fitted = pd.DataFrame(
    #         np.divide(
    #             np.array(a[fitted_list]),
    #             np.array(a[weight_list]),
    #             where=np.array(a[weight_list]) > 0,
    #         ),
    #         columns=["Fitted Avg " + str(item) for item in list(data[x2].unique())],
    #     )  # add line to secondondary axis

    #     p.xaxis[0].axis_label = x1
    #     p.yaxis[0].axis_label = self.weight
    #     p.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
    #     p.add_layout(
    #         LinearAxis(
    #             y_range_name="foo", axis_label=self.dependent + "/" + self.weight
    #         ),
    #         "right",
    #     )
    #     p.add_layout(
    #         Title(text=x1 + " vs " + x2, text_font_size="12pt", align="left"), "above"
    #     )
    #     if not pdp:
    #         p.extra_y_ranges = {
    #             "foo": Range1d(
    #                 start=np.min(np.array(outcome)) / 1.1,
    #                 end=np.max(np.array(outcome)) * 1.1,
    #             )
    #         }
    #         for i in range(len(outcome.columns)):
    #             p.line(
    #                 x=a[x1],
    #                 y=outcome.iloc[:, i],
    #                 line_width=3,
    #                 color=(Spectral9 * 100)[i],
    #                 line_cap="round",
    #                 line_alpha=0.9,
    #                 y_range_name="foo",
    #             )
    #         for i in range(len(fitted.columns)):
    #             p.line(
    #                 x=a[x1],
    #                 y=fitted.iloc[:, i],
    #                 line_width=3,
    #                 color=(Spectral9 * 100)[i],
    #                 line_cap="round",
    #                 line_dash="dashed",
    #                 line_alpha=1,
    #                 y_range_name="foo",
    #             )
    #     if pdp:
    #         pdp = np.transpose(
    #             [
    #                 np.tile(self.PDP[x1].index, len(self.PDP[x2].index)),
    #                 np.repeat(self.PDP[x2].index, len(self.PDP[x1].index)),
    #             ]
    #         )
    #         pdp = pd.DataFrame(pdp, columns=[x1, x2])
    #         pdp = (
    #             (
    #                 (self.PDP[x1].reset_index().drop(x2, axis=1)).merge(
    #                     pdp, how="inner", left_on=x1, right_on=x1
    #                 )
    #             )[self.independent]
    #         ).to_clipboard()
    #         x = self.predict(pdp)
    #         x = pd.pivot_table(
    #             x, index=[x1], columns=[x2], values=["Fitted Avg"], aggfunc="mean"
    #         )
    #         p.extra_y_ranges = {
    #             "foo": Range1d(
    #                 start=np.min(np.array(x)) / 1.1, end=np.max(np.array(x)) * 1.1
    #             )
    #         }
    #         for i in range(len(x.columns)):
    #             p.line(
    #                 x=a[x1],
    #                 y=x.iloc[:, i],
    #                 line_width=3,
    #                 color=(Spectral9 * 100)[i],
    #                 line_cap="round",
    #                 line_alpha=1,
    #                 y_range_name="foo",
    #             )
    #     p.xaxis.major_label_orientation = math.pi / 4
    #     if table:
    #         return a
    #     elif file_name is None:
    #         show(p)
    #     else:
    #         export_png(p, filename=file_name)

    # def create_comparisons(
    #     self, columns, title="", obs=True, fitted=True, model=True, ci=True, ret=False
    # ):
    #     """This function needs to be documented..."""

    #     def view_one_way(transform, column, title, obs, fitted, model, ci):
    #         data = self.transformed_data[
    #             [self.dependent, self.weight, "Fitted Avg", column]
    #         ]
    #         temp = pd.pivot_table(
    #             data=data,
    #             index=[column],
    #             values=[self.dependent, self.weight, "Fitted Avg"],
    #             aggfunc=np.sum,
    #         )
    #         temp["Observed"] = temp[self.dependent] / temp[self.weight]
    #         temp["Fitted"] = temp["Fitted Avg"] / temp[self.weight]
    #         temp = temp.merge(
    #             self.PDP[column][["Model", "CI_U", "CI_L"]],
    #             how="inner",
    #             left_index=True,
    #             right_index=True,
    #         )
    #         if transform == "Predicted Value":
    #             for item in ["Model", "CI_U", "CI_L"]:
    #                 temp[item] = self._link_transform(temp[item], "predicted value")
    #         else:
    #             for item in ["Observed", "Fitted"]:
    #                 temp[item] = self._link_transform(temp[item], "linear predictor")

    #         y_range = Range1d(start=0, end=temp[self.weight].max() * 1.8)
    #         hover = HoverTool(
    #             tooltips=[("(x,y)", "($x{0.00 a}, $y{0.00 a})")], mode="mouse"
    #         )  # 'vline')
    #         if type(temp.index) == pd.core.indexes.base.Index:  # Needed for categorical
    #             f = figure(
    #                 plot_width=800,
    #                 y_range=y_range,
    #                 x_range=list(temp.index),
    #                 toolbar_location="right",
    #                 toolbar_sticky=False,
    #             )
    #         else:
    #             f = figure(
    #                 plot_width=800,
    #                 y_range=y_range,
    #                 toolbar_location="right",
    #                 toolbar_sticky=False,
    #             )

    #         # setting bar values
    #         f.add_tools(hover)
    #         f.add_layout(
    #             Title(text=title + column, text_font_size="12pt", align="center"),
    #             "above",
    #         )
    #         f.yaxis[0].axis_label = self.weight
    #         f.yaxis[0].formatter = NumeralTickFormatter(format="0.00 a")
    #         f.add_layout(
    #             LinearAxis(
    #                 y_range_name="foo", axis_label=self.dependent + "/" + self.weight
    #             ),
    #             "right",
    #         )
    #         h = np.array(temp[self.weight])

    #         # Correcting the bottom position of the bars to be on the 0 line.
    #         adj_h = h / 2

    #         # add bar renderer
    #         f.rect(x=temp.index, y=adj_h, width=0.4, height=h, color="#e5e500")

    #         # add line to secondondary axis
    #         f.extra_y_ranges = {
    #             "foo": Range1d(
    #                 start=min(temp["Observed"].min(), temp["Model"].min()) / 1.1,
    #                 end=max(temp["Observed"].max(), temp["Model"].max()) * 1.1,
    #             )
    #         }

    #         # Observed Average line values
    #         if obs:
    #             f.line(
    #                 temp.index,
    #                 temp["Observed"],
    #                 line_width=2,
    #                 color="#ff69b4",
    #                 y_range_name="foo",
    #             )
    #         if fitted:
    #             f.line(
    #                 temp.index,
    #                 temp["Fitted"],
    #                 line_width=2,
    #                 color="#006400",
    #                 y_range_name="foo",
    #             )
    #         if model:
    #             f.line(
    #                 temp.index,
    #                 temp["Model"],
    #                 line_width=2,
    #                 color="#00FF00",
    #                 y_range_name="foo",
    #             )
    #         if ci:
    #             f.line(
    #                 temp.index,
    #                 temp["CI_U"],
    #                 line_width=2,
    #                 color="#db4437",
    #                 y_range_name="foo",
    #             )
    #             f.line(
    #                 temp.index,
    #                 temp["CI_L"],
    #                 line_width=2,
    #                 color="#db4437",
    #                 y_range_name="foo",
    #             )
    #         f.xaxis.major_label_orientation = math.pi / 4
    #         return f

    #     transform = "Predicted Value"
    #     title = title + " | " if title != "" else title
    #     columns = columns if isinstance(columns, list) else [columns]
    #     comparisons = []
    #     for column in columns:
    #         self.comparisons.append(
    #             view_one_way(
    #                 transform=transform,
    #                 column=column,
    #                 title=title,
    #                 obs=obs,
    #                 fitted=fitted,
    #                 model=model,
    #                 ci=ci,
    #             )
    #         )
    #         comparisons.append(self.comparisons[-1])
    #     if ret:
    #         return comparisons

    # def view_comparisons(self, file_name=None, ncols=2, reorder=[]):
    #     """This function needs to be documented..."""

    #     def reorder_comparisons(order):
    #         if (
    #             max([x for x in order if x is not None]) + 1 != len(self.comparisons)
    #             or min([x for x in order if x is not None]) != 0
    #         ):
    #             error = (
    #                 """ Error, unable to reorder list because the count of reorder is not equal to the number of comparisons.
    #                         Use get_comparisons_count() then reorder the list that way. For instance: for a comparison count of 3
    #                         you may wish to do this reorder_comparisons([0,2,1]). in this case you need it to go from 0 to """
    #                 + str(len(self.comparisons) - 1)
    #                 + ". you can also add None into the list to make a blank space."
    #             )
    #             return error
    #         comparisons = []
    #         for item in order:
    #             if item is None:
    #                 comparisons.append(None)
    #             else:
    #                 comparisons.append(self.comparisons[item])
    #         return comparisons

    #     if file_name:
    #         if file_name[-4:] in [".png", ".PNG"]:
    #             export_png(self.comparisons, filename=file_name)
    #             return
    #         else:
    #             output_file(file_name + ".html")

    #     if len(self.comparisons) > 0:
    #         p = (
    #             gridplot(self.comparisons, ncols=ncols)
    #             if reorder == []
    #             else gridplot(reorder_comparisons(reorder), ncols=ncols)
    #         )
    #         # bad way of labeling the columns: can't figure out rows anway
    #         # toggle1 = Toggle(label=category_one, width=800)
    #         # toggle2 = Toggle(label=category_two, width=800)
    #         # show the results
    #         # show(layout([toggle2, toggle1], [p]))
    #         show(p)
    #     else:
    #         print(
    #             "You must create_comparisons(colums) first, before you can view_comparisons(file_name,ncols). Then you can clear_comparisons()."
    #         )

    # def clear_comparisons(self):
    #     """This function needs to be documented..."""
    #     self.comparisons = []

    # def get_comparisons(self):
    #     """This function needs to be documented..."""
    #     return self.comparisons

    # def get_comparisons_count(self):
    #     """This function needs to be documented..."""
    #     return len(self.comparisons)

    # def give_comparisons(self, comparison_list):
    #     """This function needs to be documented..."""
    #     for each_comparison in comparison_list:
    #         self.comparisons.append(each_comparison)

    # def static_view(
    #     self,
    #     column,
    #     title,
    #     obs=True,
    #     fitted=True,
    #     model=True,
    #     ci=False,
    #     ret=True,
    #     file_name=None,
    # ):
    #     """ This method is built off of the comparison methods.  The purpose is to
    #     convert the dynamic view() method into a programmatic expression for quick file
    #     output."""
    #     temp = list(self.comparisons)
    #     self.clear_comparisons()
    #     """comparisons = self.create_comparisons(
    #         [column], title=title, obs=obs, fitted=fitted, model=model, ci=ci, ret=ret
    #     )"""
    #     self.view_comparisons(file_name)
    #     self.comparisons = list(temp)

    # def create_lift(self, data=None, title="", ret=False):
    #     """This function needs to be documented..."""
    #     if data is not None:
    #         data = data.reset_index()
    #     lift = self.lift_chart(data=None, title=title, dont_show_give=True)
    #     self.lifts.append(lift)
    #     if ret:
    #         return [lift]

    # def view_lifts(self, file_name=None, ncols=2, reorder=[]):
    #     """This function needs to be documented..."""

    #     def reorder_lifts(order):
    #         if (
    #             max([x for x in order if x is not None]) + 1 != len(self.lifts)
    #             or min([x for x in order if x is not None]) != 0
    #         ):
    #             error = (
    #                 """ Error, unable to reorder list because the count of reorder is not equal to the number of comparisons.
    #                         Use get_comparisons_count() then reorder the list that way. For instance: for a comparison count of 3
    #                         you may wish to do this reorder_comparisons([0,2,1]). in this case you need it to go from 0 to """
    #                 + str(len(self.lifts) - 1)
    #                 + ". you can also add None into the list to make a blank space."
    #             )
    #             return error
    #         lifts = []
    #         for item in order:
    #             if item is None:
    #                 lifts.append(None)
    #             else:
    #                 lifts.append(self.lifts[item])
    #         return lifts

    #     if file_name:
    #         output_file(file_name + ".html")

    #     if len(self.lifts) > 0:
    #         p = (
    #             gridplot(self.lifts, ncols=ncols)
    #             if reorder == []
    #             else gridplot(reorder_lifts(reorder), ncols=ncols)
    #         )
    #         # bad way of labeling the columns: can't figure out rows anway
    #         # toggle1 = Toggle(label=category_one, width=800)
    #         # toggle2 = Toggle(label=category_two, width=800)
    #         # show the results
    #         # show(layout([toggle2, toggle1], [p]))
    #         show(p)
    #     else:
    #         print(
    #             "You must create_lift() first, before you can view_lifts(file_name,ncols). Then you can clear_lifts()."
    #         )

    # def clear_lifts(self):
    #     """This function needs to be documented..."""
    #     self.lifts = []

    # def get_lifts(self):
    #     """This function needs to be documented..."""
    #     return self.lifts

    # def get_lifts_count(self):
    #     """This function needs to be documented..."""
    #     return len(self.lifts)

    # def give_lifts(self, lift_list):
    #     """This function needs to be documented..."""
    #     for each_lift in lift_list:
    #         self.lifts.append(each_lift)
