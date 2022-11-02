---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# High Dimensional Fixed Effects with Rossman Sales Data

**Intro**

This tutorial demonstrates how to create models with high dimensional fixed effects using `glum`. Using `tabmat`, we can pass categorical variables with a large range of values. `glum` and `tabmat` will handle the creation of the one-hot-encoded design matrix.

In some real-world problems, we have used millions of categories. This would be impossible with a dense matrix. General-purpose sparse matrices like compressed sparse row (CSR) matrices help but still leave a lot on the table. For a categorical matrix, we know that each row has only a single non-zero value and that value is 1. These optimizations are implemented in `tabmat.CategoricalMatrix`.


**Background**

For this tutorial, we will be predicting sales for the European drug store chain Rossman. Specifically, we are tasked with predicting daily sales for future dates. Ideally, we want a model that can capture the many factors that influence stores sales -- promotions, competition, school, holidays, seasonality, etc. As a baseline, we will start with a simple model that only uses a few basic predictors. Then, we will fit a model with a large number of fixed effects. For both models, we will use OLS with L2 regularization.

We will use a gamma distribution for our model. This choice is motivated by two main factors. First, our target variable, sales, is a positive real number, which matches the support of the gamma distribution. Second, it is expected that factors influencing sales are multiplicative rather than additive, which is better captured with a gamma regression than say, OLS.


*Note*: a few parts of this tutorial utilize local helper functions outside this notebook. If you wish to run the notebook on your own, you can find the rest of the code [here](https://github.com/Quantco/glum/tree/open-sourcing/docs/tutorials/rossman).

## Table of Contents<a class="anchor"></a>
* [1. Data Loading and Feature Engineering](#1.-Data-Loading-and-Feature-Engineering)
* [2. Fit Baseline GLM](#2.-Fit-baseline-GLM)
* [3. GLM with High Dimensional Fixed Effects](#3.-GLM-with-High-Dimensional-Fixed-Effects)
* [4. Plot Results](#4.-Plot-Results)
<!-- #endregion -->

```python
import os
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor
from sklearn.pipeline import Pipeline

from feature_engineering import apply_all_transformations
from process_data import load_test, load_train, process_data

import sys
sys.path.append("../")
from metrics import root_mean_squared_percentage_error

pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option('display.max_columns', None)
alt.data_transformers.enable("json")  # to allow for large plots
```

## 1. Data loading and feature engineering<a class="anchor"></a>
[back to table of contents](#Table-of-Contents)

We start by loading in the raw data. If you have not yet processed the raw data, it will be done below. (Initial processing consists of some basic cleaning and renaming of columns.

*Note*: if you wish to run this notebook on your own, and have not done so already, please download the data from the [Rossman Kaggle Challenge](https://www.kaggle.com/c/rossmann-store-sales). This tutorial expects that it in a folder named "raw_data" under the same directory as the notebook.


### 1.1 Load

```python
if not all(Path(p).exists() for p in ["raw_data/train.csv", "raw_data/test.csv", "raw_data/store.csv"]):
    raise Exception("Please download raw data into 'raw_data' folder")

if not all(Path(p).exists() for p in ["processed_data/train.parquet", "processed_data/test.parquet"]):
    "Processed data not found. Processing data from raw data..."
    process_data()
    "Done"

df = load_train().sort_values(["store", "date"])
df = df.iloc[:int(.1*len(df))]
df.head()
```

### 1.2 Feature engineering

As mentioned earlier, we want our model to incorporate many factors that could influence store sales. We create a number of fixed effects to capture this information. These include fixed effects for:

- A certain number days before a school or state holiday
- A certain number days after a school or state holiday
- A certain number days before a promo
- A certain number days after a promo
- A certain number days before the store is open or closed
- A certain number days after the store is open or closed
- Each month for each store
- Each year for each store
- Each day of the week for each store

We also do several other transformations like computing the z score to eliminate outliers (in the next step)

```python
df = apply_all_transformations(df)
df.head()
```

### 1.3 Train vs. validation selection

Lastly, we split our data into training and validation sets. Kaggle provides a test set for the Rossman challenge, but it does not directly include outcome data (sales), so we do not use it for our tutorial. Instead, we simulate predicting future sales by taking the last 5 months of our training data as our validation set.

```python
validation_window = [pd.to_datetime("2015-03-15"), pd.to_datetime("2015-07-31")]
select_train = (df["sales"].gt(0) & df["date"].lt(validation_window[0]) & df["zscore"].abs().lt(5)).to_numpy()

select_val = (
    df["sales"].gt(0)
    & df["date"].ge(validation_window[0])
    & df["date"].lt(validation_window[1])
).to_numpy()


(select_train.sum(), select_val.sum())
```

## 2. Fit baseline GLM<a class="anchor"></a>
[back to table of contents](#Table-of-Contents)

We start with a simple model that uses only year, month, day of the week, and store as predictors. Even with these variables alone, we should still be able to capture a lot of valuable information. Year can capture overall sales trends, month can capture seasonality, week day can capture the variation in sales across the week, and store can capture locality. We will treat these all as categorical variables.

With the `GeneralizedLinearRegressor()` class, we can pass in `pandas.Categorical` variables directly without having to encode them ourselves. This is convenient, especially when we start adding more fixed effects. But it is very important that the categories are aligned between calls to `fit` and `predict`. One way of achieving this alignment is with a `dask_ml.preprocessing.Categorizer`. Note, however, that the `Categorizer` class fails to enforce category alignment if the input column is already a categorical data type.

You can reference the [pandas documentation on Categoricals](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) to learn more about how these data types work.

```python
baseline_features = ["year", "month", "day_of_week", "store"]
baseline_categorizer = Categorizer(columns=baseline_features)
baseline_glm = GeneralizedLinearRegressor(
    family="gamma",
    scale_predictors=True,
    l1_ratio=0.0,
    alphas=1e-1,
)
```

Fit the model making sure to process the data frame with the Categorizer first and inspect the coefficients.

```python
baseline_glm.fit(
    baseline_categorizer.fit_transform(df[select_train][baseline_features]),
    df.loc[select_train, "sales"]
)

pd.DataFrame(
    {'coefficient': np.concatenate(([baseline_glm.intercept_], baseline_glm.coef_))},
    index=['intercept'] + baseline_glm.feature_names_
).T
```

And let's predict for our test set with the caveat that we will predict 0 for days when the stores are closed!

```python
df.loc[lambda x: x["open"], "predicted_sales_baseline"] = baseline_glm.predict(
    baseline_categorizer.fit_transform(df.loc[lambda x: x["open"]][baseline_features])
)

df["predicted_sales_baseline"] = df["predicted_sales_baseline"].fillna(0)
df["predicted_sales_baseline"] = df["predicted_sales_baseline"]
```

We use root mean squared percentage error (RMSPE) as our performance metric. (Useful for thinking about error relative to total sales of each store).

```python
train_err = root_mean_squared_percentage_error(
    df.loc[select_train, "sales"], df.loc[select_train, "predicted_sales_baseline"]
)
val_err = root_mean_squared_percentage_error(
    df.loc[select_val, "sales"], df.loc[select_val, "predicted_sales_baseline"]
)
print(f'Training Error: {round(train_err, 2)}%')
print(f'Validation Error: {round(val_err, 2)}%')
```

The results aren't bad for a start, but we can do better :)


## 3. GLM with high dimensional fixed effects<a class="anchor"></a>
[back to table of contents](#Table-of-Contents)

Now, we repeat a similar process to above, but, this time, we take advantage of the full range of categoricals we created in our data transformation step. Since we will create a very large number of fixed effects, we may run into cases where our validation data has categorical values not seen in our training data. In these cases, Dask-ML's `Categorizer` will output null values when transforming the validation columns to the categoricals that were created on the training set. To fix this, we add Dask-ML's  [SimpleImputer](https://ml.dask.org/modules/generated/dask_ml.impute.SimpleImputer.html) to our pipeline.

```python
highdim_features = [
    "age_quantile",
    "competition_open",
    "open_lag_1",
    "open_lag_2",
    "open_lag_3",
    "open_lead_1",
    "open_lead_2",
    "open_lead_3",
    "promo_lag_1",
    "promo_lag_2",
    "promo_lag_3",
    "promo_lead_1",
    "promo_lead_2",
    "promo_lead_3",
    "promo",
    "school_holiday_lag_1",
    "school_holiday_lag_2",
    "school_holiday_lag_3",
    "school_holiday_lead_1",
    "school_holiday_lead_2",
    "school_holiday_lead_3",
    "school_holiday",
    "state_holiday_lag_1",
    "state_holiday_lag_2",
    "state_holiday_lag_3",
    "state_holiday_lead_1",
    "state_holiday_lead_2",
    "state_holiday_lead_3",
    "state_holiday",
    "store_day_of_week",
    "store_month",
    "store_school_holiday",
    "store_state_holiday",
    "store_year",
]
highdim_categorizer = Pipeline([
    ("categorize", Categorizer(columns=highdim_features)),
    ("impute", SimpleImputer(strategy="most_frequent"))
])
highdim_glm = GeneralizedLinearRegressor(
    family="gamma",
    scale_predictors=True,
    l1_ratio=0.0, # only ridge
    alpha=1e-1,
)
```

For reference, we output the total number of predictors after fitting the model. We can see that the number getting a bit larger, so we don't print out the coefficients this time.

```python
highdim_glm.fit(
    highdim_categorizer.fit_transform(df[select_train][highdim_features]),
    df.loc[select_train, "sales"]
)

print(f"Number of predictors: {len(highdim_glm.feature_names_)}")
```

```python
df.loc[lambda x: x["open"], "predicted_sales_highdim"] = highdim_glm.predict(
    highdim_categorizer.transform(df.loc[lambda x: x["open"]][highdim_features]),
)

df["predicted_sales_highdim"] = df["predicted_sales_highdim"].fillna(0)
df["predicted_sales_highdim"] = df["predicted_sales_highdim"]


train_err = root_mean_squared_percentage_error(
    df.loc[select_train, "sales"], df.loc[select_train, "predicted_sales_highdim"]
)
val_err = root_mean_squared_percentage_error(
    df.loc[select_val, "sales"], df.loc[select_val, "predicted_sales_highdim"]
)
print(f'Training Error: {round(train_err, 2)}%')
print(f'Validation Error: {round(val_err, 2)}%')
```

From just the RMSPE, we can see a clear improvement from our baseline model.


## 4. Plot results<a class="anchor"></a>
[back to table of contents](#Table-of-Contents)

Finally, to get a better look at our results, we make some plots.

```python
sales_cols = ["sales", "predicted_sales_highdim", "predicted_sales_baseline"]
```

First, we plot true sales and the sales predictions from each model aggregated over month:

```python
_, axs = plt.subplots(2, 1, figsize=(16, 16))

for i, select in enumerate([select_train, select_val]):
    ax = axs[i]
    df_plot_date = df[select].groupby(
        ["year", "month"]
    ).agg("sum")[sales_cols].reset_index()

    year_month_date = df_plot_date['month'].map(str)+ '-' + df_plot_date['year'].map(str)
    df_plot_date['year_month'] = pd.to_datetime(year_month_date, format='%m-%Y').dt.strftime('%m-%Y')
    df_plot_date.drop(columns= ["year", "month"], inplace=True)

    df_plot_date.plot(x="year_month", ax=ax)
    ax.set_xticks(range(len(df_plot_date)))
    ax.set_xticklabels(df_plot_date.year_month, rotation=45)
    ax.set_xlabel("date")
    ax.set_ylabel("Total sales")
    ax.grid(True, linestyle='-.')

axs[0].set_title("Training Results: Total Sales per Month")
axs[1].set_title("Validation Results: Total Sales per Month")
plt.show()
```

We can also look at aggregate sales for a subset of stores. We select the first 20 stores and plot in order of increasing sales.

```python
_, axs = plt.subplots(2, 1, figsize=(14, 12))
for i, select in enumerate([select_train, select_val]):
    ax = axs[i]
    df_plot_store = df[select].groupby(
        ["store"]
    ).agg("sum")[sales_cols].reset_index()[:20].sort_values(by="sales")

    df_plot_store.plot.bar(x="store", ax=ax)
    ax.set_xlabel("Store")
    ax.set_ylabel("Total sales")

axs[0].set_title("Training Results: Total Sales by Store")
axs[1].set_title("Validation Results: Total Sales by Store")
plt.show()
```

We can see that the high dimensional model is much better at capturing the variation between months and individual stores!
