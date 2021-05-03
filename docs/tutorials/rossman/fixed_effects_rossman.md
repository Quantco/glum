---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# High Dimensional Fixed Effects with Rossman Sales Data

**Intro**

This tutorial demonstrates how you can create models with high dimensional fixed effects using `quantcore.glm`. Thanks to the utilization of `quantcore.matrix`, we can pass categorical variables with a large range of values and the rest is taken care of for us. `quantcore.glm` and `quantcore.matrix` will handle the creation of the one-hot-encoded design matrix and also take advantage of the sparse nature of the matrix to optimize operations.


**Background**

For this tutorial, we will be predicting sales for the European drug store chain. Specifically, we are tasked with predicting daily sales for future dates. Ideally, we want a model that can capture the many factors that influence stores sales -- promotions, competition, school, holidays, seasonality, etc. As a baseline, we will start with a simple model that only uses a few basic predictors. Then, we will fit a model with a large number of fixed effects. For both models, we will use OLS with L2 regularization. 

*Note*: a few parts of this tutorial utilize local helper functions outside this notebook. If you wish to run the notebook on your own, you can find the rest of the code here: <span style="color:red">**TODO**: add link once in master</span>.

## Table of Contents <a class="anchor" id="toc"></a>
* [1. Data Loading and Feature Engineering](#1-load)
* [2. Fit Baseline GLM Model](#2-baseline)
* [3. GLM with High Dimensional Fixed Effects](#3-highdim)
* [4. Plot Results](#4-plot)
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
from quantcore.glm import GeneralizedLinearRegressor
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

## 1. Data Loading and Feature Engineering<a class="anchor" id="1-load"></a>
[back to table of contents](#toc)

We start by loading in the raw data. If you have not yet processed the raw data, it will be done below. (Initial processing consists of some basic cleaning and renaming of columns.

*Note*: if you wish to run this notebook on your own, and have not done so already, please download the data from the [Rossman Kaggle Challenge](https://www.kaggle.com/c/rossmann-store-sales). This tutorial expects that it in a folder names "raw_data" under the same directory as the notebook.


### 1.1 Load

```python
if not all([Path(p).exists for p in ["raw_data/train.csv", "raw_data/test.csv", "raw_data/store.csv"]]):
    raise Exception("Please download raw data into 'raw_data' folder")

if not all([Path(p).exists for p in ["processed_data/train.parquet", "processed_data/test.parquet"]]):
    "Processed data not found. Processing data from raw data..."
    process_data()
    "Done"
    
df = load_train().sort_values(["store", "date"])
df = df.iloc[:int(.1*len(df))]
df.head()
```

### 1.2 Feature Engineering

As we mention earlier, we want our model to incorporate the many factors that influence store sales. Thus, we create a number of fixed effects to capture this information. These include:

- Fixed effects for a certain number days before a school or state holidays
- Fixed effects for a certain number days after a school or state holidays
- Fixed effects for a certain number days before a promo
- Fixed effects for a certain number days after a promo
- Fixed effects for a certain number days before the store is open or closed
- Fixed effects for a certain number days after the store is open or closed
- Fixed effects for each month for each store
- Fixed effects for each year for each store
- Fixed effects for each day of the week for each store

In addition to fixed effects, we also do several other transformations. These include:

- Taking the log of sales. It is expected that the factors influencing store sales like locality, seasonality, etc. will have a multiplicative effect on sales rather than additive, so going foward, we use log of sales as our outcome variable
- Computing the z score to eliminate outliers (in the next step)

```python
df = apply_all_transformations(df)
df.head()
```

### 1.3 Train vs. Validation selection

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

## 2. Fit Baseline GLM Model<a class="anchor" id="2-baseline"></a>
[back to table of contents](#toc)

We start with a simple model that uses only year, month, day of the week, and store as predictors. Even with these variables alone, we should still be able to capture a lot of valuable information. Year can capture overall sales trends, month can capture seasonality, week day can capture the variation in sales across the week, and store can capture locality. We will treat these all as categorical variables. 

*Note*: notice how with the `GeneralizedLinearRegressor()` class, we pass in the categorical variables directly without having to encode them ourselves. All we have to do is use the dask ml `Categorizer` to transform the columns into categorical data types. You can reference the [pandas documentation on Categoricals](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) to learn more about how these data types work.

```python
baseline_features = ["year", "month", "day_of_week", "store"]
baseline_categorizer = Categorizer(columns=baseline_features)
baseline_glm = GeneralizedLinearRegressor(
    family="normal",
    scale_predictors=True,
    l1_ratio=0.0,
    alphas=1e-1,
)
```

We fit our model (making sure to process the dataframe with the categorizer first) and inspect the coefficients. Even with the few categoricals we used, we can see that we still get a lot of fixed effects. 

```python
baseline_glm.fit(
    baseline_categorizer.fit_transform(df[select_train][baseline_features]),
    df.loc[select_train, "log_sales"]
)

pd.DataFrame(
    {'coefficient': np.concatenate(([baseline_glm.intercept_], baseline_glm.coef_))},
    index=['intercept'] + baseline_glm.feature_names_
).T
```

Note that below, we predict sales for when the stores are open vs. closed separately (as one would expect, sales for days when the stores are closed are 0).

```python
df.loc[lambda x: x["open"], "predicted_log_sales_baseline"] = baseline_glm.predict(
    baseline_categorizer.fit_transform(df.loc[lambda x: x["open"]][baseline_features])
)

df["predicted_log_sales_baseline"] = df["predicted_log_sales_baseline"].fillna(0)
df["predicted_sales_baseline"] = np.exp(df["predicted_log_sales_baseline"])
```

We use root mean squared percentage error (RMPSE) as our performance metric. (Useful for thinking about error relative to total sales of each store).  

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


## 3. GLM with High Dimensional Fixed Effects<a class="anchor" id="3-highdim"></a>
[back to table of contents](#toc)

Now, we repeat a similar process to above, but this time, we take advantage of the full range of categoricals we created in our data transformation step. Since we will create a very large number of fixed effects, we may run into cases where our validation data has categorical values not seen in our training data. In these cases, the dask ml categorizer will output null values when transforming the validation columns to the categoricals that were created on the training set. To fix this, we add the dask ml [SimpleImputer](https://ml.dask.org/modules/generated/dask_ml.impute.SimpleImputer.html) to our pipeline. 

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
    family="normal",
    scale_predictors=True,
    l1_ratio=0.0, # only ridge
    alpha=1e-1,
)
```

For reference, we output the total number of predictors after fitting the model. We can see that the number is rather large, so this time, we don't output our feature names and their coefficient values.

```python
highdim_glm.fit(
    highdim_categorizer.fit_transform(df[select_train][highdim_features]),
    df.loc[select_train, "log_sales"]
)

print(f"Number of predictors: {len(highdim_glm.feature_names_)}")
```

```python
df.loc[lambda x: x["open"], "predicted_log_sales_highdim"] = highdim_glm.predict(
    highdim_categorizer.transform(df.loc[lambda x: x["open"]][highdim_features]),
)

df["predicted_log_sales_highdim"] = df["predicted_log_sales_highdim"].fillna(0)
df["predicted_sales_highdim"] = np.exp(df["predicted_log_sales_highdim"])


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


## 4. Plot Results<a class="anchor" id="4-plot"></a>
[back to table of contents](#toc)

Finally, to get a better look at our results, we make some plots.

```python
sales_cols = ["sales", "predicted_sales_highdim", "predicted_sales_baseline"]
```

First, we plot true sales and the sales predictions from each model aggregated over month

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

```python

```
