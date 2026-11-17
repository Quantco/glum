# Formula Interface Tutorial: Revisiting French Motor Third-Party Liability Claims

**Intro**

This tutorial showcases the formula interface of `glum`. It allows for the specification of the design matrix and the response variable using so-called [Wilkinson-formulas](https://www.jstor.org/stable/2346786) instead of constructing it by hand. This kind of model specification should be familiar to R users or those who have used the `statsmodels` or `linearmodels` Python packages before. This tutorial aims to introduce the basics of working with formulas to other users, as well as highlighting some important differences between `glum`s and other packages' formula implementations.

For a more in-depth look at how formulas work, please take a look at the [documentation of `formulaic`](https://matthewwardrop.github.io/formulaic/), the package on which `glum`'s formula interface is based.

**Background**

This tutorial reimplements and extends the combined frequency-severity model from Chapter 4 of the [GLM tutorial](../glm_french_motor_tutorial/glm_french_motor.md). If you would like to know more about the setting, the data, or GLM modeling in general, please check that out first.

**Sneak Peek**

Formulas can provide a concise and convenient way to specify many of the usual pre-processing steps, such as converting to categorical types, creating interactions, applying transformations, or even spline interpolation. As an example, consider the following formula:

```
{ClaimAmountCut / Exposure} ~ C(DrivAge, missing_method='convert') * C(VehPower, missing_method="zero") + bs(BonusMalus, 3)
```

Despite its brevity, it describes all of the following:

- The outcome variable is the ratio of `ClaimAmountCut` and `Exposure`.
- The predictors should include the interactions of the categorical variables `DrivAge` and `VehPower`, as well as those two variables themselves. (Even though they behave as such, neither the individual variables nor their interaction will be dummy-encoded by glum. For categoricals with many levels, this can lead to a substantial performance improvement over dummy encoding, especially for the interaction.)
- If there are missing values in `DrivAge`, they should be treated as a separate category.
- On the other hand, missing values in `VehPower` should be treated as all-zero indicators.
- The predictors should also include a third degree B-spline interpolation of `BonusMalus`.

The following chapters demonstrate each of these features in some detail, as well as some additional advantages of using the formula interface.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats
from dask_ml.preprocessing import Categorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from glum import GeneralizedLinearRegressor
from glum import TweedieDistribution

from load_transform_formula import load_transform
```

## 1. Load and Prepare Datasets from Openml

First, we load in our [dataset from openML](https://www.openml.org/d/41214) and apply several transformations. In the interest of simplicity, we do not include the data loading and preparation code in this notebook.

```python
df = load_transform()
with pd.option_context('display.max_rows', 10):
    display(df)
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClaimNb</th>
      <th>Exposure</th>
      <th>Area</th>
      <th>VehPower</th>
      <th>VehAge</th>
      <th>DrivAge</th>
      <th>BonusMalus</th>
      <th>VehBrand</th>
      <th>VehGas</th>
      <th>Density</th>
      <th>Region</th>
      <th>ClaimAmount</th>
      <th>ClaimAmountCut</th>
    </tr>
    <tr>
      <th>IDpol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.10000</td>
      <td>D</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>1217</td>
      <td>R82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.77000</td>
      <td>D</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>1217</td>
      <td>R82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.75000</td>
      <td>B</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>54</td>
      <td>R22</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0.09000</td>
      <td>B</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>76</td>
      <td>R72</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0.84000</td>
      <td>B</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>76</td>
      <td>R72</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6114326</th>
      <td>0</td>
      <td>0.00274</td>
      <td>E</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>3317</td>
      <td>R93</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6114327</th>
      <td>0</td>
      <td>0.00274</td>
      <td>E</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>95</td>
      <td>B12</td>
      <td>Regular</td>
      <td>9850</td>
      <td>R11</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6114328</th>
      <td>0</td>
      <td>0.00274</td>
      <td>D</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>50</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>1323</td>
      <td>R82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6114329</th>
      <td>0</td>
      <td>0.00274</td>
      <td>B</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>B12</td>
      <td>Regular</td>
      <td>95</td>
      <td>R26</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6114330</th>
      <td>0</td>
      <td>0.00274</td>
      <td>B</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>54</td>
      <td>B12</td>
      <td>Diesel</td>
      <td>65</td>
      <td>R72</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>678013 rows × 13 columns</p>
</div>
</div>

## 2. Reproducing the Model From the GLM Tutorial

Now, let us start by fitting a very simple model. As usual, let's divide our samples into a training and a test set so that we get valid out-of-sample goodness-of-fit measures. Perhaps less usually, we do not create separate `y` and `X` data frames for our label and features – the formula will take care of that for us.

We still have some preprocessing to do:

- Many of the ordinal or nominal variables are encoded as integers, instead of as categoricals. We will need to convert these so that `glum` will know to estimate a separate coefficient for each of their levels.
- The outcome variable is a transformation of other columns. We need to create it first.

As we will see later on, these steps can be incorporated into the formula itself, but let's not overcomplicate things at first.

```python
ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train, test = next(ss.split(df))

df = df.assign(PurePremium=lambda x: x["ClaimAmountCut"] / x["Exposure"])

glm_categorizer = Categorizer(
    columns=["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
)
df_train = glm_categorizer.fit_transform(df.iloc[train])
df_test = glm_categorizer.transform(df.iloc[test])
```

This example demonstrates the basic idea behind formulas: the outcome variable and the predictors are separated by a tilde (`~`), and different predictors are separated by plus signs (`+`). Thus, formulas provide a concise way of specifying a model without the need to create dataframes by hand.

```python
formula = "PurePremium ~ VehBrand + VehGas + Region + Area + DrivAge + VehAge + VehPower + BonusMalus + Density"

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula,
)
t_glm1.fit(df_train, sample_weight=df["Exposure"].values[train])

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>VehBrand[B1]</th>
      <th>VehBrand[B10]</th>
      <th>VehBrand[B11]</th>
      <th>VehBrand[B12]</th>
      <th>VehBrand[B13]</th>
      <th>VehBrand[B14]</th>
      <th>VehBrand[B2]</th>
      <th>VehBrand[B3]</th>
      <th>VehBrand[B4]</th>
      <th>...</th>
      <th>VehAge[1]</th>
      <th>VehAge[2]</th>
      <th>VehPower[4]</th>
      <th>VehPower[5]</th>
      <th>VehPower[6]</th>
      <th>VehPower[7]</th>
      <th>VehPower[8]</th>
      <th>VehPower[9]</th>
      <th>BonusMalus</th>
      <th>Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>2.88667</td>
      <td>-0.064157</td>
      <td>0.0</td>
      <td>0.231868</td>
      <td>-0.211061</td>
      <td>0.054979</td>
      <td>-0.270346</td>
      <td>-0.071453</td>
      <td>0.00291</td>
      <td>0.059324</td>
      <td>...</td>
      <td>0.008117</td>
      <td>-0.229906</td>
      <td>-0.111796</td>
      <td>-0.123388</td>
      <td>0.060757</td>
      <td>0.005179</td>
      <td>-0.021832</td>
      <td>0.208158</td>
      <td>0.032508</td>
      <td>0.000002</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 60 columns</p>
</div>
</div>

## 3. Categorical Variables

`glum` also provides extensive support for categorical variables. The main function one needs to be aware of in the context of categoricals is simply called `C()`. A variable placed within it is always converted to a categorical, regardless of its type.

A huge part of tabmat's/glum's performance advantage is that categoricals need not be one-hot encoded, but are treated as if they were. For this reason, we do not support using other coding schemes within the formula interface. If one needs to use other categorical encodings than one-hot, they can always do so manually (or even using `formulaic` directly) before the estimation.

Let's try it out on our dataset!

```python
df_train_noncat = df.iloc[train]
df_test_noncat = df.iloc[test]

df_train_noncat.dtypes
```

```text
ClaimNb             int64
Exposure          float64
Area                  str
VehPower            int64
VehAge              int64
DrivAge             int64
BonusMalus          int64
VehBrand              str
VehGas                str
Density             int64
Region                str
ClaimAmount       float64
ClaimAmountCut    float64
PurePremium       float64
dtype: object
```

Even though some of the variables are integers in this dataset, they are handled as categoricals thanks to the `C()` function. Strings, such as `VehBrand` or `VehGas` would have been handled as categorical by default anyway, but using the `C()` function never hurts: if applied to something that is already a categorical variable, it does not have any effect outside of the feature name.

```python
formula_cat = (
    "PurePremium ~ C(VehBrand) + C(VehGas) + C(Region) + C(Area) "
    "+ C(DrivAge) + C(VehAge) + C(VehPower) + BonusMalus + Density"
)

t_glm3 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula_cat,
)
t_glm3.fit(df_train_noncat, sample_weight=df["Exposure"].values[train])

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm3.intercept_], t_glm3.coef_))},
    index=["intercept"] + t_glm3.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>C(VehBrand)[B1]</th>
      <th>C(VehBrand)[B10]</th>
      <th>C(VehBrand)[B11]</th>
      <th>C(VehBrand)[B12]</th>
      <th>C(VehBrand)[B13]</th>
      <th>C(VehBrand)[B14]</th>
      <th>C(VehBrand)[B2]</th>
      <th>C(VehBrand)[B3]</th>
      <th>C(VehBrand)[B4]</th>
      <th>...</th>
      <th>C(VehAge)[1]</th>
      <th>C(VehAge)[2]</th>
      <th>C(VehPower)[4]</th>
      <th>C(VehPower)[5]</th>
      <th>C(VehPower)[6]</th>
      <th>C(VehPower)[7]</th>
      <th>C(VehPower)[8]</th>
      <th>C(VehPower)[9]</th>
      <th>BonusMalus</th>
      <th>Density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>2.88667</td>
      <td>-0.064157</td>
      <td>0.0</td>
      <td>0.231868</td>
      <td>-0.211061</td>
      <td>0.054979</td>
      <td>-0.270346</td>
      <td>-0.071453</td>
      <td>0.00291</td>
      <td>0.059324</td>
      <td>...</td>
      <td>0.008117</td>
      <td>-0.229906</td>
      <td>-0.111796</td>
      <td>-0.123388</td>
      <td>0.060757</td>
      <td>0.005179</td>
      <td>-0.021832</td>
      <td>0.208158</td>
      <td>0.032508</td>
      <td>0.000002</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 60 columns</p>
</div>
</div>

Finally, prediction works as expected with categorical variables. `glum` keeps track of the levels present in the training dataset, and makes sure that categorical variables in unseen datasets are also properly aligned, even if they have missing or unknown levels.<sup>3</sup> Therefore, one can simply use predict, and `glum` does The Right Thing™ by default.

<sup>3</sup>: This is made possible due to `glum` saving a [`ModelSpec` object](https://matthewwardrop.github.io/formulaic/guides/model_specs/), which contains any information necessary for reapplying the transitions that were done during the formula materialization process. It is especially relevant in the case of [stateful transforms](https://matthewwardrop.github.io/formulaic/guides/transforms/), such as creating categorical variables.

```python
t_glm3.predict(df_test_noncat)
```

```text
array([303.77443311, 548.47789523, 244.34438579, ..., 109.81572865,
        67.98332028, 297.21717383], shape=(67802,))
```

## 4. Interactions and Structural Full-Rankness

One of the biggest strengths of Wilkinson-formulas lie in their ability of concisely specifying interactions between terms. `glum` implements this as well, and in a very efficient way: the interactions of categorical features are encoded as a new categorical feature, making it possible to interact high-cardinality categoricals with each other. If this is not possible, because, for example, a categorical is interacted with a numeric variable, sparse representations are used when appropriate. In general, just as with `glum`'s categorical handling in general, you can be assured that you don't have to worry too much about the actual implementation, and can expect that `glum` will do the most efficient thing behind the scenes.

Let's see how that looks like on the insurance example! Suppose that we expect `VehPower` to have a different effect depending on `DrivAge` (e.g. performance cars might not be great for new drivers, but may be less problematic for more experienced ones). We can include the interaction of these variables as follows.

```python
formula_int = (
    "PurePremium ~ C(VehBrand) + C(VehGas) + C(Region) + C(Area)"
    " + C(DrivAge) * C(VehPower) + C(VehAge) + BonusMalus + Density"
)

t_glm4 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula_int,
)
t_glm4.fit(df_train, sample_weight=df["Exposure"].values[train])

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm4.intercept_], t_glm4.coef_))},
    index=["intercept"] + t_glm4.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>C(VehBrand)[B1]</th>
      <th>C(VehBrand)[B10]</th>
      <th>C(VehBrand)[B11]</th>
      <th>C(VehBrand)[B12]</th>
      <th>C(VehBrand)[B13]</th>
      <th>C(VehBrand)[B14]</th>
      <th>C(VehBrand)[B2]</th>
      <th>C(VehBrand)[B3]</th>
      <th>C(VehBrand)[B4]</th>
      <th>...</th>
      <th>C(DrivAge)[4]:C(VehPower)[8]</th>
      <th>C(DrivAge)[5]:C(VehPower)[8]</th>
      <th>C(DrivAge)[6]:C(VehPower)[8]</th>
      <th>C(DrivAge)[0]:C(VehPower)[9]</th>
      <th>C(DrivAge)[1]:C(VehPower)[9]</th>
      <th>C(DrivAge)[2]:C(VehPower)[9]</th>
      <th>C(DrivAge)[3]:C(VehPower)[9]</th>
      <th>C(DrivAge)[4]:C(VehPower)[9]</th>
      <th>C(DrivAge)[5]:C(VehPower)[9]</th>
      <th>C(DrivAge)[6]:C(VehPower)[9]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>2.88023</td>
      <td>-0.069076</td>
      <td>0.0</td>
      <td>0.221037</td>
      <td>-0.211854</td>
      <td>0.052355</td>
      <td>-0.272058</td>
      <td>-0.074836</td>
      <td>0.0</td>
      <td>0.052523</td>
      <td>...</td>
      <td>-0.147844</td>
      <td>-0.03567</td>
      <td>0.504407</td>
      <td>0.682528</td>
      <td>-0.106569</td>
      <td>-0.308257</td>
      <td>0.173206</td>
      <td>0.010684</td>
      <td>-0.220273</td>
      <td>0.070334</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 102 columns</p>
</div>
</div>

Note that, in addition to the interactions, the non-interacted variants of `DrivAge` and `VehPower` are also included in the model. This is a result of using the `*` operator to interact the variables. Using `:` instead would only include the interactions, and not the marginals. (In short, `a * b` is equivalent to `a + b + a:b`.)

```python
[name for name in t_glm4.feature_names_ if "VehPower" in name]
```

```text
['C(VehPower)[4]',
 'C(VehPower)[5]',
 'C(VehPower)[6]',
 'C(VehPower)[7]',
 'C(VehPower)[8]',
 'C(VehPower)[9]',
 'C(DrivAge)[0]:C(VehPower)[4]',
 'C(DrivAge)[1]:C(VehPower)[4]',
 'C(DrivAge)[2]:C(VehPower)[4]',
 'C(DrivAge)[3]:C(VehPower)[4]',
 'C(DrivAge)[4]:C(VehPower)[4]',
 'C(DrivAge)[5]:C(VehPower)[4]',
 'C(DrivAge)[6]:C(VehPower)[4]',
 'C(DrivAge)[0]:C(VehPower)[5]',
 'C(DrivAge)[1]:C(VehPower)[5]',
 'C(DrivAge)[2]:C(VehPower)[5]',
 'C(DrivAge)[3]:C(VehPower)[5]',
 'C(DrivAge)[4]:C(VehPower)[5]',
 'C(DrivAge)[5]:C(VehPower)[5]',
 'C(DrivAge)[6]:C(VehPower)[5]',
 'C(DrivAge)[0]:C(VehPower)[6]',
 'C(DrivAge)[1]:C(VehPower)[6]',
 'C(DrivAge)[2]:C(VehPower)[6]',
 'C(DrivAge)[3]:C(VehPower)[6]',
 'C(DrivAge)[4]:C(VehPower)[6]',
 'C(DrivAge)[5]:C(VehPower)[6]',
 'C(DrivAge)[6]:C(VehPower)[6]',
 'C(DrivAge)[0]:C(VehPower)[7]',
 'C(DrivAge)[1]:C(VehPower)[7]',
 'C(DrivAge)[2]:C(VehPower)[7]',
 'C(DrivAge)[3]:C(VehPower)[7]',
 'C(DrivAge)[4]:C(VehPower)[7]',
 'C(DrivAge)[5]:C(VehPower)[7]',
 'C(DrivAge)[6]:C(VehPower)[7]',
 'C(DrivAge)[0]:C(VehPower)[8]',
 'C(DrivAge)[1]:C(VehPower)[8]',
 'C(DrivAge)[2]:C(VehPower)[8]',
 'C(DrivAge)[3]:C(VehPower)[8]',
 'C(DrivAge)[4]:C(VehPower)[8]',
 'C(DrivAge)[5]:C(VehPower)[8]',
 'C(DrivAge)[6]:C(VehPower)[8]',
 'C(DrivAge)[0]:C(VehPower)[9]',
 'C(DrivAge)[1]:C(VehPower)[9]',
 'C(DrivAge)[2]:C(VehPower)[9]',
 'C(DrivAge)[3]:C(VehPower)[9]',
 'C(DrivAge)[4]:C(VehPower)[9]',
 'C(DrivAge)[5]:C(VehPower)[9]',
 'C(DrivAge)[6]:C(VehPower)[9]']
```

The attentive reader might have also noticed that the first level of each categorical variable is omitted from the model. This is a manifestation of the more general concept of [ensuring structural full-rankedness](https://matthewwardrop.github.io/formulaic/guides/contrasts/#guaranteeing-structural-full-rankness)<sup>4</sup>. By default, `glum` and `formulaic` will try to make sure that one does not fall into the [Dummy Variable Trap](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)). Moreover, it even does it in the case of (possibly multi-way) interactions involving categorical variables. It will always drop the necessary number of levels, and no more. If you want to opt out of this behavior (for example because you would like to penalize all levels equally), simply set the `drop_first` parameter during model initialization to `False`. If one only aims to include all levels of a certain variable, and not others, it is possible to do so by using the `spans_intercept` parameter (e.g. `C(VehPower, spans_intercept=False)` would include all levels of `VehPower` even if `drop_first` is set to `True`).

<sup>4</sup>: Note, that it does not guarantee that the design matrix is actually full rank. For example, two identical numerical variables will still lead to a rank-deficient design matrix.

## 5. Fun with Functions

The previous example is only scratching the surface of what formulas are capable of. For example, they are capable of evaluating arbitrary Python expressions, which act as if they saw the columns of the input data frame as local variables (`pandas.Series`). The way to tell `glum` that a part of the formula should be evaluated as a Python expression before applying the formula grammar to it is to enclose it in curly braces. As an example, we can easily do the following within the formula itself:

 1. Create the outcome variable on the fly instead of doing it beforehand.
 2. Include the logarithm of a certain variable in the model.
 3. Include a basis spline interpolation of a variable to capture non-linearities in its effect.

1\. works because formulas can contain [Python operations](https://matthewwardrop.github.io/formulaic/guides/grammar/). 2. and 3. work because formulas are evaluated within a context that is aware of a number of [transforms](https://matthewwardrop.github.io/formulaic/guides/transforms/). To be precise, 2. is a regular transform and 3. is a stateful transform.

Let's try it out!

```python
formula_fun = (
    "{ClaimAmountCut / Exposure} ~ VehBrand + VehGas + Region + Area"
    " + DrivAge + VehAge + VehPower + bs(BonusMalus, 3) + np.log(Density)"
)

t_glm5 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula_fun,
)
t_glm5.fit(df_train, sample_weight=df["Exposure"].values[train])

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm5.intercept_], t_glm5.coef_))},
    index=["intercept"] + t_glm5.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>VehBrand[B1]</th>
      <th>VehBrand[B10]</th>
      <th>VehBrand[B11]</th>
      <th>VehBrand[B12]</th>
      <th>VehBrand[B13]</th>
      <th>VehBrand[B14]</th>
      <th>VehBrand[B2]</th>
      <th>VehBrand[B3]</th>
      <th>VehBrand[B4]</th>
      <th>...</th>
      <th>VehPower[4]</th>
      <th>VehPower[5]</th>
      <th>VehPower[6]</th>
      <th>VehPower[7]</th>
      <th>VehPower[8]</th>
      <th>VehPower[9]</th>
      <th>bs(BonusMalus, 3)[1]</th>
      <th>bs(BonusMalus, 3)[2]</th>
      <th>bs(BonusMalus, 3)[3]</th>
      <th>np.log(Density)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>3.808829</td>
      <td>-0.060201</td>
      <td>0.0</td>
      <td>0.242194</td>
      <td>-0.202517</td>
      <td>0.063471</td>
      <td>-0.345415</td>
      <td>-0.072546</td>
      <td>0.00777</td>
      <td>0.079391</td>
      <td>...</td>
      <td>-0.113038</td>
      <td>-0.127255</td>
      <td>0.060209</td>
      <td>0.005577</td>
      <td>-0.032114</td>
      <td>0.207355</td>
      <td>3.178178</td>
      <td>0.361951</td>
      <td>8.231846</td>
      <td>0.121944</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 62 columns</p>
</div>
</div>

To allow for even more flexibility, you can add custom transformations that are defined in the context from which the call is made. E.g., we can define a transformation that takes the logarithm of `VehAge + 1` after casting it to numeric. To make the formula recognize this transform, you need to explicitly set `context=0` when calling the fit method (note that this differs from `formulaic`'s default, which is already `context=0`).

```python
def _log_plus_one(x):
    return np.log(pd.to_numeric(x) + 1)

formula_custom_fun = (
    "{ClaimAmountCut / Exposure} ~ _log_plus_one(VehAge)"
)

t_glm6 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula_custom_fun,
)
t_glm6.fit(df_train, sample_weight=df["Exposure"].values[train], context=0)

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm6.intercept_], t_glm6.coef_))},
    index=["intercept"] + t_glm6.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>_log_plus_one(VehAge)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>5.046712</td>
      <td>-0.151043</td>
    </tr>
  </tbody>
</table>
</div>
</div>

## 6. Monotonic Constraints

When using the formula interface, `glum` supports **monotonic constraints** via the `monotonic_constraints` parameter. This is useful when domain knowledge tells you that a variable's effect should only go in one direction. For example, we might expect that a higher `BonusMalus` score (which indicates a worse claims history) should never *decrease* the predicted pure premium.

Monotonic constraints work with any term that expands to multiple columns — such as B-splines or ordered categoricals — by enforcing that consecutive coefficients are ordered. For single-column terms (e.g. a bare numeric variable), they enforce a sign constraint (non-negative for `"increasing"`, non-positive for `"decreasing"`).

Constraints are specified as a dictionary mapping **variable names** (not formula terms) to `"increasing"` or `"decreasing"`. `glum` automatically resolves variable names to the corresponding expanded terms. Under the hood, this builds linear inequality constraints and solves them via a penalty-based IRLS algorithm (`solver='irls-ls-monotonic'`). You can also set `solver='trust-constr'` explicitly for an alternative approach based on scipy's constrained optimizer.

```python
formula_mono = (
    "{ClaimAmountCut / Exposure} ~ VehBrand + VehGas + Region + Area"
    " + DrivAge + VehAge + VehPower + bs(BonusMalus, 3) + np.log(Density)"
)

t_glm_mono = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha=0.001,
    l1_ratio=0,
    fit_intercept=True,
    formula=formula_mono,
    monotonic_constraints={"BonusMalus": "increasing"},
    max_iter=500,
)
t_glm_mono.fit(df_train, sample_weight=df["Exposure"].values[train])

# The BonusMalus spline coefficients are monotonically increasing:
bm_mask = ["BonusMalus" in n for n in t_glm_mono.feature_names_]
bm_coefs = t_glm_mono.coef_[bm_mask]
bm_names = [n for n in t_glm_mono.feature_names_ if "BonusMalus" in n]

pd.DataFrame({"coefficient": bm_coefs}, index=bm_names).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bs(BonusMalus, 3)[1]</th>
      <th>bs(BonusMalus, 3)[2]</th>
      <th>bs(BonusMalus, 3)[3]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>2.651015</td>
      <td>2.651015</td>
      <td>2.651015</td>
    </tr>
  </tbody>
</table>
</div>
</div>

## 7. Miscellaneous Features

### Variable Names

`glum`'s formula interface provides a lot of control over how the resulting features are named. By default, it follows `formulaic`'s standards, but it can be customized by setting the `interaction_separator` and `categorical_format` parameters.

```python
formula_name = "PurePremium ~ DrivAge * VehPower"

t_glm7 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=True,
    formula=formula_name,
    interaction_separator="__x__",
    categorical_format="{name}__{category}",
)
t_glm7.fit(df_train, sample_weight=df["Exposure"].values[train])

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm7.intercept_], t_glm7.coef_))},
    index=["intercept"] + t_glm7.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>DrivAge__0</th>
      <th>DrivAge__1</th>
      <th>DrivAge__2</th>
      <th>DrivAge__3</th>
      <th>DrivAge__4</th>
      <th>DrivAge__5</th>
      <th>DrivAge__6</th>
      <th>VehPower__4</th>
      <th>VehPower__5</th>
      <th>...</th>
      <th>DrivAge__4__x__VehPower__8</th>
      <th>DrivAge__5__x__VehPower__8</th>
      <th>DrivAge__6__x__VehPower__8</th>
      <th>DrivAge__0__x__VehPower__9</th>
      <th>DrivAge__1__x__VehPower__9</th>
      <th>DrivAge__2__x__VehPower__9</th>
      <th>DrivAge__3__x__VehPower__9</th>
      <th>DrivAge__4__x__VehPower__9</th>
      <th>DrivAge__5__x__VehPower__9</th>
      <th>DrivAge__6__x__VehPower__9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>5.007277</td>
      <td>1.497079</td>
      <td>0.53565</td>
      <td>0.0</td>
      <td>-0.152974</td>
      <td>-0.210998</td>
      <td>-0.205689</td>
      <td>0.017896</td>
      <td>-0.096153</td>
      <td>-0.05484</td>
      <td>...</td>
      <td>-0.143822</td>
      <td>-0.002094</td>
      <td>0.512258</td>
      <td>0.730534</td>
      <td>-0.280869</td>
      <td>-0.367669</td>
      <td>0.171063</td>
      <td>0.022052</td>
      <td>-0.270456</td>
      <td>0.119634</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 56 columns</p>
</div>
</div>

### Intercept Term

Just like in the case of the non-formula interface, the presence of an intercept is determined by the `fit_intercept` argument. In case that the formula specifies a different behavior (e.g., adding `+0` or `-1` while `fit_intercept=True`), an error will be raised.

```python
formula_noint = "PurePremium ~ DrivAge * VehPower - 1"

try:
    t_glm8 = GeneralizedLinearRegressor(
        family=TweedieDist,
        alpha_search=True,
        l1_ratio=1,
        fit_intercept=True,
        formula=formula_noint,
        interaction_separator="__x__",
        categorical_format="{name}__{category}",
    ).fit(df_train, sample_weight=df["Exposure"].values[train])
    raise AssertionError("Expected ValueError was not raised")
except ValueError as e:
    assert "The formula sets the intercept to False" in str(e)
    print(f"Caught expected ValueError: {e}")
```

```text
Caught expected ValueError: The formula sets the intercept to False, contradicting fit_intercept=True. You should use fit_intercept to specify the intercept.
```

### One-Sided Formulas

Even when using formulas, the outcome variable can be specified as a vector, as in the interface without formulas. In that case the supplied formula should be one-sided (not contain a `~`), and only describe the right-hand side of the regression.

```python
formula_onesie = "DrivAge * VehPower"

t_glm8 = GeneralizedLinearRegressor(
    family=TweedieDist,
    alpha_search=True,
    l1_ratio=1,
    fit_intercept=False,
    formula=formula_onesie,
    interaction_separator="__x__",
    categorical_format="{name}__{category}",
)
t_glm8.fit(
    X=df_train, y=df_train["PurePremium"], sample_weight=df["Exposure"].values[train]
)

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm8.intercept_], t_glm8.coef_))},
    index=["intercept"] + t_glm8.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>DrivAge__0</th>
      <th>DrivAge__1</th>
      <th>DrivAge__2</th>
      <th>DrivAge__3</th>
      <th>DrivAge__4</th>
      <th>DrivAge__5</th>
      <th>DrivAge__6</th>
      <th>VehPower__4</th>
      <th>VehPower__5</th>
      <th>...</th>
      <th>DrivAge__4__x__VehPower__8</th>
      <th>DrivAge__5__x__VehPower__8</th>
      <th>DrivAge__6__x__VehPower__8</th>
      <th>DrivAge__0__x__VehPower__9</th>
      <th>DrivAge__1__x__VehPower__9</th>
      <th>DrivAge__2__x__VehPower__9</th>
      <th>DrivAge__3__x__VehPower__9</th>
      <th>DrivAge__4__x__VehPower__9</th>
      <th>DrivAge__5__x__VehPower__9</th>
      <th>DrivAge__6__x__VehPower__9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>0.0</td>
      <td>1.713298</td>
      <td>0.783505</td>
      <td>0.205914</td>
      <td>0.016085</td>
      <td>0.0</td>
      <td>0.000094</td>
      <td>0.223685</td>
      <td>4.66123</td>
      <td>4.736272</td>
      <td>...</td>
      <td>-0.144927</td>
      <td>0.001657</td>
      <td>0.515373</td>
      <td>0.714834</td>
      <td>-0.325666</td>
      <td>-0.370935</td>
      <td>0.20417</td>
      <td>0.013222</td>
      <td>-0.273913</td>
      <td>0.115693</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 56 columns</p>
</div>
</div>

### Missing Values in Categorical Columns

By default, `glum` raises a `ValueError` when it encounters a missing value in a categorical variable (`"raise"` option). However, there are two other options for handling these cases. They can also be treated as if they represented all-zeros indicators (`"zero"` option, which is also the way `pandas.get_dummies` works) or missing values can be treated as their own separate category (`"convert"` option).

Similarly to the non-formula-based interface, `glum`'s behavior can be set globally using the `cat_missing_method` parameter during model initialization. However, formulas provide some additional flexibility: the `C` function has a `missing_method` parameter, with which users can select an option on a column-by-column basis. Here is an example of doing that (although our dataset does not have any missing values, so these options have no actual effect in this case):

```python
formula_missing = "C(DrivAge, missing_method='zero') + C(VehPower, missing_method='convert')"

t_glm9 = GeneralizedLinearRegressor(
    family=TweedieDist,

    alpha_search=True,
    l1_ratio=1,
    fit_intercept=False,
    formula=formula_missing,

)
t_glm9.fit(
    X=df_train, y=df_train["PurePremium"], sample_weight=df["Exposure"].values[train]
)

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm9.intercept_], t_glm9.coef_))},
    index=["intercept"] + t_glm9.feature_names_,
).T
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>C(DrivAge, missing_method='zero')[0]</th>
      <th>C(DrivAge, missing_method='zero')[1]</th>
      <th>C(DrivAge, missing_method='zero')[2]</th>
      <th>C(DrivAge, missing_method='zero')[3]</th>
      <th>C(DrivAge, missing_method='zero')[4]</th>
      <th>C(DrivAge, missing_method='zero')[5]</th>
      <th>C(DrivAge, missing_method='zero')[6]</th>
      <th>C(VehPower, missing_method='convert')[4]</th>
      <th>C(VehPower, missing_method='convert')[5]</th>
      <th>C(VehPower, missing_method='convert')[6]</th>
      <th>C(VehPower, missing_method='convert')[7]</th>
      <th>C(VehPower, missing_method='convert')[8]</th>
      <th>C(VehPower, missing_method='convert')[9]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coefficient</th>
      <td>0.0</td>
      <td>1.786703</td>
      <td>0.742765</td>
      <td>0.239528</td>
      <td>0.096531</td>
      <td>0.071118</td>
      <td>0.0</td>
      <td>0.201078</td>
      <td>4.637267</td>
      <td>4.679391</td>
      <td>4.863387</td>
      <td>4.77263</td>
      <td>4.749673</td>
      <td>4.970188</td>
    </tr>
  </tbody>
</table>
</div>
</div>
