# The Cox Proportional Hazards Model in Glum

**Intro**

This tutorial shows how the Cox proportional hazards model (from now on: Cox model) which cannot be represented as an Exponential Dispersion Model (EDM), can still be estimated by a simple data transformation followed by a standard Poisson regression in `glum` (from now on: Poisson approach). The Poisson approach requires estimating the coefficients of a high-dimensional categorical of time-fixed-effects, which can be done efficiently in glum, leveraging the capabilities of `tabmat`'s `CategoricalMatrix`. The exposition of the Poisson approach here is based on [1], but the equivalence has been described before [2].

## 1. Equivalence Between the Cox Likelihood and a Profile Poisson Likelihood

In the Cox model, the rate of event occurrence, $\lambda(t,x_i)$, factorizes nicely into a linear predictor $\eta_i=\sum_k \beta_k x_{ik}$ that depends on individual $i$'s characteristics but not on time $t$, and a baseline hazard $\lambda_0$ that depends only on time: $\lambda(t,x_i)=\lambda_0(t)\exp(\eta_i)$. This is known as the proportional hazards assumption). The partial log-likelihood of $\eta_i$ is

$$
\sum_{\text{event times}}\log\left(\frac{y_{i,t}\exp(\eta_{i})}{\sum_{i \in \mathcal{R}_t} \exp(\eta_i)} \right),
$$

where $\mathcal{R}_t$ is the set of individuals observed at event time $t$ and $y_{i,t}$ is one if the individual has an event at $t$ and zero otherwise.<sup>1</sup> This partial log-likelihood cannot be represented as the log-likelihood of an EDM.<sup>2</sup> Now consider an alternative Poisson regression with  $y_{i,t}$ as an outcome. Apart from a constant, the log likelihood is

$$
\sum_{\text{event times}}\sum_{i \in \mathcal{R}_t} y_{i,t} \log(\lambda(t,x_i)) - \lambda(t,x_i).
$$

Using the proportional hazards assumption and letting $\alpha_t = \log(\lambda_0(t))$, this becomes

$$
\sum_{\text{event times}}\sum_{i \in \mathcal{R}_t} y_{i,t} \left(\alpha_t + \eta_i\right) - \exp(\alpha_t + \eta_i).
$$

Solving the first order condition with respect to $\alpha_t$ yields $\exp(\hat{\alpha}_t) = \left(\sum_{i \in \mathcal{R}_t} \exp(\eta_i)\right)^{-1}$. This can be plugged back into the log likelihood to yield, after some simplifications,

$$
\sum_{\text{event times}}\log\left(\frac{y_{i,t}\exp(\eta_{i})}{\sum_{i \in \mathcal{R}_t} \exp(\eta_i)} \right) - 1,
$$

which is the same as the partial likelihood in the Cox model, apart from the -1 which drops out when taking derivatives. In short, the Cox partial log likelihood is equivalent to a Poisson log likelihood with the estimate for time period effects fed back in ("profiled out"). This means that, to estimate the parameters of the Cox model, one can simply run a Poisson regression with time fixed effects $\alpha_t$. The data structures for the two objectives are different: the Cox partial log-likelihood operates on data with one row per observed individual, while the Poisson log-likelihood uses one row per individual and time period.

## 2. Estimating a Cox Model in Glum

We now show that a Poisson approach in `glum` yields the same parameter estimates as a Cox model. For the latter, we use the [lifelines](https://github.com/CamDavidsonPilon/lifelines) library. We also take the dataset from lifelines, which is from an RCT on recidivism for 432 convicts released from Maryland state prisons with first arrest after release as event. We first load imports and the dataset. The dataset has one row per convict, with two outcome columns: the `week` until which the observation lasts and `arrest`, which indicates whether an arrest event happened or not (censoring).

```python
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import glum

df = load_rossi()
df.head()
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>arrest</th>
      <th>fin</th>
      <th>age</th>
      <th>race</th>
      <th>wexp</th>
      <th>mar</th>
      <th>paro</th>
      <th>prio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
</div>

Next, we estimate the Cox model with an indicator for financial aid (the treatment), a B-spline in the age at time of release, indicators for race, prior experience, marital status, and parole, and a B-spline in the number of prior convictions:

```python
cph = CoxPHFitter()
cph.fit(
    df,
    duration_col="week",
    event_col="arrest",
    formula="fin + bs(age, df=4) + race + wexp + mar + paro + bs(prio, df=3)",
)
cph.print_summary()
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'week'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'arrest'</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>432</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>114</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-656.25</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2026-11-17 09:32:40 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fin</th>
      <td>-0.35</td>
      <td>0.70</td>
      <td>0.19</td>
      <td>-0.73</td>
      <td>0.03</td>
      <td>0.48</td>
      <td>1.03</td>
      <td>0.00</td>
      <td>-1.82</td>
      <td>0.07</td>
      <td>3.85</td>
    </tr>
    <tr>
      <th>bs(age, df=4)[1]</th>
      <td>-0.49</td>
      <td>0.61</td>
      <td>0.64</td>
      <td>-1.74</td>
      <td>0.76</td>
      <td>0.18</td>
      <td>2.13</td>
      <td>0.00</td>
      <td>-0.77</td>
      <td>0.44</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>bs(age, df=4)[2]</th>
      <td>-1.81</td>
      <td>0.16</td>
      <td>0.84</td>
      <td>-3.46</td>
      <td>-0.16</td>
      <td>0.03</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>-2.15</td>
      <td>0.03</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>bs(age, df=4)[3]</th>
      <td>-0.91</td>
      <td>0.40</td>
      <td>1.41</td>
      <td>-3.67</td>
      <td>1.86</td>
      <td>0.03</td>
      <td>6.40</td>
      <td>0.00</td>
      <td>-0.64</td>
      <td>0.52</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>bs(age, df=4)[4]</th>
      <td>-1.76</td>
      <td>0.17</td>
      <td>1.10</td>
      <td>-3.92</td>
      <td>0.41</td>
      <td>0.02</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>-1.59</td>
      <td>0.11</td>
      <td>3.17</td>
    </tr>
    <tr>
      <th>race</th>
      <td>0.36</td>
      <td>1.43</td>
      <td>0.31</td>
      <td>-0.25</td>
      <td>0.96</td>
      <td>0.78</td>
      <td>2.62</td>
      <td>0.00</td>
      <td>1.15</td>
      <td>0.25</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>wexp</th>
      <td>-0.09</td>
      <td>0.91</td>
      <td>0.22</td>
      <td>-0.52</td>
      <td>0.33</td>
      <td>0.60</td>
      <td>1.39</td>
      <td>0.00</td>
      <td>-0.43</td>
      <td>0.67</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>mar</th>
      <td>-0.33</td>
      <td>0.72</td>
      <td>0.39</td>
      <td>-1.09</td>
      <td>0.42</td>
      <td>0.34</td>
      <td>1.53</td>
      <td>0.00</td>
      <td>-0.87</td>
      <td>0.39</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>paro</th>
      <td>-0.14</td>
      <td>0.87</td>
      <td>0.20</td>
      <td>-0.53</td>
      <td>0.25</td>
      <td>0.59</td>
      <td>1.28</td>
      <td>0.00</td>
      <td>-0.70</td>
      <td>0.48</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>bs(prio, df=3)[1]</th>
      <td>1.36</td>
      <td>3.91</td>
      <td>0.96</td>
      <td>-0.53</td>
      <td>3.25</td>
      <td>0.59</td>
      <td>25.87</td>
      <td>0.00</td>
      <td>1.42</td>
      <td>0.16</td>
      <td>2.67</td>
    </tr>
    <tr>
      <th>bs(prio, df=3)[2]</th>
      <td>-0.24</td>
      <td>0.78</td>
      <td>1.05</td>
      <td>-2.30</td>
      <td>1.81</td>
      <td>0.10</td>
      <td>6.11</td>
      <td>0.00</td>
      <td>-0.23</td>
      <td>0.82</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>bs(prio, df=3)[3]</th>
      <td>2.74</td>
      <td>15.47</td>
      <td>0.81</td>
      <td>1.16</td>
      <td>4.32</td>
      <td>3.19</td>
      <td>74.97</td>
      <td>0.00</td>
      <td>3.40</td>
      <td>&lt;0.005</td>
      <td>10.54</td>
    </tr>
  </tbody>
</table><br><div>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.65</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>1336.50</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>38.26 on 12 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>12.81</td>
    </tr>
  </tbody>
</table>
</div>
</div>

The results imply that financial aid leads to an about 30% reduction in the hazard of being arrested again, but the effect is just barely significant. We can now go about replicating these results in `glum`. For that, we first need to define a function that creates a dataset with one row per convict and period:

```python
def survival_split(df, time, outcome):
    """Split survival data into one row per observation and period. Inspired by `SurvSplit` in R."""

    # table with unique event or censoring times
    df_times = df[[time]].drop_duplicates().sort_values(time)

    # create table with one row per time and left row
    df["temp"] = 1
    # optional: add id
    df_times["temp"] = 1
    df_ = pd.merge(df, df_times, on="temp", how="left", suffixes=["_end", ""]).drop(
        columns="temp"
    )
    df = df.drop(columns="temp")

    # remove rows after censoring or end time
    df_ = df_.loc[df_[time + "_end"] >= df_[time]].reset_index(drop=True)

    # add outcome
    df_[outcome] = np.where(
        df_[time + "_end"] == df_[time],
        df_[outcome],
        False if pd.api.types.is_bool_dtype(df_[time]) else 0,
    )

    return df_.drop(columns=time + "_end")
```

All that remains is to estimate a Poisson regression on this transformed dataset. Note that the formula here contains the week of the year as a categorical in addition to the regressors of the Cox model:

```python
df_split = survival_split(df, "week", "arrest")
model_glum = glum.GeneralizedLinearRegressor(
    family="poisson",
    formula="arrest ~ fin + bs(age, df=4) + race + wexp + mar + paro + bs(prio, df=3) + C(week)",
    fit_intercept=False,
).fit(df_split)
```

We can check that the Poisson regression yields estimates and standard errors that, for all practical purposes, are the same as those of the Cox regression:

```python
cph.summary.reset_index()[["covariate", "coef", "se(coef)"]].merge(
    model_glum.coef_table(X=df_split, robust=False)[["coef", "se"]],
    left_on="covariate",
    right_index=True,
).rename(
    columns={
        "coef_x": "coef_coxph",
        "coef_y": "coef_poisson",
        "se(coef)": "se_coxph",
        "se": "se_poisson",
    }
)
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>covariate</th>
      <th>coef_coxph</th>
      <th>se_coxph</th>
      <th>coef_poisson</th>
      <th>se_poisson</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fin</td>
      <td>-0.350001</td>
      <td>0.192611</td>
      <td>-0.349668</td>
      <td>0.192792</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bs(age, df=4)[1]</td>
      <td>-0.490560</td>
      <td>0.636493</td>
      <td>-0.488086</td>
      <td>0.637050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bs(age, df=4)[2]</td>
      <td>-1.807838</td>
      <td>0.840686</td>
      <td>-1.797258</td>
      <td>0.841228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bs(age, df=4)[3]</td>
      <td>-0.906026</td>
      <td>1.409066</td>
      <td>-0.905313</td>
      <td>1.410311</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bs(age, df=4)[4]</td>
      <td>-1.756215</td>
      <td>1.102792</td>
      <td>-1.747288</td>
      <td>1.103601</td>
    </tr>
    <tr>
      <th>5</th>
      <td>race</td>
      <td>0.356941</td>
      <td>0.310188</td>
      <td>0.356586</td>
      <td>0.310484</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wexp</td>
      <td>-0.093144</td>
      <td>0.215796</td>
      <td>-0.095144</td>
      <td>0.215862</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mar</td>
      <td>-0.334348</td>
      <td>0.385933</td>
      <td>-0.333904</td>
      <td>0.386238</td>
    </tr>
    <tr>
      <th>8</th>
      <td>paro</td>
      <td>-0.138912</td>
      <td>0.198157</td>
      <td>-0.138637</td>
      <td>0.198338</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bs(prio, df=3)[1]</td>
      <td>1.364030</td>
      <td>0.963899</td>
      <td>1.355261</td>
      <td>0.964604</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bs(prio, df=3)[2]</td>
      <td>-0.244846</td>
      <td>1.048481</td>
      <td>-0.234007</td>
      <td>1.048920</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bs(prio, df=3)[3]</td>
      <td>2.738914</td>
      <td>0.805202</td>
      <td>2.714382</td>
      <td>0.805035</td>
    </tr>
  </tbody>
</table>
</div>
</div>

## 3. Speed Considerations

The Poisson model estimates each time fixed effect parameter and, therefore, many more parameters than the Cox model. In our example, there are 61 coefficients in the Poisson model as opposed to 12 in the Cox model:

```python
len(cph.summary), len(model_glum.coef_)
```

```text
(12, 61)
```

One might, therefore, wonder if the Poisson approach is competitive in terms of estimation speed. For the dataset here, the Poisson approach, including the data transformation by `survival_split`, turns out to be faster than the Cox model. This speedup is aided by tabmat's optimizations for the high-dimensional `week` categorical.

```python
%timeit cox_model = CoxPHFitter().fit(df, duration_col='week', event_col='arrest', formula="fin + bs(age, df=4) + race + wexp + mar + paro + bs(prio,df=3)")
```

```text
27.9 ms ± 297 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
%timeit model_glum = glum.GeneralizedLinearRegressor(family="poisson",formula="arrest ~ fin + bs(age, df=4) + race + wexp + mar + paro + bs(prio,df=3) + C(week)").fit(survival_split(df, 'week', 'arrest'))
```

```text
25.1 ms ± 531 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Conclusion

This tutorial has shown how the Cox proportional hazards model can be reformulated as a Poisson regression on a transformed dataset, which can be run without specialized survival estimators. The Poisson regression involves estimating a high-dimensional categorical with one category per event time, for which glum's handling of categoricals is handy. This reformulation also allows for exploration beyond the Cox model, including the option to replace the piecewise constant baseline hazard with a smooth one, and to swap the Poisson objective for another, such as a binomial.

## Footnotes
<span id="fn1"> 1</span>:
The Cox model assumes that at most one individual has an event at any time ("no ties"). Different tie breaking methods available for datasets with duplicate event times. The most popular ones are an exact method, which is expensive to compute, Efron's method which is reasonably fast and accurate, and the Breslow approximation, which is the fastest. In the case of many ties, an inherently discrete survival model is probably the best option.

<span id="fn2"> 2</span>:
The log likelihood of an EDM with parameters $\theta_i$ is

$$
\sum_i\frac{y_i\theta_i - b(\theta_i)}{\sigma_i} + \text{constant},
$$

with cumulant function $b$ and dispersion parameter $\sigma_i$. The linear predictor is given by $\eta_i=g((b'(\theta_i))$ for link function $g$. The Cox partial log-likelihood cannot be broken down into such a sum over individuals $i$.

## References

[1] Carstensen, B. 2023. Who needs the Cox model anyway. December. Available at: http://bendixcarstensen.com/WntCma.pdf.

[2] Whitehead, J., 1980. Fitting Cox's regression model to survival data using GLIM. _Journal of the Royal Statistical Society Series C: Applied Statistics_, 29(3), pp.268-275.
