import numpy as np
import pandas as pd


def compute_zscore(df, window=100, min_periods=None, outcome="sales", id="store"):
    """Compute rolling zscore."""
    mu = (
        df[[id, outcome]]
        .groupby(id)[outcome]
        .rolling(window, min_periods=min_periods)
        .mean()
        .groupby(level=0)
        .shift()
        .droplevel(0)
    )
    sd = (
        df[[id, outcome]]
        .groupby(id)[outcome]
        .rolling(window, min_periods=min_periods)
        .std(ddof=0)
        .groupby(level=0)
        .shift()
        .droplevel(0)
    )
    return ((df[outcome] - mu) / sd).replace([np.inf, -np.inf], np.nan)


def compute_age_quantile(df, cuts=5):
    """Compute age quantiles."""
    cumcount = df.groupby("store")["date"].transform("cumcount")
    potential_age = (df["date"] - df["date"].min()).dt.days
    return pd.qcut(
        cumcount.where(potential_age > cumcount), cuts, duplicates="drop"
    ).cat.codes


def compute_competition_open(df):
    """Create a competitor open feature."""
    # competition open?
    missing_mask = (
        df["competition_open_since_year"].isna()
        | df["competition_open_since_month"].isna()
    )
    competition_open = (
        pd.to_datetime(
            df["competition_open_since_year"].astype("Int64").astype("str")
            + "-"
            + df["competition_open_since_month"]
            .astype("Int64")
            .astype(str)
            .str.zfill(2)
            + "-01",
            errors="coerce",
        )
        .le(df["date"])
        .astype(int)
    )
    return competition_open.where(~missing_mask).astype("category")


def compute_lags(df, outcome="sales", suffix="", offset=1, **kwargs):
    """Compute moving averages."""
    df[f"ewm_{outcome}_lag{suffix}"] = (
        df.assign(value=lambda x: x[outcome].where(x["open"].eq(1)))[["store", "value"]]
        .groupby("store")
        .ewm(**kwargs)
        .mean()["value"]
        .groupby(level=0)
        .shift(offset)
        .droplevel(0)
    ).to_numpy()

    df[f"ewm_{outcome}_lag{suffix}_day_of_week"] = np.nan
    for day_of_week in range(1, 8):
        roll = (
            df.assign(
                value=lambda x: x[outcome].where(
                    x["day_of_week"].eq(day_of_week) & x["open"].eq(1)
                )
            )[["store", "value"]]
            .groupby("store")
            .ewm(**kwargs)
            .mean()["value"]
            .groupby(level=0)
            .shift(offset)
            .droplevel(0)
        ).to_numpy()
        mask = df["day_of_week"].eq(day_of_week).to_numpy()
        df.loc[mask, f"ewm_{outcome}_lag{suffix}_day_of_week"] = roll[mask]

    df[f"ewm_{outcome}_lag{suffix}_month"] = np.nan
    for month in range(1, 13):
        roll = (
            df.assign(
                value=lambda x: x[outcome].where(x["month"].eq(month) & x["open"].eq(1))
            )[["store", "value"]]
            .groupby("store")
            .ewm(**kwargs)
            .mean()["value"]
            .groupby(level=0)
            .shift(offset)
            .droplevel(0)
        ).to_numpy()
        mask = df["month"].eq(month).to_numpy()
        df.loc[mask, f"ewm_{outcome}_lag{suffix}_month"] = roll[mask]

    return df


def compute_open_lead(df, shift=-1):
    """Compute leading indicator for open."""
    open_lead = df.groupby("store")["open"].shift(shift)
    return open_lead.combine_first(df["day_of_week"].ne(6).astype("double"))


def compute_open_lag(df, shift=1):
    """Compute lagged indicator for open."""
    open_lag = df.groupby("store")["open"].shift(shift)
    return open_lag.combine_first(df["day_of_week"].ne(1).astype("double"))


def compute_school_holiday_lead(df, shift=-1):
    """Compute leading indicator for school_holiday."""
    school_holiday_lead = df.groupby("store")["school_holiday"].shift(shift)
    return school_holiday_lead.fillna(0).astype("double")


def compute_school_holiday_lag(df, shift=1):
    """Compute lagged indicator for school_holiday."""
    school_holiday_lag = df.groupby("store")["school_holiday"].shift(shift)
    return school_holiday_lag.fillna(0).astype("double")


def compute_state_holiday_lead(df, shift=-1):
    """Compute leading indicator for state_holiday."""
    state_holiday_lead = df.groupby("store")["state_holiday"].shift(shift)
    return state_holiday_lead.fillna("0").astype(object)


def compute_state_holiday_lag(df, shift=1):
    """Compute lagged indicator for state_holiday."""
    state_holiday_lag = df.groupby("store")["state_holiday"].shift(shift)
    return state_holiday_lag.fillna("0").astype(object)


def compute_promo_lead(df, shift=-1):
    """Compute leading indicator for promo."""
    promo_lead = df.groupby("store")["promo"].shift(shift)
    return promo_lead.fillna(0)


def compute_promo_lag(df, shift=1):
    """Compute lagged indicator for promo."""
    promo_lag = df.groupby("store")["promo"].shift(shift)
    return promo_lag.fillna(0)


def compute_store_state_holiday(df):
    """Compute store state holiday interaction."""
    return df["store"].astype(str) + "_" + df["state_holiday"].ne("0").astype(str)


def compute_store_school_holiday(df):
    """Compute store school vacation interaction effect."""
    return df["store"].astype(str) + "_" + df["school_holiday"].astype(str)


def compute_store_year(df):
    """Compute store year fixed effect."""
    return df["store"].astype(str) + "_" + df["year"].astype(str)


def compute_store_month(df):
    """Compute store month fixed effect."""
    return df["store"].astype(str) + "_" + df["month"].astype(str)


def compute_store_day_of_week(df):
    """Compute store day of week fixed effect."""
    return df["store"].astype(str) + "_" + df["day_of_week"].astype(str)


def compute_lagged_ma(df, lag=48, window=48, min_periods=48):
    """Compute lagged moving average."""
    lagged_ma = (
        df[["store", "sales"]]
        .groupby("store")["sales"]
        .rolling(window, min_periods=min_periods)
        .sum()
        .groupby(level=0)
        .shift(lag)
        .droplevel(0)
    )
    lagged_open = (
        df[["store", "open"]]
        .groupby("store")["open"]
        .rolling(window, min_periods=min_periods)
        .sum()
        .groupby(level=0)
        .shift(lag)
        .droplevel(0)
    )
    return lagged_ma / lagged_open


def compute_lagged_ewma(
    df, lag=48, com=None, span=None, halflife=None, alpha=None, min_periods=50
):
    """Compute lagged exponentially weighted moving average."""
    lagged_ma = (
        df[["store", "sales"]]
        .assign(sales=lambda x: x["sales"].replace(0, np.nan))
        .groupby("store")["sales"]
        .ewm(
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            ignore_na=True,
        )
        .mean()
        .groupby(level=0)
        .shift(lag)
        .droplevel(0)
    )
    return lagged_ma


def apply_all_transformations(df):
    """Apply all feature transformations."""
    df["age_quantile"] = compute_age_quantile(df, 5)
    df["competition_open"] = compute_competition_open(df)
    df["count"] = df.groupby("store")[["date"]].transform("cumcount")
    df["open_lag_1"] = compute_open_lag(df)
    df["open_lag_2"] = compute_open_lag(df, 2)
    df["open_lag_3"] = compute_open_lag(df, 3)
    df["open_lead_1"] = compute_open_lead(df)
    df["open_lead_2"] = compute_open_lead(df, -2)
    df["open_lead_3"] = compute_open_lead(df, -3)
    df["promo_lag_1"] = compute_promo_lag(df)
    df["promo_lag_2"] = compute_promo_lag(df, 2)
    df["promo_lag_3"] = compute_promo_lag(df, 3)
    df["promo_lead_1"] = compute_promo_lead(df)
    df["promo_lead_2"] = compute_promo_lead(df, -2)
    df["promo_lead_3"] = compute_promo_lead(df, -3)
    df["school_holiday_lag_1"] = compute_school_holiday_lag(df)
    df["school_holiday_lag_2"] = compute_school_holiday_lag(df, 2)
    df["school_holiday_lag_3"] = compute_school_holiday_lag(df, 3)
    df["school_holiday_lead_1"] = compute_school_holiday_lead(df)
    df["school_holiday_lead_2"] = compute_school_holiday_lead(df, -2)
    df["school_holiday_lead_3"] = compute_school_holiday_lead(df, -3)
    df["state_holiday_lag_1"] = compute_state_holiday_lag(df)
    df["state_holiday_lag_2"] = compute_state_holiday_lag(df, 2)
    df["state_holiday_lag_3"] = compute_state_holiday_lag(df, 3)
    df["state_holiday_lead_1"] = compute_state_holiday_lead(df)
    df["state_holiday_lead_2"] = compute_state_holiday_lead(df, -2)
    df["state_holiday_lead_3"] = compute_state_holiday_lead(df, -3)
    df["store_day_of_week"] = compute_store_day_of_week(df)
    df["store_month"] = compute_store_month(df)
    df["store_school_holiday"] = compute_store_school_holiday(df)
    df["store_state_holiday"] = compute_store_state_holiday(df)
    df["store_year"] = compute_store_year(df)
    df["zscore"] = compute_zscore(df, window=150)

    return df
