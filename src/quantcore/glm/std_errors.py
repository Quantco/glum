import logging

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.stats import norm

_logger = logging.getLogger(__name__)


def coefficient_table(names, coefficients, variance=None):
    """Calculate standard errors for generalized linear models.

    Parameters
    ----------
    names : array-like
        A vector of coefficient names.
    coefficients : array-like
        A vector of coefficients.
    variance : array-like, optional, default=None
        A matrix with the variance of each coefficient along the diagonal.
    """
    out = pd.DataFrame({"name": names, "coef": coefficients})

    if variance is not None:

        std_errors = np.sqrt(variance.diagonal())
        p_values = 2 * norm.cdf(-np.abs(coefficients / std_errors))

        stars = (
            (p_values < 0.05).astype(int)
            + (p_values < 0.01).astype(int)
            + (p_values < 0.001).astype(int)
        )

        out["std_errors"] = std_errors
        out["p_values"] = [
            format(p, ".2%") + " " + "*" * s + " " * (3 - s)
            for p, s in zip(p_values, stars)
        ]

    return out


def std_errors(*args, **kwargs):
    """Calculate standard errors for generalized linear models.

    Parameters
    ----------
    estimator : LogitRegressor or TweedieRegressor
        An estimator.
    X : pandas.DataFrame
        The design matrix.
    y : array-like
        Array with outcomes.
    mu : array-like, optional, default=None
        Array with predictions. Estimated if absent.
    offset : array-like, optional, default=None
        Array with additive offsets.
    sample_weight: array-like, optional, default=None
        Array with sampling weights.
    dispersion : float, optional, default=None
        The dispersion parameter. Estimated if absent.
    robust : boolean, optional, default=True
        Whether to compute robust standard errors instead of normal ones.
    expected_information : boolean, optional, default=False
        Whether to use the expected or observed information matrix.
        Only relevant when computing robust std-errors.
    """
    return np.sqrt(covariance_matrix(*args, **kwargs).diagonal())


def _group_sum(groups: np.ndarray, data: np.ndarray):
    out = np.empty((len(np.unique(groups)), data.shape[1]))
    for i in range(data.shape[1]):
        out[:, i] = np.bincount(x=groups, weights=data[:, i])
    return out


def covariance_matrix(
    estimator,
    X,
    y,
    mu=None,
    offset=None,
    sample_weight=None,
    dispersion=None,
    robust=True,
    clusters: np.ndarray = None,
    expected_information=False,
):
    """Calculate the covariance matrix for generalized linear models.

    Parameters
    ----------
    estimator : LogitRegressor or TweedieRegressor
        An estimator.
    X : pandas.DataFrame
        The design matrix.
    y : array-like
        Array with outcomes.
    mu : array-like, optional, default=None
        Array with predictions. Estimated if absent.
    offset : array-like, optional, default=None
        Array with additive offsets.
    sample_weight: array-like, optional, default=None
        Array with sampling weights.
    dispersion : float, optional, default=None
        The dispersion parameter. Estimated if absent.
    robust : boolean, optional, default=True
        Whether to compute robust standard errors instead of normal ones.
    expected_information : boolean, optional, default=False
        Whether to use the expected or observed information matrix.
        Only relevant when computing robust std-errors.
    """
    if sparse.issparse(X):
        _logger.warning(
            "Can't compute standard errors for sparse matrices. "
            "Please convert to dense."
        )
        return None

    sum_weights = len(X) if sample_weight is None else sample_weight.sum()
    mu = estimator.predict(X, offset=offset) if mu is None else np.asanyarray(mu)

    if dispersion is None:
        dispersion = estimator._family_instance.dispersion(
            y, mu, sample_weight, n_parameters=estimator.n_parameters
        )

    try:
        if robust or clusters is not None:
            if expected_information:
                oim = fisher_information(estimator, X, y, mu, sample_weight, dispersion)
            else:
                oim = observed_information(
                    estimator, X, y, mu, sample_weight, dispersion
                )
            gradient = score_matrix(estimator, X, y, mu, sample_weight, dispersion)
            vcov = linalg.solve(oim, linalg.solve(oim, gradient.T @ gradient).T)
            vcov *= sum_weights / (sum_weights - estimator.n_parameters)
        else:
            fisher = fisher_information(estimator, X, y, mu, sample_weight, dispersion)
            vcov = linalg.inv(fisher)
            vcov /= sum_weights - estimator.n_parameters

        return vcov

    except linalg.misc.LinAlgError:

        gradient = score_matrix(estimator, X, y, mu, sample_weight, dispersion)
        zero_gradient = np.abs(np.sum(gradient, axis=0)) < 1e-12

        if zero_gradient.any():

            if hasattr(estimator, "columns_"):
                if hasattr(estimator, "fit_intercept") and estimator.fit_intercept:
                    columns = np.array(["(INTERCEPT)"] + estimator.columns_)
                else:
                    columns = np.array(estimator.columns_)
            else:
                columns = np.arange(len(zero_gradient))

            _logger.warning(
                "Estimation of standard errors failed. Matrix is singular. "
                "The following coefficients have zero gradients: "
                f"{columns[zero_gradient]}"
            )

            return np.full((len(gradient), len(gradient)), np.nan)


def fisher_information(estimator, X, y, mu=None, sample_weight=None, dispersion=None):
    """Compute the expected information matrix.

    Parameters
    ----------
    estimator : sklearn.linear_model.GeneralizedLinearRegressor
        An estimator.
    X : pandas.DataFrame
        The design matrix.
    y : array-like
        Array with outcomes.
    mu : array-like, optional, default=None
        Array with predictions. Estimated if absent.
    sample_weight : array-like, optional, default=None
        Array with weights.
    dispersion : float, optional, default=None
        The dispersion parameter. Estimated if absent.
    """
    mu = estimator.predict(X) if mu is None else np.asanyarray(mu)

    if dispersion is None:
        dispersion = estimator._family_instance.dispersion(
            y, mu, sample_weight, n_parameters=estimator.n_parameters
        )

    sum_weights = len(X) if sample_weight is None else sample_weight.sum()

    W = estimator._link_instance.inverse_derivative(estimator._link_instance.link(mu))
    W **= 2
    W /= estimator._family_instance.unit_variance(mu)
    W /= dispersion * sum_weights

    if sample_weight is not None:
        W *= np.asanyarray(sample_weight)

    return _safe_sandwich_dot(np.asanyarray(X), W, estimator.fit_intercept)


def observed_information(estimator, X, y, mu=None, sample_weight=None, dispersion=None):
    """Compute the observed information matrix.

    Parameters
    ----------
    estimator : sklearn.linear_model.GeneralizedLinearRegressor
        An estimator.
    X : pandas.DataFrame
        The design matrix.
    y : array-like
        Array with outcomes.
    mu : array-like, optional, default=None
        Array with predictions. Estimated if absent.
    sample_weight : array-like, optional, default=None
        Array with weights.
    dispersion : float, optional, default=None
        The dispersion parameter. Estimated if absent.
    """
    mu = estimator.predict(X) if mu is None else np.asanyarray(mu)
    linpred = estimator._link_instance.link(mu)
    y = np.asanyarray(y)

    if dispersion is None:
        dispersion = estimator._family_instance.dispersion(
            y, mu, sample_weight, n_parameters=estimator.n_parameters
        )

    sum_weights = len(X) if sample_weight is None else sample_weight.sum()
    inv_unit_variance = 1 / estimator._family_instance.unit_variance(mu)
    temp = inv_unit_variance / (dispersion * sum_weights)

    if sample_weight is not None:
        temp *= np.asanyarray(sample_weight)

    dp = estimator._link_instance.inverse_derivative2(linpred)
    d2 = estimator._link_instance.inverse_derivative(linpred) ** 2
    v = estimator._family_instance.unit_variance_derivative(mu)
    v *= inv_unit_variance
    r = y - mu
    temp *= -dp * r + d2 * v * r + d2

    return _safe_sandwich_dot(np.asanyarray(X), temp, intercept=estimator.fit_intercept)


def score_matrix(estimator, X, y, mu=None, sample_weight=None, dispersion=None):
    """Compute the score.

    Parameters
    ----------
    estimator : LogitRegressor or TweedieRegressor
        An estimator.
    X : pandas.DataFrame
        The design matrix.
    y : array-like
        Array with outcomes.
    mu : array-like, optional, default=None
        Array with predictions. Estimated if absent.
    sample_weight: array-like, optional, default=None
        Array with sampling weights.
    dispersion : float, optional, default=None
        The dispersion parameter. Estimated if absent.
    """
    mu = estimator.predict(X) if mu is None else np.asanyarray(mu)
    linpred = estimator._link_instance.link(mu)
    y = np.asanyarray(y)
    X = np.asanyarray(X.todense()) if sparse.issparse(X) else np.asanyarray(X)

    if dispersion is None:
        dispersion = estimator._family_instance.dispersion(
            y, mu, sample_weight, n_parameters=estimator.n_parameters
        )

    sum_weights = len(X) if sample_weight is None else sample_weight.sum()
    W = 1 / estimator._family_instance.unit_variance(mu)
    W /= dispersion * sum_weights

    if sample_weight is not None:
        W *= np.asanyarray(sample_weight)

    W *= estimator._link_instance.inverse_derivative(linpred)
    W *= y - mu
    W = W.reshape(-1, 1)

    if estimator.fit_intercept:
        return np.hstack((W, np.multiply(X, W)))
    else:
        return np.multiply(X, W)


def _safe_sandwich_dot(X, d, intercept=False):
    """Compute sandwich product X.T @ diag(d) @ X."""
    temp = (X.T * d) @ X
    if intercept:
        dim = X.shape[1] + 1
        order = "F" if X.flags["F_CONTIGUOUS"] else "C"
        res = np.empty((dim, dim), dtype=max(X.dtype, d.dtype), order=order)
        res[0, 0] = d.sum()
        res[1:, 0] = d @ X
        res[0, 1:] = res[1:, 0]
        res[1:, 1:] = temp
    else:
        res = temp
    return res
