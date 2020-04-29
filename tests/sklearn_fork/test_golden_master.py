import numpy as np
import pytest

from glm_benchmarks.sklearn_fork._glm import GeneralizedLinearRegressor


@pytest.fixture
def poisson_reg_data():
    n_rows = 1000
    n_features = 10

    rand = np.random.default_rng(42)
    X = rand.standard_normal(size=(n_rows, n_features))
    coefs = np.array([1.0, 0.5, 0.1, -0.1, -0.5, -1.0, 0, 0, 0, 0])
    intercept = 0.2
    y = rand.poisson(np.exp(intercept + X @ coefs))
    weights = rand.uniform(size=n_rows)
    data = {"intercept": intercept, "X": X, "b": coefs, "y": y, "weights": weights}
    return data


def _make_P2():
    rand = np.random.default_rng(42)
    a = rand.uniform(size=(10, 10)) - 0.5  # centered uniform distribution
    P2 = a.T @ a  # make sure P2 is positive semi-definite
    return P2


@pytest.mark.parametrize(
    ["parameters", "expected"],
    zip(
        [
            {},  # default params
            {"alpha": 0},  # no-regularization
            {"alpha": 0.5},  # regularization (other than alpha = 1)
            {"l1_ratio": 0.5},  # elastic-net
            {"l1_ratio": 1},  # lasso
            {"l1_ratio": 1, "P1": np.arange(10)},  # lasso with variable penalty
            {"l1_ratio": 0, "P2": _make_P2()},  # ridge with Tikhonov regularization
            # elastic net with P1 and P2 variable penalty
            {"l1_ratio": 0.5, "P1": np.arange(10), "P2": _make_P2()},
            {"fit_intercept": False},  # do not fit the intercept
        ],
        [
            np.array(
                [
                    0.74246842,
                    0.40384742,
                    0.06631426,
                    -0.13836918,
                    -0.30609168,
                    -0.76019387,
                    0.03221517,
                    0.011959,
                    0.02966291,
                    0.00824136,
                ]
            ),
            np.array(
                [
                    0.99842983,
                    0.51260062,
                    0.10328809,
                    -0.13867686,
                    -0.43781645,
                    -0.99253102,
                    0.00510553,
                    0.01189323,
                    -0.03126507,
                    0.01618529,
                ]
            ),
            np.array(
                [
                    0.84508013,
                    0.44822589,
                    0.08039357,
                    -0.14109234,
                    -0.35675013,
                    -0.85340402,
                    0.02276335,
                    0.01242991,
                    0.00956274,
                    0.01279488,
                ]
            ),
            np.array(
                [
                    0.70630898,
                    0.34764296,
                    0.0,
                    -0.06360327,
                    -0.23552957,
                    -0.74153126,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.66081698,
                    0.27324762,
                    0.0,
                    0.0,
                    -0.14164699,
                    -0.7109816,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.98627342, 0.2869042, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [
                    0.7960021,
                    0.37712049,
                    0.08507042,
                    -0.13016336,
                    -0.28966785,
                    -0.72379518,
                    0.04472295,
                    0.03730088,
                    0.0420659,
                    -0.07125174,
                ]
            ),
            np.array(
                [0.86917307, 0.33610618, 0.0, 0.0, 0.0, -0.30775257, 0.0, 0.0, 0.0, 0.0]
            ),
            np.array(
                [
                    0.93456185,
                    0.5248033,
                    0.0824139,
                    -0.16104805,
                    -0.39707757,
                    -0.99017071,
                    0.02444319,
                    0.01501054,
                    -0.00637704,
                    0.00447107,
                ]
            ),
        ],
    ),
)
def test_poisson_golden_master(parameters, expected, poisson_reg_data):
    model = GeneralizedLinearRegressor(family="poisson", **parameters)

    model.fit(
        X=poisson_reg_data["X"],
        y=poisson_reg_data["y"],
        sample_weight=poisson_reg_data["weights"],
    )

    np.testing.assert_allclose(model.coef_, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    zip(
        [
            {},  # default params
            {"alpha": 0},  # no-regularization
            {"alpha": 0.5},  # regularization (other than alpha = 1)
            {"l1_ratio": 0.5},  # elastic-net
            {"l1_ratio": 1},  # lasso
            {"l1_ratio": 1, "P1": np.arange(10)},  # lasso with variable penalty
            {"l1_ratio": 0, "P2": _make_P2()},  # ridge with Tikhonov regularization
            # elastic net with P1 and P2 variable penalty
            {"l1_ratio": 0.5, "P1": np.arange(10), "P2": _make_P2()},
            {"fit_intercept": False},  # do not fit the intercept
        ],
        [
            np.array(
                [
                    0.7416926,
                    0.43263022,
                    0.06136691,
                    -0.12375087,
                    -0.33502745,
                    -0.7404941,
                    0.01458928,
                    0.00484578,
                    0.01996707,
                    0.01935922,
                ]
            ),
            np.array(
                [
                    1.00359154,
                    0.51284335,
                    0.10418622,
                    -0.12589639,
                    -0.45988364,
                    -0.98551971,
                    0.00259697,
                    0.00921262,
                    -0.01912348,
                    0.00655004,
                ]
            ),
            np.array(
                [
                    0.84671323,
                    0.46814922,
                    0.07732654,
                    -0.12736152,
                    -0.38418673,
                    -0.83937284,
                    0.01044215,
                    0.00625469,
                    0.00733929,
                    0.01612754,
                ]
            ),
            np.array(
                [
                    0.70410893,
                    0.39016393,
                    0.0,
                    -0.03933241,
                    -0.26840552,
                    -0.70695923,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.65708719,
                    0.33345358,
                    0.0,
                    0.0,
                    -0.17873681,
                    -0.66388454,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.95193168, 0.34808442, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [
                    0.79799482,
                    0.41655693,
                    0.07926552,
                    -0.10972668,
                    -0.32485778,
                    -0.70339345,
                    0.02521004,
                    0.0318932,
                    0.02996625,
                    -0.06292129,
                ]
            ),
            np.array(
                [0.84998681, 0.3970036, 0.0, 0.0, 0.0, -0.24422268, 0.0, 0.0, 0.0, 0.0]
            ),
            np.array(
                [
                    0.93291162,
                    0.5435488,
                    0.07059911,
                    -0.15057257,
                    -0.42508895,
                    -0.96735852,
                    0.00812953,
                    0.00910666,
                    0.00432619,
                    0.01610756,
                ]
            ),
        ],
    ),
)
def test_poisson_golden_master_noweights(parameters, expected, poisson_reg_data):
    model = GeneralizedLinearRegressor(family="poisson", **parameters)

    model.fit(
        X=poisson_reg_data["X"], y=poisson_reg_data["y"],
    )

    np.testing.assert_allclose(model.coef_, expected, rtol=1e-5, atol=0)
