import numpy as np
import pytest
from scipy import linalg

from glum._linalg import _solve_well_conditioned_psd


def test_solve_well_conditioned_psd_near_singular():
    # The Hessian tabmat 4.3.2 produces for a constant column next to the
    # intercept. It is not exactly singular, so a plain solve returns huge
    # coefficients instead of raising.
    hessian = np.array(
        [
            [1.0, -4.4e-16, -2.3e-17],
            [-4.4e-16, 2.0e-31, 6.8e-33],
            [-2.3e-17, 6.8e-33, 1.0],
        ]
    )
    rhs = np.array([0.5, 0.0, 0.3])
    assert _solve_well_conditioned_psd(hessian, rhs) is None


@pytest.mark.parametrize(
    "hessian",
    [
        # Exactly singular: duplicated column
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        # Not positive definite at all
        np.zeros((2, 2)),
    ],
    ids=["duplicated", "zero"],
)
def test_solve_well_conditioned_psd_singular(hessian):
    assert _solve_well_conditioned_psd(hessian, np.ones(2)) is None


def test_solve_well_conditioned_psd_well_conditioned():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    hessian = X.T @ X
    rhs = rng.normal(size=5)
    np.testing.assert_allclose(
        _solve_well_conditioned_psd(hessian, rhs), linalg.solve(hessian, rhs)
    )


def test_solve_well_conditioned_psd_empty():
    out = _solve_well_conditioned_psd(np.zeros((0, 0)), np.zeros(0))
    assert out.shape == (0,)
