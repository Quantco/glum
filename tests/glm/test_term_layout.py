import time

import numpy as np
import pandas as pd
import pytest

from glum._glm import GeneralizedLinearRegressor
from glum._term_layout import TermLayout


def _slices_partition(layout: TermLayout) -> bool:
    """True iff layout slices form a contiguous, ordered partition of [0, n_coefs)."""
    expected_start = 0
    for term in layout:
        if term.start != expected_start or term.stop <= term.start:
            return False
        expected_start = term.stop
    return expected_start == layout.n_coefs


def test_consecutive_duplicate_term_names_collapse_into_one_slice():
    # Arrange
    term_names = ["x1", "c1", "c1", "c1", "x2"]

    # Act
    layout = TermLayout.from_term_names(term_names, fit_intercept=False)

    # Assert
    assert [t.name for t in layout] == ["x1", "c1", "x2"]
    assert layout["c1"].n_coefs == 3


def test_intercept_owns_first_coefficient_when_fit_intercept_true():
    # Arrange
    term_names = ["x1", "x2"]

    # Act
    layout = TermLayout.from_term_names(term_names, fit_intercept=True)

    # Assert
    assert layout.has_intercept is True
    assert "intercept" in layout
    assert layout["intercept"].start == 0
    assert layout["intercept"].stop == 1
    assert layout["x1"].start == 1


def test_layout_without_intercept_starts_at_index_zero():
    # Arrange
    term_names = ["x1", "x2"]

    # Act
    layout = TermLayout.from_term_names(term_names, fit_intercept=False)

    # Assert
    assert layout.has_intercept is False
    assert "intercept" not in layout
    assert layout["x1"].start == 0


def test_empty_term_names_with_intercept_yields_intercept_only_layout():
    # Arrange / Act
    layout = TermLayout.from_term_names([], fit_intercept=True)

    # Assert
    assert layout.n_coefs == 1
    assert [t.name for t in layout] == ["intercept"]


def test_empty_term_names_without_intercept_yields_empty_layout():
    # Arrange / Act
    layout = TermLayout.from_term_names([], fit_intercept=False)

    # Assert
    assert layout.n_coefs == 0
    assert list(layout) == []


@pytest.mark.parametrize(
    "term_names,fit_intercept,expected_n_coefs",
    [
        (["x1", "x2", "x3"], False, 3),
        (["x1", "x2", "x3"], True, 4),
        (["c1", "c1", "c1"], True, 4),
        (["a", "b", "b", "c", "c", "c"], False, 6),
    ],
)
def test_slices_partition_the_full_coefficient_vector(
    term_names, fit_intercept, expected_n_coefs
):
    # Arrange / Act
    layout = TermLayout.from_term_names(term_names, fit_intercept=fit_intercept)

    # Assert
    assert layout.n_coefs == expected_n_coefs
    assert _slices_partition(layout)


def test_lookup_by_name_returns_slice_covering_that_terms_coefficients():
    # Arrange
    term_names = ["x1", "c1", "c1", "c1", "x2"]
    layout = TermLayout.from_term_names(term_names, fit_intercept=True)

    # Act
    c1_slice = layout["c1"]

    # Assert: slice covers exactly the c1 entries in term_names (intercept offset 1)
    coef_indices = range(c1_slice.start, c1_slice.stop)
    assert [term_names[i - 1] for i in coef_indices] == ["c1", "c1", "c1"]


def test_lookup_by_unknown_name_raises_key_error():
    # Arrange
    layout = TermLayout.from_term_names(["x1"], fit_intercept=False)

    # Act / Assert
    with pytest.raises(KeyError):
        layout["does_not_exist"]


def test_membership_check_distinguishes_known_from_unknown_terms():
    # Arrange
    layout = TermLayout.from_term_names(["x1", "x2"], fit_intercept=True)

    # Assert (no separate Act; membership is the operation under test)
    assert "x1" in layout
    assert "intercept" in layout
    assert "x3" not in layout
    assert 42 not in layout  # non-string membership returns False


def test_iteration_yields_terms_in_coefficient_order():
    # Arrange
    term_names = ["a", "b", "b", "c"]
    layout = TermLayout.from_term_names(term_names, fit_intercept=True)

    # Act
    iterated = [(t.name, t.start) for t in layout]

    # Assert: each successive term starts at or after the previous one
    starts = [start for _, start in iterated]
    assert starts == sorted(starts)
    assert [name for name, _ in iterated] == ["intercept", "a", "b", "c"]


def test_term_slice_n_coefs_equals_stop_minus_start():
    # Arrange
    layout = TermLayout.from_term_names(["c1", "c1", "c1"], fit_intercept=False)

    # Act
    term = layout["c1"]

    # Assert
    assert term.n_coefs == term.stop - term.start


# --- Integration: end-to-end after .fit() ---------------------------------


def test_fit_layout_n_coefs_matches_total_coefficients():
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)

    # Act
    model = GeneralizedLinearRegressor(family="normal", alpha=0.1).fit(X, y)

    # Assert
    expected_total = (1 if model.fit_intercept else 0) + len(model.coef_)
    assert model.term_layout_.n_coefs == expected_total


def test_fit_without_intercept_layout_excludes_intercept_slice():
    # Arrange
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    y = rng.standard_normal(40)

    # Act
    model = GeneralizedLinearRegressor(
        family="normal", alpha=0.1, fit_intercept=False
    ).fit(X, y)

    # Assert
    assert "intercept" not in model.term_layout_
    assert model.term_layout_.has_intercept is False


def test_fit_layout_slices_partition_the_coefficient_vector():
    # Arrange
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(60),
            "x1": rng.standard_normal(60),
            "c1": pd.Categorical(rng.choice(["a", "b", "c"], 60)),
        }
    )

    # Act
    model = GeneralizedLinearRegressor(
        family="normal", formula="y ~ x1 + c1", alpha=0.1, drop_first=False
    ).fit(df)

    # Assert
    assert _slices_partition(model.term_layout_)


def test_fit_layout_term_lookup_yields_correct_design_matrix_columns():
    """The strongest behavioral test: layout slices must point at the right
    columns of the design matrix that produced ``coef_``.
    """
    # Arrange
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(80),
            "x1": rng.standard_normal(80),
            "c1": pd.Categorical(rng.choice(["a", "b", "c"], 80)),
        }
    )
    model = GeneralizedLinearRegressor(
        family="normal", formula="y ~ x1 + c1", alpha=0.1, drop_first=False
    ).fit(df)
    intercept_offset = 1 if model.fit_intercept else 0

    # Act: pick the categorical term and read back the column names it spans
    c1_slice = model.term_layout_["c1"]
    coef_indices = range(c1_slice.start, c1_slice.stop)
    feature_indices = [i - intercept_offset for i in coef_indices]
    spanned_features = [model.feature_names_[i] for i in feature_indices]

    # Assert: every spanned feature name carries the c1 prefix
    assert len(spanned_features) >= 2  # at least two levels remain
    assert all(name.startswith("c1") for name in spanned_features)


def test_fit_layout_is_deterministic_for_identical_fits():
    # Arrange
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 4))
    y = rng.standard_normal(30)

    # Act
    layout_a = (
        GeneralizedLinearRegressor(family="normal", alpha=0.1).fit(X, y).term_layout_
    )
    layout_b = (
        GeneralizedLinearRegressor(family="normal", alpha=0.1).fit(X, y).term_layout_
    )

    # Assert
    assert [t.name for t in layout_a] == [t.name for t in layout_b]
    assert [(t.start, t.stop) for t in layout_a] == [
        (t.start, t.stop) for t in layout_b
    ]


# --- Performance regression guard -----------------------------------------


def test_layout_construction_overhead_is_bounded():
    """Construction is O(n_terms) and runs once per .fit(). Lock in a generous
    upper bound so an accidental quadratic regression would fail loudly.
    """
    # Arrange
    n = 1000
    term_names = [f"x{i}" for i in range(n)]

    # Act
    start = time.perf_counter()
    for _ in range(100):
        TermLayout.from_term_names(term_names, fit_intercept=True)
    elapsed_per_build_ms = (time.perf_counter() - start) / 100 * 1000

    # Assert: 1000 single-coef terms should build in well under 5 ms
    assert elapsed_per_build_ms < 5.0, (
        f"TermLayout build regressed: {elapsed_per_build_ms:.2f} ms/build"
    )
