from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import formulaic
import numpy as np


def capture_context(context) -> Mapping[str, Any] | None:
    if isinstance(context, int):
        context = context + 2
    return formulaic.utils.context.capture_context(context)


def parse_formula(
    formula: formulaic.FormulaSpec, include_intercept: bool = True
) -> tuple[formulaic.Formula | None, formulaic.Formula]:
    """
    Parse and transform the formula for use in a GeneralizedLinearRegressor.

    The left-hand side and right-hand side of the formula are separated. If an
    intercept is present, it will be removed from the right-hand side, and a
    boolean flag to indicate whether or not an intercept should be added to
    the model will be returned.

    Parameters
    ----------
    formula : formulaic.FormulaSpec
        The formula to parse.
    include_intercept: bool, default True
        Whether to include an intercept column.

    Returns
    -------
    tuple[formulaic.Formula, formulaic.Formula]
        The left-hand side and right-hand sides of the formula.
    """
    if isinstance(formula, str):
        terms = formulaic.parser.DefaultFormulaParser(
            include_intercept=include_intercept
        ).get_terms(formula)
    elif isinstance(formula, formulaic.Formula):
        terms = formula
    else:
        raise TypeError("formula must be a string or Formula object.")

    if hasattr(terms, "lhs"):
        lhs_terms = terms.lhs
        rhs_terms = terms.rhs
        if len(lhs_terms) != 1:
            msg = "formula must have exactly one term on the left-hand side."
            raise ValueError(msg)
    else:
        lhs_terms = None
        rhs_terms = terms

    return lhs_terms, rhs_terms


_FACTOR_RE = re.compile(
    r"""
    (?P<name>[^:\[\]]+)    # factor name — everything up to ':', '[', or ']'
    (?:\[(?P<idx>[^\]]*)\])?  # optional bracketed index
    """,
    re.VERBOSE,
)


def _parse_column_name(col: str, separator: str = ":") -> list[tuple[str, str | None]]:
    """Parse ``"bs(x, df=5)[3]:g[a]"`` into ``[('bs(x, df=5)', '3'), ('g', 'a')]``."""
    factors = []
    for part in col.split(separator):
        m = _FACTOR_RE.fullmatch(part.strip())
        if m is None:
            raise ValueError(f"Cannot parse factor {part!r} in column {col!r}")
        factors.append((m.group("name"), m.group("idx")))
    return factors


def _group_columns_by_factor(
    feature_names: Sequence[str],
    factor_name: str,
    separator: str = ":",
) -> dict[tuple[tuple[str, str | None], ...], list[tuple[int, str | None]]]:
    """Group columns containing *factor_name*, keyed by the other factors."""
    groups: dict[tuple[tuple[str, str | None], ...], list[tuple[int, str | None]]] = (
        defaultdict(list)
    )

    for col_idx, col in enumerate(feature_names):
        factors = _parse_column_name(col, separator=separator)
        matched = [
            (i, name, idx)
            for i, (name, idx) in enumerate(factors)
            if name == factor_name
        ]
        if not matched:
            continue
        if len(matched) > 1:
            raise ValueError(
                f"Factor {factor_name!r} appears more than once in "
                f"column {col!r}. Self-interactions are not supported."
            )
        pos, _, factor_idx = matched[0]
        others = tuple((name, idx) for j, (name, idx) in enumerate(factors) if j != pos)
        groups[others].append((col_idx, factor_idx))

    return dict(groups)


def _sort_key(member: tuple[int, str | None]) -> tuple[float, str]:
    """Sort key for (col_idx, factor_idx): numeric first, then lexicographic."""
    _, idx = member
    if idx is None:
        return (0.0, "")
    try:
        return (float(idx), "")
    except ValueError:
        return (float("inf"), idx)


def _build_monotonic_constraints(
    feature_names: Sequence[str],
    monotonic: Mapping[str, str] | Sequence[tuple[str, str]],
    n_features: int | None = None,
    *,
    separator: str = ":",
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(A_ineq, b_ineq)`` enforcing monotonicity on ordered terms.

    Multi-column terms (splines, ordered categoricals) get consecutive-pair
    constraints; single-column terms get a sign constraint.  Interactions
    are constrained within each level of the other factor(s).
    """
    if n_features is None:
        n_features = len(feature_names)

    if isinstance(monotonic, Mapping):
        monotonic = list(monotonic.items())

    rows: list[np.ndarray] = []

    for factor_name, direction in monotonic:
        if direction not in ("increasing", "decreasing"):
            raise ValueError(
                f"Direction must be 'increasing' or 'decreasing'; "
                f"got {direction!r} for factor {factor_name!r}."
            )

        groups = _group_columns_by_factor(
            feature_names, factor_name, separator=separator
        )
        if not groups:
            raise ValueError(f"Factor {factor_name!r} not found in feature_names.")

        sign = 1.0 if direction == "increasing" else -1.0

        for group_key, members in groups.items():
            if len(members) > 1:
                members.sort(key=_sort_key)
            if len(members) == 1:
                row = np.zeros(n_features)
                row[members[0][0]] = -sign
                rows.append(row)
            else:
                for (idx_i, _), (idx_j, _) in zip(members[:-1], members[1:]):
                    row = np.zeros(n_features)
                    row[idx_i] = sign
                    row[idx_j] = -sign
                    rows.append(row)

    A_ineq = np.vstack(rows)
    b_ineq = np.zeros(A_ineq.shape[0])
    return A_ineq, b_ineq


def _resolve_monotonic_constraints_from_model_spec(
    model_spec,
    monotonic_constraints: Mapping[str, str],
) -> dict[str, str]:
    """Map variable names (e.g. ``"x"``) to factor names (e.g. ``"bs(x, df=5)"``)."""
    _INCOMPATIBLE_TRANSFORMS = ("poly(", "cc(")

    var_to_factors: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for _term, scoped_terms, column_names in model_spec.structure:
        for st in scoped_terms:
            for f in st.factors:
                factor_name = str(f)
                data_vars = {
                    str(v)
                    for v in f.factor.variables
                    if hasattr(v, "source") and v.source == "data"
                }
                n_cols = len(column_names)
                for dv in data_vars:
                    var_to_factors[dv].append((factor_name, n_cols))

    result: dict[str, str] = {}

    for var_name, direction in monotonic_constraints.items():
        if var_name not in var_to_factors:
            raise ValueError(
                f"Variable {var_name!r} specified in monotonic_constraints "
                f"was not found in any term of the formula."
            )
        for factor_name, n_cols in var_to_factors[var_name]:
            for prefix in _INCOMPATIBLE_TRANSFORMS:
                if factor_name.startswith(prefix):
                    raise ValueError(
                        f"Monotonic constraints are not supported for {factor_name!r}."
                    )
            result[factor_name] = direction

    return result
