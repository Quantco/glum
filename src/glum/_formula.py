from collections.abc import Mapping
from typing import Any, Optional

import formulaic


def capture_context(context) -> Optional[Mapping[str, Any]]:
    if isinstance(context, int):
        context = context + 2
    return formulaic.utils.context.capture_context(context)


def parse_formula(
    formula: formulaic.FormulaSpec, include_intercept: bool = True
) -> tuple[Optional[formulaic.Formula], formulaic.Formula]:
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
