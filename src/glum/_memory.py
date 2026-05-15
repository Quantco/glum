"""
Memory footprint estimator for glum fits.

A pure function that predicts peak RAM usage for a GLM fit given the
dataset shape and solver choice, decomposed into the contributing
terms (data storage, coefficient vector, IRLS transients, Hessian,
cache state).  Useful for capacity planning *before* kicking off a fit
on data you've never seen at scale.

The decomposition:

    M_total ≈ M_data + M_coef + M_irls_transient + M_hessian + M_cache

where the dominant term flips between ``M_data`` and ``M_hessian``
around ``p_expanded² ≈ n × p_num``.

See the validation tests under ``tests/glm/test_memory.py`` for
empirical calibration of these formulas against measured peak RSS.
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = ["estimate_memory"]


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# These come from the cost analysis in docs/stepwise.rst's caching section
# and are validated by the empirical calibration tests.  See those tests
# for residual analysis if you need to tune them.

# IRLS holds ~7 arrays of size n simultaneously inside a single outer
# iteration: eta, mu, dmu_deta, variance, working_response, working_weight,
# and one or two scratch buffers.  This count is conservative.
_IRLS_TRANSIENT_ARRAYS_PER_N = 7

# glum holds several O(p_expanded) arrays simultaneously (coef, coef_prev,
# gradient, P1, P2_diag) — the count of 5 is empirical from
# inspecting _irls_solver state.
_COEF_ARRAYS_PER_P = 5

# L-BFGS approximates the Hessian using m = 10 gradient pairs (default).
# Each pair is two p-vectors, so total is 2*m*p elements.
_LBFGS_HISTORY_M = 10

# Categorical metadata: a Python list of category strings.  Each entry
# costs roughly (string-header + content) bytes.  This is a wild estimate;
# calibration tests can refine it.
_BYTES_PER_CATEGORY_LEVEL = 64


def estimate_memory(
    n: int,
    p_num: int,
    p_cat_levels: Optional[Sequence[int]] = None,
    *,
    dtype_bytes: int = 8,
    solver: str = "irls-cd",
    n_folds: int = 0,
    has_weight: bool = False,
    has_offset: bool = False,
) -> dict:
    """
    Predict peak RAM usage for a glum fit.

    Decomposes the prediction into five terms so a caller can see which
    one dominates and where the model would gain or lose if dimensions
    change.

    Parameters
    ----------
    n : int
        Number of rows in the training data.
    p_num : int
        Number of numeric feature columns.
    p_cat_levels : sequence of int, optional
        Number of levels for each categorical feature column.  Pass
        ``[]`` (or omit) for no categoricals.  Each entry contributes
        one column to the design matrix's *stored* form but
        ``levels`` columns to the *expanded* form after one-hot.
    dtype_bytes : int, default 8
        Bytes per element for the design matrix.  8 = ``float64``
        (default for glum); 4 = ``float32`` (halves every memory term).
    solver : {"irls-cd", "irls-ls", "irls-ls-monotonic", "closed-form",
              "lbfgs", "trust-constr"}, default ``"irls-cd"``
        Which solver glum's ``auto`` would dispatch to (or you've
        forced).  Affects the Hessian term:
        IRLS variants allocate a full ``p_expanded × p_expanded``
        Hessian; L-BFGS allocates only ``2 m p`` history; trust-constr
        and closed-form are estimated as IRLS-equivalent.
    n_folds : int, default 0
        Number of fold matrices held by an associated
        :class:`~glum.TabmatCache`.  Each fold slice is roughly
        ``train_size / n`` times the data footprint.  0 means no
        cache.
    has_weight : bool, default False
        Whether ``sample_weight`` adds an O(n) array.
    has_offset : bool, default False
        Whether ``offset`` adds an O(n) array.

    Returns
    -------
    dict
        Keys:

        - ``data_MB``       — design matrix + y (+ weight/offset)
        - ``coef_MB``       — coefficient vector + O(p) bookkeeping
        - ``transient_MB``  — IRLS inner-loop O(n) scratch arrays
        - ``hessian_MB``    — Hessian storage for the chosen solver
        - ``cache_MB``      — optional fold-slice cache footprint
        - ``total_MB``      — sum of the above
        - ``p_expanded``    — design matrix column count after one-hot
        - ``bottleneck``    — which term is largest:
                              ``"data"``, ``"hessian"``, or ``"transient"``

    Examples
    --------
    >>> est = estimate_memory(n=678_013, p_num=5, p_cat_levels=[6, 11, 2, 22])
    >>> est["bottleneck"]
    'data'
    >>> est["total_MB"]     # doctest: +SKIP
    65.1

    A high-cardinality categorical flips the regime:

    >>> hot = estimate_memory(n=1_000_000, p_num=10, p_cat_levels=[50_000])
    >>> hot["bottleneck"]
    'hessian'

    Notes
    -----
    The formulas are calibrated against measured peak RSS via the
    opt-in ``pytest -m memory`` test suite (see
    ``tests/glm/test_memory.py``).  On a 2026 macOS workstation,
    typical predictions land at **0.5–0.8× of measured peak RSS** —
    i.e. the formula systematically *underestimates*.  This is
    because RSS includes overhead the formula doesn't model:

    * imported libraries (pandas, numpy, tabmat, glum themselves
      contribute ~150 MB before the user code runs);
    * memory allocator pool fragmentation;
    * OS page cache for the freshly-read parquet;
    * un-collected GC garbage.

    For capacity planning, treat the predicted ``total_MB`` as a
    **floor** and add a 50–100% safety margin.  The estimate is
    most accurate for *relative* comparisons (does workload A need
    more RAM than B?) and for identifying the binding constraint
    (data vs. Hessian vs. transient), which the ``bottleneck``
    field surfaces.
    """
    p_cat_levels = list(p_cat_levels) if p_cat_levels else []
    p_cat = len(p_cat_levels)
    p_expanded_one_hot = sum(p_cat_levels)
    p_expanded = p_num + p_expanded_one_hot

    # ── Data term ──────────────────────────────────────────────────────
    # Dense numeric columns: full float64 (or float32) array.
    bytes_numeric = n * p_num * dtype_bytes
    # Categorical codes: tabmat insists on int32 regardless of pandas dtype.
    bytes_cat_codes = n * p_cat * 4
    # Categorical level metadata: a Python list of strings per categorical.
    bytes_cat_meta = p_expanded_one_hot * _BYTES_PER_CATEGORY_LEVEL
    # y and (optional) weight, offset.
    bytes_target = n * dtype_bytes * (1 + int(has_weight) + int(has_offset))
    data = bytes_numeric + bytes_cat_codes + bytes_cat_meta + bytes_target

    # ── Coefficient + O(p) state ───────────────────────────────────────
    coef = p_expanded * dtype_bytes * _COEF_ARRAYS_PER_P

    # ── IRLS transient ─────────────────────────────────────────────────
    transient = n * dtype_bytes * _IRLS_TRANSIENT_ARRAYS_PER_N

    # ── Hessian (solver-dependent) ─────────────────────────────────────
    if solver in ("irls-cd", "irls-ls", "irls-ls-monotonic"):
        hessian = p_expanded * p_expanded * dtype_bytes
    elif solver == "closed-form":
        # One linear solve over the Gram matrix; same footprint.
        hessian = p_expanded * p_expanded * dtype_bytes
    elif solver == "lbfgs":
        # Only 2*m*p history vectors.
        hessian = 2 * _LBFGS_HISTORY_M * p_expanded * dtype_bytes
    elif solver == "trust-constr":
        # scipy.optimize approximate Hessian; comparable to IRLS.
        hessian = p_expanded * p_expanded * dtype_bytes
    else:
        # Unknown solver — assume worst case.
        hessian = p_expanded * p_expanded * dtype_bytes

    # ── Cache state (optional) ─────────────────────────────────────────
    # Each fold slice is approximately ((k-1)/k) × data for k-fold CV;
    # we approximate as 0.8 × data per fold.  This is a rough upper
    # bound; tabmat sharing means the actual cost is smaller for
    # dense numeric columns.
    cache = int(0.8 * data) * n_folds if n_folds else 0

    total = data + coef + transient + hessian + cache

    # ── Bottleneck identification ──────────────────────────────────────
    terms = {"data": data, "hessian": hessian, "transient": transient}
    bottleneck = max(terms, key=terms.get)

    return {
        "data_MB":      data      / 1e6,
        "coef_MB":      coef      / 1e6,
        "transient_MB": transient / 1e6,
        "hessian_MB":   hessian   / 1e6,
        "cache_MB":     cache     / 1e6,
        "total_MB":     total     / 1e6,
        "p_expanded":   p_expanded,
        "bottleneck":   bottleneck,
    }
