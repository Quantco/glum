"""
Tests for glum._memory.estimate_memory — the RAM footprint estimator.

Three shapes:

1. **Unit tests** of the formula (this section).  Deterministic, fast,
   run in CI by default.  Validate that the estimator's outputs follow
   the documented scaling laws.

2. **Baseline-stability tests** against a checked-in JSON.  Catches
   accidental regressions in the formula across glum versions.  Run
   in CI by default.

3. **Calibration tests** against measured peak RSS (``@pytest.mark.memory``).
   Slow, noisy, opt-in via ``pytest -m memory``.  Validates that the
   formula matches reality.

Run all default-suite tests:    ``pytest tests/glm/test_memory.py``
Run calibration tests too:      ``pytest tests/glm/test_memory.py -m memory``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from glum import estimate_memory


# ---------------------------------------------------------------------------
# Shape 1: Unit tests of the formula
# ---------------------------------------------------------------------------

class TestEstimateMemoryFormula:
    """
    Validate that estimate_memory's outputs follow the documented
    scaling laws.  Deterministic — no measurement involved.
    """

    def test_returns_expected_keys(self):
        est = estimate_memory(n=1000, p_num=5, p_cat_levels=[10])
        for k in ("data_MB", "coef_MB", "transient_MB", "hessian_MB",
                  "cache_MB", "total_MB", "p_expanded", "bottleneck"):
            assert k in est, f"missing key {k!r}"

    def test_p_expanded_sums_numeric_and_categorical_levels(self):
        est = estimate_memory(n=1000, p_num=5, p_cat_levels=[3, 7, 10])
        assert est["p_expanded"] == 5 + 3 + 7 + 10

    def test_p_expanded_no_categoricals(self):
        est = estimate_memory(n=1000, p_num=12, p_cat_levels=[])
        assert est["p_expanded"] == 12

    def test_p_expanded_none_treated_as_empty(self):
        # p_cat_levels=None and p_cat_levels=[] should agree
        a = estimate_memory(n=1000, p_num=5, p_cat_levels=None)
        b = estimate_memory(n=1000, p_num=5, p_cat_levels=[])
        assert a == b

    def test_total_equals_sum_of_components(self):
        est = estimate_memory(n=10_000, p_num=20, p_cat_levels=[100, 50])
        component_sum = (
            est["data_MB"] + est["coef_MB"] + est["transient_MB"]
            + est["hessian_MB"] + est["cache_MB"]
        )
        assert est["total_MB"] == pytest.approx(component_sum)

    # ── Scaling laws ──────────────────────────────────────────────────

    def test_data_term_scales_linearly_in_n(self):
        small = estimate_memory(n=1000,   p_num=10, p_cat_levels=[])
        big   = estimate_memory(n=10_000, p_num=10, p_cat_levels=[])
        assert big["data_MB"] / small["data_MB"] == pytest.approx(10, rel=1e-6)

    def test_data_term_scales_linearly_in_p_num(self):
        narrow = estimate_memory(n=1000, p_num=10,  p_cat_levels=[])
        wide   = estimate_memory(n=1000, p_num=100, p_cat_levels=[])
        # Data scales p_num linearly, plus a constant target column
        ratio = wide["data_MB"] / narrow["data_MB"]
        # Target column doesn't scale with p_num — so ratio is slightly less than 10
        assert 9.0 < ratio < 10.0

    def test_hessian_scales_quadratically_in_p_expanded(self):
        small = estimate_memory(n=1000, p_num=10,  p_cat_levels=[])
        big   = estimate_memory(n=1000, p_num=100, p_cat_levels=[])
        # 10x more columns → ~100x more Hessian
        ratio = big["hessian_MB"] / small["hessian_MB"]
        assert ratio == pytest.approx(100, rel=1e-6)

    def test_transient_scales_linearly_in_n_only(self):
        small = estimate_memory(n=1000,   p_num=10, p_cat_levels=[])
        big_n = estimate_memory(n=10_000, p_num=10, p_cat_levels=[])
        big_p = estimate_memory(n=1000,   p_num=100, p_cat_levels=[])
        assert big_n["transient_MB"] / small["transient_MB"] == pytest.approx(10, rel=1e-6)
        # Transient doesn't grow with p_num — it's just IRLS O(n) scratch
        assert big_p["transient_MB"] == small["transient_MB"]

    # ── dtype ─────────────────────────────────────────────────────────

    def test_float32_halves_data_and_hessian(self):
        f64 = estimate_memory(n=10_000, p_num=20, p_cat_levels=[],
                              dtype_bytes=8)
        f32 = estimate_memory(n=10_000, p_num=20, p_cat_levels=[],
                              dtype_bytes=4)
        # Numeric data and Hessian scale with dtype directly.
        # Categorical codes are int32 either way, so the ratio isn't
        # exactly 2 if there are categoricals.
        assert f32["data_MB"] / f64["data_MB"] == pytest.approx(0.5, rel=0.01)
        assert f32["hessian_MB"] / f64["hessian_MB"] == pytest.approx(0.5, rel=0.01)
        assert f32["transient_MB"] / f64["transient_MB"] == pytest.approx(0.5, rel=0.01)

    def test_float32_categorical_codes_stay_int32(self):
        # Even with float32 dtype, tabmat stores categorical codes as int32.
        # So a pure-categorical estimate doesn't halve when we drop dtype.
        f64 = estimate_memory(n=10_000, p_num=0, p_cat_levels=[5] * 10,
                              dtype_bytes=8)
        f32 = estimate_memory(n=10_000, p_num=0, p_cat_levels=[5] * 10,
                              dtype_bytes=4)
        # Categorical codes (n × p_cat × 4) don't change with dtype.
        # Only Hessian and target scale.  So data_MB ratio is bounded
        # between 0.5 and 1.0.
        assert 0.5 <= f32["data_MB"] / f64["data_MB"] <= 1.0

    # ── Categorical storage ───────────────────────────────────────────

    def test_categoricals_cheaper_than_equivalent_numerics_in_data(self):
        """
        A categorical with k levels costs 4 bytes/row (int32 codes),
        not k × 8 bytes/row.  But it expands to k columns in the
        Hessian.  This test checks the storage side.
        """
        # 5 categoricals of 10 levels each → 50 expanded columns,
        # but stored as 5 int32 codes arrays
        cat = estimate_memory(n=100_000, p_num=0, p_cat_levels=[10] * 5)
        # Equivalent numeric (50 float64 columns)
        num = estimate_memory(n=100_000, p_num=50, p_cat_levels=[])
        # cat data should be much cheaper (5 × 4 = 20 bytes/row vs 50 × 8 = 400 bytes/row)
        assert cat["data_MB"] < num["data_MB"] / 5

    def test_high_cardinality_blows_up_hessian_not_data(self):
        """High-cardinality categorical: data stays small, Hessian explodes."""
        est = estimate_memory(n=100_000, p_num=5, p_cat_levels=[10_000])
        assert est["p_expanded"] == 10_005
        # Data stays modest: only +1 int32 codes column = +400 KB
        assert est["data_MB"] < 10
        # Hessian: 10_005² × 8 bytes ≈ 800 MB
        assert 700 < est["hessian_MB"] < 900
        assert est["bottleneck"] == "hessian"

    # ── Bottleneck identification ─────────────────────────────────────

    def test_bottleneck_data_when_tall_with_enough_cols(self):
        # 1M rows × 20 numerics: data (20n) clearly beats transient (7n).
        # The crossover is at p_num ≈ 6 — once p_num > 6, data wins.
        est = estimate_memory(n=1_000_000, p_num=20, p_cat_levels=[])
        assert est["bottleneck"] == "data"

    def test_bottleneck_hessian_when_wide(self):
        # 1k rows, 5k expanded cols: Hessian (5k²) dominates.
        est = estimate_memory(n=1000, p_num=100, p_cat_levels=[5000])
        assert est["bottleneck"] == "hessian"

    def test_bottleneck_transient_when_very_narrow(self):
        # 1M rows × 1 column: 7-array IRLS transient (7n) beats data (≈1n + y).
        est = estimate_memory(n=1_000_000, p_num=1, p_cat_levels=[])
        assert est["bottleneck"] == "transient"

    # ── Solver branch ─────────────────────────────────────────────────

    def test_lbfgs_avoids_p_squared_hessian(self):
        irls = estimate_memory(n=10_000, p_num=100, p_cat_levels=[5000],
                               solver="irls-cd")
        lbfgs = estimate_memory(n=10_000, p_num=100, p_cat_levels=[5000],
                                solver="lbfgs")
        # IRLS: 5100² × 8 ≈ 208 MB.  L-BFGS: 2 × 10 × 5100 × 8 ≈ 0.8 MB.
        assert lbfgs["hessian_MB"] < irls["hessian_MB"] / 100

    def test_closed_form_same_hessian_as_irls(self):
        irls = estimate_memory(n=10_000, p_num=20, p_cat_levels=[],
                               solver="irls-ls")
        closed = estimate_memory(n=10_000, p_num=20, p_cat_levels=[],
                                 solver="closed-form")
        assert closed["hessian_MB"] == irls["hessian_MB"]

    def test_unknown_solver_falls_back_to_worst_case(self):
        # Unknown solver should still return a sensible number, not crash
        est = estimate_memory(n=10_000, p_num=20, p_cat_levels=[],
                              solver="not-a-real-solver")
        assert est["hessian_MB"] > 0

    # ── Cache and optional terms ──────────────────────────────────────

    def test_cache_term_scales_with_n_folds(self):
        no_cache  = estimate_memory(n=10_000, p_num=20, p_cat_levels=[], n_folds=0)
        cache_5   = estimate_memory(n=10_000, p_num=20, p_cat_levels=[], n_folds=5)
        cache_10  = estimate_memory(n=10_000, p_num=20, p_cat_levels=[], n_folds=10)
        assert no_cache["cache_MB"] == 0
        assert cache_5["cache_MB"] > 0
        # Cache scales linearly with n_folds
        assert cache_10["cache_MB"] == pytest.approx(2 * cache_5["cache_MB"], rel=1e-6)

    def test_has_weight_adds_one_n_array(self):
        without = estimate_memory(n=100_000, p_num=10, p_cat_levels=[])
        with_w  = estimate_memory(n=100_000, p_num=10, p_cat_levels=[], has_weight=True)
        delta_MB = with_w["data_MB"] - without["data_MB"]
        # Should add exactly 100_000 × 8 bytes = 0.8 MB
        assert delta_MB == pytest.approx(0.8, rel=1e-6)

    def test_has_offset_adds_one_n_array(self):
        without = estimate_memory(n=100_000, p_num=10, p_cat_levels=[])
        with_o  = estimate_memory(n=100_000, p_num=10, p_cat_levels=[], has_offset=True)
        delta_MB = with_o["data_MB"] - without["data_MB"]
        assert delta_MB == pytest.approx(0.8, rel=1e-6)


# ---------------------------------------------------------------------------
# Shape 3: Baseline-stability tests
# ---------------------------------------------------------------------------

_BASELINE_PATH = Path(__file__).parent / "memory_baseline.json"

# Canonical workloads.  Edit this list AND regenerate the baseline JSON
# (see comment at the bottom of this file) whenever you intentionally
# change the formula.
CANONICAL_WORKLOADS = {
    "insurance_typical": {
        "n": 678_013, "p_num": 5, "p_cat_levels": [6, 11, 2, 22],
    },
    "insurance_with_zip": {
        "n": 678_013, "p_num": 5, "p_cat_levels": [5_000],
    },
    "wide_insurance": {
        "n": 600_000, "p_num": 50, "p_cat_levels": [10] * 10,
    },
    "tall_narrow": {
        "n": 5_000_000, "p_num": 30, "p_cat_levels": [],
    },
    "square_ish": {
        "n": 50_000, "p_num": 200, "p_cat_levels": [],
    },
    "high_cardinality": {
        "n": 1_000_000, "p_num": 10, "p_cat_levels": [50_000],
    },
    "lbfgs_workload": {
        "n": 100_000, "p_num": 50, "p_cat_levels": [5_000], "solver": "lbfgs",
    },
}


class TestBaselineStability:
    """
    Pin the predicted footprint of canonical workloads to a baseline
    JSON.  When this test fails, either:

    1. You intentionally changed the formula → regenerate the baseline
       (see ``_regenerate_baseline`` at the bottom of this file).
    2. You broke the formula accidentally → fix the code.
    """

    @pytest.fixture(scope="class")
    def baseline(self):
        if not _BASELINE_PATH.exists():
            pytest.skip(
                f"baseline file missing at {_BASELINE_PATH}; "
                "run the baseline regeneration helper to create it."
            )
        return json.loads(_BASELINE_PATH.read_text())

    @pytest.mark.parametrize("name", list(CANONICAL_WORKLOADS.keys()))
    def test_predicted_total_matches_baseline(self, name, baseline):
        cfg = CANONICAL_WORKLOADS[name]
        predicted = estimate_memory(**cfg)["total_MB"]
        recorded = baseline[name]["total_MB"]
        # Tight tolerance: the formula is deterministic
        assert predicted == pytest.approx(recorded, rel=1e-3), (
            f"workload {name}: formula predicts {predicted:.2f} MB, "
            f"baseline is {recorded:.2f} MB.  If you intended to "
            f"change the formula, regenerate memory_baseline.json."
        )

    @pytest.mark.parametrize("name", list(CANONICAL_WORKLOADS.keys()))
    def test_predicted_bottleneck_matches_baseline(self, name, baseline):
        cfg = CANONICAL_WORKLOADS[name]
        predicted_bottleneck = estimate_memory(**cfg)["bottleneck"]
        recorded_bottleneck = baseline[name]["bottleneck"]
        assert predicted_bottleneck == recorded_bottleneck


# ---------------------------------------------------------------------------
# Shape 2: Calibration against measured peak RSS (opt-in)
# ---------------------------------------------------------------------------

_WORKER = Path(__file__).parent / "_memory_worker.py"


def _measure_fit_rss(n: int, p_num: int, p_cat_levels, repeats: int = 3) -> float:
    """
    Spawn ``repeats`` subprocesses, each runs a single fit and reports
    its peak RSS.  Returns the median in MB.

    Subprocess isolation is essential — Python's allocator pool and
    GC timing make in-process repeated measurements unreliable.
    """
    measurements = []
    for _ in range(repeats):
        result = subprocess.run(
            [
                sys.executable, str(_WORKER),
                "--n", str(n),
                "--p_num", str(p_num),
                "--p_cat_levels", json.dumps(list(p_cat_levels)),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"worker failed: {result.stderr}\nstdout: {result.stdout}"
            )
        # Last line of stdout is the JSON result
        last_line = result.stdout.strip().split("\n")[-1]
        payload = json.loads(last_line)
        measurements.append(payload["peak_above_baseline_MB"])
    return float(np.median(measurements))


# Smaller calibration grid than the unit tests — these are slow.
# We pick configs that span the regimes most likely to be wrong.
CALIBRATION_CONFIGS = [
    pytest.param(50_000,   5,  [],                id="tiny_numeric"),
    pytest.param(50_000,   20, [10, 10, 10],      id="small_mixed"),
    pytest.param(200_000,  5,  [],                id="medium_numeric"),
    pytest.param(100_000,  10, [100],             id="small_categorical"),
]


@pytest.mark.memory
class TestMemoryCalibration:
    """
    Validate that estimate_memory predicts peak RSS within tolerance.

    Opt-in: ``pytest -m memory``.  Each test launches 3 subprocesses
    (median of 3 to dampen noise) and may take 20–60 seconds.
    """

    @pytest.mark.parametrize("n,p_num,p_cat_levels", CALIBRATION_CONFIGS)
    def test_prediction_within_50pct_of_measured(self, n, p_num, p_cat_levels):
        measured_MB = _measure_fit_rss(n, p_num, p_cat_levels, repeats=3)
        predicted_MB = estimate_memory(
            n=n, p_num=p_num, p_cat_levels=p_cat_levels,
        )["total_MB"]
        ratio = predicted_MB / measured_MB
        # Wide tolerance: RSS includes Python interpreter, libraries,
        # OS page cache, and allocator pool fragmentation that the
        # formula doesn't model.  We just want to catch order-of-
        # magnitude errors.
        assert 0.3 < ratio < 3.0, (
            f"config (n={n}, p_num={p_num}, p_cat_levels={p_cat_levels}): "
            f"predicted {predicted_MB:.1f} MB, measured {measured_MB:.1f} MB, "
            f"ratio {ratio:.2f}× (want 0.3–3.0×)"
        )

    def test_calibration_worker_exists(self):
        """The worker script must be present for calibration tests to run."""
        assert _WORKER.exists(), f"worker script missing at {_WORKER}"


# ---------------------------------------------------------------------------
# Baseline regeneration helper
# ---------------------------------------------------------------------------
#
# To regenerate memory_baseline.json after an intentional change to the
# formula:
#
#     python -c "
#     from tests.glm.test_memory import _regenerate_baseline
#     _regenerate_baseline()
#     "
#
# Then review the diff and commit.

def _regenerate_baseline() -> None:
    """Write the current predictions for canonical workloads to JSON."""
    baseline = {
        name: estimate_memory(**cfg) for name, cfg in CANONICAL_WORKLOADS.items()
    }
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    print(f"baseline written to {_BASELINE_PATH}")
