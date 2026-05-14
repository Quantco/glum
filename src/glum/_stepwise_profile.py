"""
Profiling & validation harness for StepwiseGLM.

Compares four variants across 10 stepwise steps on a mixed DataFrame
(numeric + categorical + sparse, n=50k):

  Baseline  — vanilla glum, raw DataFrame, cold start
  Opt-1     — pre-built tabmat via from_pandas, cold start
  Opt-2     — pre-built tabmat + warm_start
  Opt-3     — StepwiseGLM v1 (tabmat + warm_start + std cache)
  Opt-4     — StepwiseGLM v2 (hstack column cache + warm_start + std cache)

Then validates the score-test screener:
  - Correctly ranks a true-signal column above noise columns
  - Matches full-refit deviance ordering across 8 candidates
  - Reports timing: score-test screening vs full IRLS refits
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd
import tabmat as tm

import glum
from glum._stepwise import StepwiseGLM

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_dataset(n=50_000, p_num=20, p_cat=6, p_sparse=6, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((n, p_num)),
        columns=[f"num_{i}" for i in range(p_num)],
    )
    for i in range(p_cat):
        df[f"cat_{i}"] = pd.Categorical(rng.choice(list("abcde"), n))
    for i in range(p_sparse):
        vals = np.where(rng.random(n) < 0.05, rng.standard_normal(n), 0.0)
        df[f"sparse_{i}"] = pd.arrays.SparseArray(vals)
    mu = np.exp(0.3 + 0.15 * df["num_0"] - 0.1 * df["num_1"])
    y = rng.poisson(mu).astype(float)
    return df, y


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def _run_timed(fit_fn, step_cols):
    times = []
    for cols in step_cols:
        t0 = time.perf_counter()
        fit_fn(cols)
        times.append(time.perf_counter() - t0)
    return times


def _run_profiled(fit_fn, step_cols):
    pr = cProfile.Profile()
    times = []
    pr.enable()
    for cols in step_cols:
        t0 = time.perf_counter()
        fit_fn(cols)
        times.append(time.perf_counter() - t0)
    pr.disable()
    return times, pr


def _pr_str(pr, n=200):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(n)
    return s.getvalue()


def _bucket(pr_str, total):
    buckets = {
        "from_pandas / as_tabmat":  ["from_df", "as_tabmat", "from_pandas"],
        "pandas interleave":        ["_interleave", "as_array"],
        "standardize":              ["standardize"],
        "hstack":                   ["hstack"],
        "sandwich / hessian":       ["_safe_sandwich_dot", "update_hessian", "build_hessian_delta"],
        "categorical ops":          ["_cross_sandwich", "_cross_categorical", "categorical_matrix"],
        "transpose_matvec":         ["transpose_matvec"],
        "astype":                   ["astype"],
    }
    lines = pr_str.split("\n")

    def get_ct(kw):
        best = 0.0
        for line in lines:
            if kw in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        best = max(best, float(parts[3]))
                    except ValueError:
                        pass
        return best

    return [(label, max(get_ct(k) for k in keys), total)
            for label, keys in buckets.items()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_steps=10, n=50_000):
    print("=" * 72)
    print("StepwiseGLM profiling harness")
    print("=" * 72)

    df, y = make_dataset(n=n)
    all_cols = list(df.columns)
    step_cols = [all_cols[: 10 + i * 4] for i in range(n_steps)]
    GLM = dict(family="poisson", alpha=0.01)

    # ── Baseline ─────────────────────────────────────────────────────────
    print("\n[1/5] BASELINE  (DataFrame, cold start)")
    glm_base = glum.GeneralizedLinearRegressor(**GLM, warm_start=False)
    times_b, pr_b = _run_profiled(lambda c: glm_base.fit(df[c], y), step_cols)
    total_b = sum(times_b)
    pr_b_str = _pr_str(pr_b)
    print(f"  Total {total_b:.3f}s | Per-step {np.mean(times_b)*1e3:.1f}ms")

    # ── Opt-1: from_pandas per step, cold ─────────────────────────────────
    print("\n[2/5] OPT-1     (from_pandas each step, cold start)")
    glm_o1 = glum.GeneralizedLinearRegressor(**GLM, warm_start=False)

    def o1(cols):
        glm_o1.fit(tm.from_pandas(df[cols]), y)

    times_1, pr_1 = _run_profiled(o1, step_cols)
    total_1 = sum(times_1)
    pr_1_str = _pr_str(pr_1)
    print(f"  Total {total_1:.3f}s | Per-step {np.mean(times_1)*1e3:.1f}ms")

    # ── Opt-2: from_pandas per step, warm ─────────────────────────────────
    print("\n[3/5] OPT-2     (from_pandas each step, warm_start)")
    glm_o2 = glum.GeneralizedLinearRegressor(**GLM, warm_start=True)

    def o2(cols):
        X = tm.from_pandas(df[cols])
        if hasattr(glm_o2, "coef_") and len(glm_o2.coef_) != X.shape[1]:
            del glm_o2.coef_
        glm_o2.fit(X, y)

    times_2, pr_2 = _run_profiled(o2, step_cols)
    total_2 = sum(times_2)
    pr_2_str = _pr_str(pr_2)
    print(f"  Total {total_2:.3f}s | Per-step {np.mean(times_2)*1e3:.1f}ms")

    # ── Opt-3: StepwiseGLM (std cache, warm, from_pandas inside) ─────────
    print("\n[4/5] OPT-3     (StepwiseGLM: warm + std cache + from_pandas)")
    sglm3 = StepwiseGLM(**GLM)
    times_3, pr_3 = _run_profiled(lambda c: sglm3.fit(df[c], y), step_cols)
    total_3 = sum(times_3)
    pr_3_str = _pr_str(pr_3)
    print(f"  Total {total_3:.3f}s | Per-step {np.mean(times_3)*1e3:.1f}ms")

    # ── Opt-4: StepwiseGLM (hstack col cache + std cache + warm) ─────────
    print("\n[5/5] OPT-4     (StepwiseGLM: warm + std cache + hstack col cache)")
    # Pre-register all columns once
    sglm4 = StepwiseGLM(**GLM)
    sglm4._mat_cache.register_cols(df)   # warm per-col cache for score tests
    times_4, pr_4 = _run_profiled(lambda c: sglm4.fit(df[c], y), step_cols)
    total_4 = sum(times_4)
    pr_4_str = _pr_str(pr_4)
    print(f"  Total {total_4:.3f}s | Per-step {np.mean(times_4)*1e3:.1f}ms")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  {'Variant':<48} {'Total':>7}  {'Speedup':>8}  {'Per-step':>9}")
    print("  " + "-" * 76)
    for label, total, times in [
        ("Baseline (DataFrame, cold)",             total_b, times_b),
        ("Opt-1: from_pandas, cold",               total_1, times_1),
        ("Opt-2: from_pandas, warm_start",         total_2, times_2),
        ("Opt-3: StepwiseGLM (std cache)",         total_3, times_3),
        ("Opt-4: StepwiseGLM (hstack + std cache)",total_4, times_4),
    ]:
        print(f"  {label:<48} {total:>6.3f}s  {total_b/total:>7.2f}x  "
              f"{np.mean(times)*1e3:>7.1f}ms")

    # ── Per-step breakdown ────────────────────────────────────────────────
    print(f"\n  {'Step':<5} {'Base':>8} {'Opt1':>8} {'Opt2':>8} "
          f"{'Opt3':>8} {'Opt4':>8}  {'Std cache':>12}")
    print("  " + "-" * 68)
    for i, (tb, t1, t2, t3, t4) in enumerate(
        zip(times_b, times_1, times_2, times_3, times_4)
    ):
        cs = sglm4.cache_stats_.get(i, {})
        hr = cs.get("hit_rate", 0.0)
        hits = cs.get("hits", 0)
        nc   = cs.get("n_cols", 0)
        print(f"  {i:<5} {tb*1e3:>7.1f} {t1*1e3:>8.1f} {t2*1e3:>8.1f} "
              f"{t3*1e3:>8.1f} {t4*1e3:>8.1f}ms  {hits}/{nc} ({hr:.0%})")

    # ── Profiler breakdown ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PROFILER BREAKDOWN  (% of each variant's total wall time)")
    print("=" * 72)
    rows_b = _bucket(pr_b_str, total_b)
    rows_1 = _bucket(pr_1_str, total_1)
    rows_2 = _bucket(pr_2_str, total_2)
    rows_3 = _bucket(pr_3_str, total_3)
    rows_4 = _bucket(pr_4_str, total_4)
    print(f"\n  {'Category':<30} {'Base':>7} {'Opt1':>7} {'Opt2':>7} "
          f"{'Opt3':>7} {'Opt4':>7}")
    print("  " + "-" * 62)
    for rb, r1, r2, r3, r4 in zip(rows_b, rows_1, rows_2, rows_3, rows_4):
        label = rb[0]
        pcts  = [100 * r[1] / r[2] if r[2] > 0 else 0
                 for r in (rb, r1, r2, r3, r4)]
        print(f"  {label:<30} " +
              " ".join(f"{p:>6.1f}%" for p in pcts))

    # ── Correctness ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CORRECTNESS  (last step: baseline coef vs Opt-4 coef)")
    print("=" * 72)
    last = step_cols[-1]
    glm_ck = glum.GeneralizedLinearRegressor(**GLM)
    glm_ck.fit(df[last], y)

    sglm_ck = StepwiseGLM(**GLM)
    sglm_ck._mat_cache.register_cols(df)
    for cols in step_cols:
        sglm_ck.fit(df[cols], y)

    diff = np.max(np.abs(glm_ck.coef_ - sglm_ck.glm_.coef_))
    rel  = diff / (np.max(np.abs(glm_ck.coef_)) + 1e-12)
    print(f"\n  Max absolute coef diff: {diff:.2e}")
    print(f"  Max relative coef diff: {rel:.2e}")
    print(f"  {'PASS ✓' if rel < 1e-4 else 'WARN ✗ coefs differ > 1e-4'}")

    # ── Score-test screener validation ────────────────────────────────────
    print("\n" + "=" * 72)
    print("SCORE-TEST SCREENER VALIDATION")
    print("=" * 72)

    rng = np.random.default_rng(99)
    n_sc = 20_000
    df_sc = pd.DataFrame(rng.standard_normal((n_sc, 10)),
                         columns=[f"x{i}" for i in range(10)])
    # True signal: x2 and x5.  Active model: x0, x1 only.
    y_sc = rng.poisson(
        np.exp(0.3 + 0.3 * df_sc["x0"] + 0.5 * df_sc["x2"] - 0.4 * df_sc["x5"])
    ).astype(float)

    sglm_sc = StepwiseGLM(family="poisson", alpha=0.0)
    sglm_sc.fit(df_sc[["x0", "x1"]], y_sc)

    candidates = [f"x{i}" for i in range(2, 10)]

    # ── Score test timing ────────────────────────────────────────────────
    N_rep = 50
    t0 = time.perf_counter()
    for _ in range(N_rep):
        scores = sglm_sc.screen_candidates(df_sc, ["x0", "x1"], candidates)
    t_screen = (time.perf_counter() - t0) / N_rep

    # Full IRLS refit for each candidate (the naive approach)
    t0 = time.perf_counter()
    for _ in range(N_rep):
        for cand in candidates:
            g = glum.GeneralizedLinearRegressor(family="poisson", alpha=0.0)
            g.fit(df_sc[["x0", "x1", cand]], y_sc)
    t_refit = (time.perf_counter() - t0) / N_rep

    print(f"\n  Candidates: {len(candidates)}")
    print(f"  Score test  (all candidates): {t_screen*1e3:>7.2f}ms")
    print(f"  Full refits ({len(candidates)} × IRLS):  {t_refit*1e3:>7.2f}ms")
    print(f"  Speedup: {t_refit/t_screen:.1f}x")

    # ── Ranking accuracy ─────────────────────────────────────────────────
    print(f"\n  Score test ranking (true signals: x2, x5):")
    print(f"  {'Column':<12} {'Statistic':>12} {'p-value':>12} {'Signal?':>8}")
    print("  " + "-" * 48)
    true_signals = {"x2", "x5"}
    for r in scores:
        flag = "✓ TRUE" if r.column in true_signals else ""
        print(f"  {r.column:<12} {r.statistic:>12.2f} {r.pvalue:>12.4f} {flag:>8}")

    # ── Compare score ranking to full-refit deviance ranking ─────────────
    print(f"\n  Comparing score-test rank to full-refit deviance rank:")
    deviances = {}
    for cand in candidates:
        g = glum.GeneralizedLinearRegressor(family="poisson", alpha=0.0)
        g.fit(df_sc[["x0", "x1", cand]], y_sc)
        mu_cand = g.predict(df_sc[["x0", "x1", cand]])
        deviances[cand] = g._family_instance.deviance(
            y_sc, mu_cand, np.ones(n_sc)
        )

    refit_rank  = sorted(candidates, key=lambda c: deviances[c])
    score_rank  = [r.column for r in scores]
    top3_match  = refit_rank[:3] == score_rank[:3]

    print(f"  Score-test top-3:  {score_rank[:3]}")
    print(f"  Refit-deviance top-3: {refit_rank[:3]}")
    print(f"  Top-3 rank agreement: {'PASS ✓' if top3_match else 'PARTIAL (check manually)'}")

    print()


if __name__ == "__main__":
    run()
