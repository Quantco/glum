"""
StepwiseGLM demo — French Motor Insurance
==========================================

Forward stepwise variable selection on the French motor insurance dataset
(678k policies) using a Poisson GLM for claim frequency with log-exposure
offset.  Demonstrates the two-stage workflow:

  Stage 1  screen_candidates()  — score (Rao) test ranks all candidates
                                   at O(n) per candidate (vs O(n·p·iters))
  Stage 2  cv_select()          — 5-fold CV on the score-test shortlist,
                                   with cached fold matrices + std stats

Run with:
    pixi run -e benchmark python src/glum/_stepwise_demo.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glum
from glum._stepwise import StepwiseGLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_FILE  = REPO_ROOT / "data" / "insurance.parquet"
OUTPUT_PNG = REPO_ROOT / "stepwise_demo.png"

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------

print("=" * 70)
print("StepwiseGLM demo — French Motor Insurance")
print("=" * 70)

df = pd.read_parquet(DATA_FILE)
print(f"\nDataset: {len(df):,} policies, {df.shape[1]} raw columns")

# Target and offset
y       = df["ClaimNb"].to_numpy(dtype=float)
offset  = np.log(df["Exposure"].clip(lower=1e-6).to_numpy(dtype=float))

# Candidate predictors — mix of numeric and categorical
numeric_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
cat_cols     = ["Area", "VehBrand", "VehGas", "Region"]

# Encode categoricals as pd.Categorical (tabmat handles them natively)
df_model = df[numeric_cols + cat_cols].copy()
for c in cat_cols:
    df_model[c] = pd.Categorical(df_model[c])

all_candidates = numeric_cols + cat_cols
print(f"Candidates: {len(all_candidates)} variables "
      f"({len(numeric_cols)} numeric, {len(cat_cols)} categorical)")
print(f"  Numeric : {numeric_cols}")
print(f"  Categorical: {', '.join(f'{c} ({df_model[c].nunique()} levels)' for c in cat_cols)}")
print(f"Claim rate: {y.mean():.4f}  |  Mean exposure: {df['Exposure'].mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Stepwise loop
# ---------------------------------------------------------------------------

# BonusMalus is the dominant actuarial predictor — start with it in the model.
# Let the loop discover everything else.
INITIAL_ACTIVE = ["BonusMalus"]
MAX_STEPS      = 6
CV_FOLDS       = 5
N_ALPHAS       = 15
SCORE_ALPHA    = 0.05   # p-value threshold for score-test shortlist
TOP_K_SCREEN   = 5      # max candidates passed from score test to CV

print(f"\nInitial model: {INITIAL_ACTIVE}")
print(f"Max steps: {MAX_STEPS}  |  CV folds: {CV_FOLDS}  |  Alphas: {N_ALPHAS}")
print(f"Score-test shortlist: top {TOP_K_SCREEN} with p < {SCORE_ALPHA}")

sglm = StepwiseGLM(family="poisson", alpha=1e-4, l1_ratio=0, drop_first=True)

active     = list(INITIAL_ACTIVE)
candidates = [c for c in all_candidates if c not in active]

# Fit initial model (with offset)
sglm.fit(df_model[active], y)

step_records = []   # for the plot

print()
print(f"{'Step':<5} {'Added':<14} {'Score ms':>9} {'CV ms':>9} "
      f"{'Speedup':>8}  {'CV deviance':>13}  {'Shortlist'}")
print("─" * 80)

for step in range(MAX_STEPS):
    if not candidates:
        break

    # ── Stage 1: score-test screening ────────────────────────────────────
    t0     = time.perf_counter()
    scores = sglm.screen_candidates(df_model, active, candidates)
    t_score = (time.perf_counter() - t0) * 1e3

    # Shortlist: top-K significant candidates
    shortlist = [r.column for r in scores if r.pvalue < SCORE_ALPHA][:TOP_K_SCREEN]
    if not shortlist:
        print(f"  {step+1:<4} No candidates pass score-test threshold — stopping.")
        break

    # ── Stage 2: CV on shortlist ──────────────────────────────────────────
    t0    = time.perf_counter()
    cv_results = sglm.cv_select(
        df_model, active, shortlist, y,
        cv=CV_FOLDS, n_alphas=N_ALPHAS,
    )
    t_cv = (time.perf_counter() - t0) * 1e3

    # Naive baseline: one GeneralizedLinearRegressorCV per shortlist candidate
    t0 = time.perf_counter()
    for cand in shortlist:
        _g = glum.GeneralizedLinearRegressorCV(
            family="poisson", n_alphas=N_ALPHAS, cv=CV_FOLDS,
            l1_ratio=0, drop_first=True,
        )
        _g.fit(df_model[active + [cand]], y)
    t_naive = (time.perf_counter() - t0) * 1e3

    best = cv_results[0]
    speedup = t_naive / t_cv

    print(f"  {step+1:<4} {best.column:<14} {t_score:>8.0f} {t_cv:>8.0f}  "
          f"{speedup:>7.1f}x  {best.cv_deviance:>13.4f}  "
          f"{shortlist}")

    step_records.append(dict(
        step          = step + 1,
        added         = best.column,
        n_active      = len(active) + 1,
        cv_deviance   = best.cv_deviance,
        t_score_ms    = t_score,
        t_cv_ms       = t_cv,
        t_naive_ms    = t_naive,
        speedup       = speedup,
        shortlist_len = len(shortlist),
    ))

    # Stop if CV deviance is not improving by more than 0.1%
    if step > 0 and (step_records[-2]["cv_deviance"] - best.cv_deviance) / step_records[-2]["cv_deviance"] < 0.001:
        print(f"       Marginal improvement < 0.1% — stopping.")
        break

    active.append(best.column)
    candidates.remove(best.column)
    sglm.fit(df_model[active], y)

# ---------------------------------------------------------------------------
# 3. Final model summary
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("FINAL MODEL")
print("=" * 70)
print(f"\n  Selected variables ({len(active)}): {active}")
print(f"\n  Coefficients:")
feat_names = sglm.glm_.feature_names_
coefs      = sglm.glm_.coef_
for name, coef in sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:15]:
    print(f"    {name:<40} {coef:+.4f}")
if len(feat_names) > 15:
    print(f"    ... ({len(feat_names) - 15} more)")

total_score_ms = sum(r["t_score_ms"] for r in step_records)
total_cv_ms    = sum(r["t_cv_ms"]    for r in step_records)
total_naive_ms = sum(r["t_naive_ms"] for r in step_records)

print(f"\n  Timing summary across {len(step_records)} steps:")
print(f"    Score-test total : {total_score_ms/1e3:.2f}s")
print(f"    cv_select total  : {total_cv_ms/1e3:.2f}s")
print(f"    Naive CV total   : {total_naive_ms/1e3:.2f}s")
print(f"    Overall speedup  : {total_naive_ms/total_cv_ms:.1f}x")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

if not step_records:
    print("\nNo steps completed — skipping plot.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("StepwiseGLM — French Motor Insurance (678k policies, Poisson)",
                 fontsize=13, fontweight="bold")

    steps      = [r["step"]        for r in step_records]
    deviances  = [r["cv_deviance"] for r in step_records]
    speedups   = [r["speedup"]     for r in step_records]
    t_cv       = [r["t_cv_ms"]/1e3 for r in step_records]
    t_naive    = [r["t_naive_ms"]/1e3 for r in step_records]
    labels     = [r["added"]       for r in step_records]

    # ── Panel 1: CV deviance path ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, deviances, "o-", color="#2196F3", linewidth=2, markersize=8)
    for s, d, lbl in zip(steps, deviances, labels):
        ax.annotate(lbl, (s, d), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#555")
    ax.set_xlabel("Stepwise step")
    ax.set_ylabel("Mean CV deviance (hold-out)")
    ax.set_title("Variable selection path")
    ax.set_xticks(steps)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.grid(True, alpha=0.3)

    # ── Panel 2: cv_select vs naive timing ────────────────────────────────
    ax = axes[1]
    x = np.arange(len(steps))
    w = 0.35
    ax.bar(x - w/2, t_naive, w, label="Naive (GLM-CV per cand.)",
           color="#EF5350", alpha=0.85)
    ax.bar(x + w/2, t_cv,    w, label="cv_select (cached)",
           color="#66BB6A", alpha=0.85)
    ax.set_xlabel("Stepwise step")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("cv_select vs naive CV time")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}\n+{lbl}" for s, lbl in zip(steps, labels)],
                       fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: speedup by step ──────────────────────────────────────────
    ax = axes[2]
    colors = ["#1565C0" if sp >= 1.0 else "#B71C1C" for sp in speedups]
    bars = ax.bar(steps, speedups, color=colors, alpha=0.85, width=0.6)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{sp:.1f}×", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("Stepwise step")
    ax.set_ylabel("Speedup vs naive")
    ax.set_title("cv_select speedup (cached fold matrices)")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUTPUT_PNG}")

print()
