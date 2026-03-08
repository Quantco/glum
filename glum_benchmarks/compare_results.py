#!/usr/bin/env python
"""Compare base/head benchmark CSV files and fail on regressions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glum_benchmarks.run_benchmarks import BenchmarkConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark results.")
    parser.add_argument("--base", required=True, help="Base results.csv path")
    parser.add_argument("--head", required=True, help="Head results.csv path")
    parser.add_argument(
        "--config",
        default="glum_benchmarks/config_ci.yaml",
        help="Config path with regression thresholds",
    )
    parser.add_argument(
        "--summary-out",
        default="glum_benchmarks/results/ci_runtime_summary.md",
        help="Path to write markdown summary",
    )
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml(Path(args.config))
    max_rel = cfg.max_rel_slowdown
    max_abs = cfg.max_abs_slowdown_sec
    max_cases = cfg.max_regressed_cases

    base = pd.read_csv(Path(args.base))
    head = pd.read_csv(Path(args.head))

    keys = ["problem_name", "library_name", "num_rows", "alpha"]
    merged = base.merge(head, on=keys, suffixes=("_base", "_head"), how="inner")

    # delta_sec: absolute slowdow
    # delta_ratio: relative slowdown
    # regressed: True when both relative AND absolute thresholds are exceeded,
    # so tiny absolute differences on fast benchmarks don't trigger failures.
    merged["delta_sec"] = merged["runtime_head"] - merged["runtime_base"]
    base_runtime = merged["runtime_base"].astype(float)
    delta_sec = merged["delta_sec"].astype(float)
    merged["delta_ratio"] = np.where(
        base_runtime != 0.0,
        delta_sec / base_runtime,
        np.where(delta_sec > 0.0, np.inf, np.where(delta_sec < 0.0, -np.inf, 0.0)),
    )
    merged["regressed"] = (merged["delta_ratio"] > max_rel) & (
        merged["delta_sec"] > max_abs
    )
    merged = merged.sort_values("delta_ratio", ascending=False)

    regressed_count = sum(bool(flag) for flag in merged["regressed"].to_list())
    lines = [
        "## Runtime Regression Summary",
        "",
        f"- Thresholds: rel slowdown > {max_rel:.1%} and abs slowdown > {max_abs:.3f}s",
        f"- Allowed regressed cases: {max_cases}",
        f"- Detected regressed cases: {regressed_count}",
        "",
        "| Case | Base (s) | Head (s) | Delta (s) | Delta (%) | Regressed |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in merged.to_dict(orient="records"):
        row_template = (
            "| {case} | {base:.4f} | {head:.4f} | {delta:+.4f} | "
            "{pct:+.2f}% | {regressed} |"
        )
        lines.append(
            row_template.format(
                case=f"{row['problem_name']} / {row['library_name']}",
                base=float(row["runtime_base"]),
                head=float(row["runtime_head"]),
                delta=float(row["delta_sec"]),
                pct=100.0 * float(row["delta_ratio"]),
                regressed="yes" if bool(row["regressed"]) else "no",
            )
        )
    lines.append("")
    summary = "\n".join(lines)
    print(summary)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(summary + "\n")

    if regressed_count > max_cases:
        print(f"Runtime regression check failed: {regressed_count} regressed case(s).")
        return 1
    print("Runtime regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
