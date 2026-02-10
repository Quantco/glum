#!/usr/bin/env python
"""Compare base/head benchmark CSV files and fail on regressions."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import pandas as pd
from ruamel.yaml import YAML


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

    cfg_loaded: Any = YAML(typ="safe", pure=True).load(Path(args.config).read_text())
    cfg = cfg_loaded if isinstance(cfg_loaded, dict) else {}
    max_rel = float(cfg.get("max_rel_slowdown", 0.15))
    max_abs = float(cfg.get("max_abs_slowdown_sec", 0.05))
    max_cases = int(cfg.get("max_regressed_cases", 0))

    base = pd.read_csv(Path(args.base))
    head = pd.read_csv(Path(args.head))

    keys = ["problem_name", "library_name", "num_rows", "alpha"]
    merged = base.merge(head, on=keys, suffixes=("_base", "_head"), how="inner")
    merged["delta_sec"] = merged["runtime_head"] - merged["runtime_base"]
    merged["delta_ratio"] = merged["delta_sec"] / merged["runtime_base"]
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

    rich_console = None
    rich_table = None
    try:
        rich_console = importlib.import_module("rich.console")
        rich_table = importlib.import_module("rich.table")
    except ModuleNotFoundError:
        pass

    if rich_console is not None and rich_table is not None:
        console = rich_console.Console()
        console.print(
            (
                f"Thresholds: rel > {max_rel:.1%}, abs > {max_abs:.3f}s, "
                f"allowed={max_cases}, detected={regressed_count}"
            ),
            style="bold",
        )
        table = rich_table.Table(show_header=True, header_style="bold magenta")
        table.add_column("Case")
        table.add_column("Base (s)", justify="right")
        table.add_column("Head (s)", justify="right")
        table.add_column("Delta (s)", justify="right")
        table.add_column("Delta (%)", justify="right")
        table.add_column("Regressed", justify="center")
        for row in merged.to_dict(orient="records"):
            regressed = bool(row["regressed"])
            table.add_row(
                f"{row['problem_name']} / {row['library_name']}",
                f"{float(row['runtime_base']):.4f}",
                f"{float(row['runtime_head']):.4f}",
                f"{float(row['delta_sec']):+.4f}",
                f"{100.0 * float(row['delta_ratio']):+.2f}%",
                "yes" if regressed else "no",
                style="red" if regressed else None,
            )
        console.print(table)
    else:
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
