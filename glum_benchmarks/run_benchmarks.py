#!/usr/bin/env python
"""
Benchmark runner script for comparing GLM libraries.

Usage:
    pixi run -e benchmark run-benchmarks

Configuration:
    Edit the CONFIGURATION section below to select which libraries, datasets,
    regularizations, and distributions to benchmark. You can also control which
    steps to run (RUN_BENCHMARKS, ANALYZE_RESULTS, GENERATE_PLOTS).

Output:
    - glum_benchmarks/results/RUN_NAME/pickles/: Pickle files with detailed results
    - glum_benchmarks/results/RUN_NAME/figures/: PNG plots comparing library performance
    - glum_benchmarks/results/RUN_NAME/results.csv: Summary CSV for reproducibility
"""

from __future__ import annotations

import pickle
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glum_benchmarks.problems import get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    execute_problem_library,
    get_all_libraries,
    get_params_from_fname,
)

# TODO: update README and documentation
# TODO: improve plotting to handle "not converged" and "doesnt supported" cases better

# CONFIGURATION

# Keep as-is to run the same benchmark suite as in the documentation. In particular
# if you just want to update the benchmark results,
# you do not need to change anything here.

# Steps to run
RUN_BENCHMARKS = True  # Run the benchmarks (can be slow)
ANALYZE_RESULTS = True  # Analyze and print results (writes CSV_FILE)
GENERATE_PLOTS = True  # Generate comparison plots (reads from CSV_FILE and writes PNGs)


# Change RUN_NAME to separate different benchmark runs into different folders.
_SCRIPT_DIR = Path(__file__).parent
RUN_NAME = "docs"  # Subfolder name within results/ ("docs" CSV is git-tracked)
RESULTS_DIR = _SCRIPT_DIR / "results" / RUN_NAME
PICKLE_DIR = RESULTS_DIR / "pickles"
FIGURE_DIR = RESULTS_DIR / "figures"
CSV_FILE = RESULTS_DIR / "results.csv"
CLEAR_OUTPUT = True  # Clear pickle output directory before running

# Libraries to benchmark: "glum", "sklearn", "h2o", "liblinear",
# "skglm", "celer", "zeros"
LIBRARIES = [
    "glum",
    "sklearn",
    "h2o",
    "liblinear",
    "skglm",
    "celer",
]

# Datasets to run: "intermediate-housing", "intermediate-insurance",
# "narrow-insurance", "wide-insurance"
DATASETS = [
    "intermediate-housing",
    "intermediate-insurance",
    "narrow-insurance",
    "wide-insurance",
]

# Regularization types to include: "lasso", "l2", "net"
REGULARIZATIONS = ["lasso", "l2"]

# Distributions to include: "gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"
DISTRIBUTIONS = ["gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"]

# Benchmark settings
NUM_THREADS = 16
REG_STRENGTH = 0.001
STANDARDIZE = True
ITERATIONS = 2  # Run each benchmark N times, report minimum runtime (>=2 for skglm)
NUM_ROWS = None  # None = use full dataset, or set an int for quick test runs


def get_problems_to_run() -> list[str]:
    """Get list of problem names matching the configuration."""
    all_problems = get_all_problems()
    selected = []

    for name in all_problems.keys():
        # Only benchmark "-no-weights" problems (skip offset and weighted variants)
        if "-no-weights-" not in name:
            continue

        # Filter by dataset
        if DATASETS is not None:
            if not any(d in name for d in DATASETS):
                continue

        # Filter by regularization
        if REGULARIZATIONS is not None:
            if not any(reg in name for reg in REGULARIZATIONS):
                continue

        # Filter by distribution
        if DISTRIBUTIONS is not None:
            if not any(dist in name for dist in DISTRIBUTIONS):
                continue

        selected.append(name)

    return sorted(selected)


def run_single_benchmark(
    problem_name: str, library_name: str
) -> tuple[dict, BenchmarkParams]:
    """Run a single benchmark and return results."""
    params = BenchmarkParams(
        problem_name=problem_name,
        library_name=library_name,
        num_rows=NUM_ROWS,
        storage="dense",
        threads=NUM_THREADS,
        regularization_strength=REG_STRENGTH,
    )

    result, _ = execute_problem_library(
        params,
        iterations=ITERATIONS,
        diagnostics_level=None,
        standardize=STANDARDIZE,
    )

    return result, params


def run_all_benchmarks():
    """Run all configured benchmarks."""
    # Set up output directory
    if CLEAR_OUTPUT and RESULTS_DIR.exists():
        print(f"Clearing output directory: {RESULTS_DIR}")
        shutil.rmtree(RESULTS_DIR)
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    problems = get_problems_to_run()
    available = get_all_libraries()
    libraries = [lib for lib in LIBRARIES if lib in available]

    print(f"Problems: {len(problems)}")
    print(f"Libraries: {libraries}")
    print()

    total = len(problems) * len(libraries)
    current = 0

    for problem_name in problems:
        for library_name in libraries:
            current += 1
            print(
                f"[{current}/{total}] {library_name} / {problem_name}",
                end=" ",
                flush=True,
            )

            try:
                # Capture warnings to display after the result
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    # Ignore deprecation warnings from third-party libraries
                    warnings.filterwarnings(
                        "ignore",
                        message=".*asyncio.iscoroutinefunction.*",
                    )
                    result, params = run_single_benchmark(problem_name, library_name)

                # Save result
                fname = params.get_result_fname() + ".pkl"
                with open(PICKLE_DIR / fname, "wb") as f:
                    pickle.dump(result, f)

                if len(result) > 0 and "runtime" in result:
                    print(f"-> {result['runtime']:.4f}s")
                else:
                    print("-> (skipped)")

                # Print captured warnings
                for w in caught_warnings:
                    print(f"{w.message}")

            except Exception as e:
                print(f"-> ERROR: {e}")


def analyze_results() -> pd.DataFrame:
    """Analyze benchmark results and print summary."""
    print()
    print("=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)

    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("display.precision", display_precision)

    results = []

    for fname in PICKLE_DIR.glob("*.pkl"):
        with open(fname, "rb") as f:
            data = pickle.load(f)

        if not data or "coef" not in data:
            continue

        params = get_params_from_fname(fname.name)
        if params.problem_name is None:
            continue
        problem = get_all_problems()[params.problem_name]
        coefs = data["coef"]

        # Calculate runtime per iteration
        n_iter = data.get("n_iter")
        runtime = data.get("runtime")
        if n_iter is not None and n_iter > 0:
            runtime_per_iter = runtime / n_iter
        else:
            runtime_per_iter = runtime

        # Calculate coefficient norms
        l1_norm: float = np.sum(np.abs(coefs))
        l2_norm: float = np.sum(coefs**2)
        num_nonzero_coef: int = np.sum(np.abs(coefs) > 1e-8)

        # Get regularization strength from params or problem default
        reg_strength = (
            problem.regularization_strength
            if params.regularization_strength is None
            else params.regularization_strength
        )

        results.append(
            {
                "problem_name": params.problem_name,
                "library_name": params.library_name,
                "num_rows": data.get("num_rows"),
                "regularization_strength": reg_strength,
                "storage": params.storage,
                "threads": params.threads,
                "n_iter": n_iter,
                "runtime": runtime,
                "runtime per iter": runtime_per_iter,
                "intercept": data.get("intercept"),
                "l1": l1_norm,
                "l2": l2_norm,
                "num_nonzero_coef": num_nonzero_coef,
                "obj_val": data.get("obj_val"),
            }
        )

    if not results:
        print("No results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Format: set index and sort
    problem_id_cols = ["problem_name", "num_rows", "regularization_strength"]
    df = df.set_index(problem_id_cols).sort_values("library_name").sort_index()

    # Calculate relative objective value (how far from best)
    df["rel_obj_val"] = df[["obj_val"]] - df.groupby(level=[0, 1, 2])[["obj_val"]].min()

    # Display columns
    cols_to_show = [
        "library_name",
        "storage",
        "threads",
        "n_iter",
        "runtime",
        "intercept",
        "num_nonzero_coef",
        "obj_val",
        "rel_obj_val",
    ]

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        print(df[cols_to_show])

    # Export to CSV for figure generation and reproducibility
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(CSV_FILE, index=False)
    print(f"\nExported results to: {CSV_FILE}")

    return df.reset_index()


def plot_results():
    """Generate benchmark comparison plots from CSV file.

    Reads from CSV_FILE, which allows regenerating figures without
    re-running benchmarks. The CSV can be committed to the repository.
    """
    print()
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        print("Run ANALYZE_RESULTS first to generate the CSV.")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"Reading results from: {CSV_FILE}")

    if df.empty:
        print("No data to plot!")
        return

    FIGURE_DIR.mkdir(exist_ok=True)

    # Extract distribution, regularization and dataset from problem_name
    # Format: {dataset}-no-weights-{regularization}-{distribution}
    # where dataset is 2 parts, regularization is 1 part (lasso/l2/net)
    # and distribution may contain hyphens (e.g., "tweedie-p=1.5")
    df = df.copy()

    def parse_problem_name(name):
        """Parse problem name into (dataset, regularization, distribution)."""
        parts = name.split("-")
        dataset = "-".join(parts[:2])
        reg = parts[4]
        dist = "-".join(parts[5:])
        return dataset, reg, dist

    parsed = df["problem_name"].apply(parse_problem_name)
    df["dataset"] = parsed.apply(lambda x: x[0])
    reg_map = {"lasso": "lasso", "l2": "ridge", "net": "elastic-net"}
    df["regularization"] = parsed.apply(lambda x: reg_map.get(x[1], x[1]))
    df["distribution"] = parsed.apply(lambda x: x[2])

    # Drop duplicates (keep latest result for each unique combo)
    # Use all derived columns to ensure no duplicates in pivot
    df = df.drop_duplicates(
        subset=["dataset", "regularization", "distribution", "library_name"],
        keep="last",
    )

    # Ensure library colors are consistent across plots
    colors = {
        "glum": "#a6cee3",
        "h2o": "#fdbf6f",
        "glmnet": "#b15928",
        "sklearn": "#b15928",
        "liblinear": "#33a02c",
        "skglm": "#fb9a99",
        "celer": "#cab2d6",
    }

    # Generate one plot per dataset/regularization combo
    for dataset in df["dataset"].unique():
        for reg in df["regularization"].unique():
            subset = df[(df["dataset"] == dataset) & (df["regularization"] == reg)]

            if subset.empty:
                continue

            # Pivot for plotting
            pivot = subset.pivot(
                index="distribution",
                columns="library_name",
                values="runtime",
            ).fillna(0)

            # Calculate y-axis limit (10x fastest runtime)
            min_runtime = pivot.values[pivot.values > 0].min()
            y_max = min_runtime * 10

            # Get colors for the libraries in this plot
            plot_colors = [colors.get(lib, "#999999") for lib in pivot.columns]

            # Create bar chart with clipped bars and annotations
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot_clipped = pivot.clip(upper=y_max)
            pivot_clipped.plot(kind="bar", ax=ax, color=plot_colors)

            # Add annotations for clipped bars (show original value on bar with arrow)
            n_dists = len(pivot.index)
            bars = ax.patches
            for i, dist in enumerate(pivot.index):
                for j, lib in enumerate(pivot.columns):
                    original_val = pivot.loc[dist, lib]
                    if original_val > y_max:
                        bar_idx = j * n_dists + i
                        bar = bars[bar_idx]
                        x = bar.get_x() + bar.get_width() / 2
                        # Number on the bar
                        ax.text(
                            x,
                            y_max * 0.75,
                            f"{original_val:.4f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            rotation=90,
                        )
                        # Arrow starting inside bar, pointing up to top of bar
                        ax.annotate(
                            "",
                            xy=(x, y_max),
                            xytext=(x, y_max * 0.88),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                        )

            ax.set_ylim(0, y_max * 1.08)
            reg_label = reg.replace("-", " ").title().replace(" ", "-")
            # Title with hyphens like in the example
            title_dataset = dataset.replace(" ", "-").title()
            ax.set_title(f"{title_dataset}-{reg_label}")
            ax.set_ylabel("run time (s)")
            ax.set_xlabel("")  # No x-label, distribution names are self-explanatory
            ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
            # Capitalize x-tick labels
            ax.set_xticklabels(
                [label.get_text().title() for label in ax.get_xticklabels()]
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save
            fname = f"{dataset}-{reg}"
            plt.savefig(FIGURE_DIR / f"{fname}.png", dpi=300)
            plt.close()
            print(f"Saved: {fname}.png")

            # Generate normalized plot (glum = 1.0)
            if "glum" in pivot.columns:
                pivot_norm = pivot.div(pivot["glum"], axis=0)

                # Limit to 10x glum (so y_max = 10)
                norm_y_max = 10.0
                pivot_norm_clipped = pivot_norm.clip(upper=norm_y_max)

                fig, ax = plt.subplots(figsize=(10, 5))
                pivot_norm_clipped.plot(kind="bar", ax=ax, color=plot_colors)

                # Add annotations for clipped bars (show original value on bar)
                n_dists = len(pivot_norm.index)
                bars = ax.patches
                for i, dist in enumerate(pivot_norm.index):
                    for j, lib in enumerate(pivot_norm.columns):
                        original_val = pivot_norm.loc[dist, lib]
                        if original_val > norm_y_max:
                            bar_idx = j * n_dists + i
                            bar = bars[bar_idx]
                            x = bar.get_x() + bar.get_width() / 2
                            # Number on the bar
                            ax.text(
                                x,
                                norm_y_max * 0.75,
                                f"{original_val:.1f}x",
                                ha="center",
                                va="center",
                                fontsize=9,
                                fontweight="bold",
                                rotation=90,
                            )
                            # Arrow starting inside bar, pointing up to top of bar
                            ax.annotate(
                                "",
                                xy=(x, norm_y_max),
                                xytext=(x, norm_y_max * 0.88),
                                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                            )

                ax.set_ylim(0, norm_y_max * 1.08)
                ax.set_title(f"{title_dataset}-{reg_label} (normalized)")
                ax.set_ylabel("run time relative to glum")
                ax.set_xlabel("")
                ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
                ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.set_xticklabels(
                    [label.get_text().title() for label in ax.get_xticklabels()]
                )
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                fname_norm = f"{dataset}-{reg}-normalized"
                plt.savefig(FIGURE_DIR / f"{fname_norm}.png", dpi=300)
                plt.close()
                print(f"Saved: {fname_norm}.png")


def main():
    if RUN_BENCHMARKS:
        run_all_benchmarks()

    if ANALYZE_RESULTS:
        analyze_results()

    if GENERATE_PLOTS:
        plot_results()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    if RUN_BENCHMARKS or ANALYZE_RESULTS:
        print(f"Results saved to: {RESULTS_DIR}/")
    if GENERATE_PLOTS:
        print(f"Figures saved to: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
