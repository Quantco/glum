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
    - glum_benchmarks/results/pickles/: Pickle files with detailed results
    - glum_benchmarks/results/figures/: PNG plots comparing library performance
    - glum_benchmarks/results/results.csv: Summary CSV for reproducibility
"""

from __future__ import annotations

import os
import pickle
import shutil
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

# TODO: Rerun the benchmarks
# TODO: update README and the documentation

# CONFIGURATION

# Keep as-is to run the same benchmark suite as in the documentation. In particular
# if you just want to update the benchmark results,
# you do not need to change anything here.

# Steps to run
RUN_BENCHMARKS = True  # Run the benchmarks (can be slow)
ANALYZE_RESULTS = True  # Analyze and print results (writes CSV_FILE)
GENERATE_PLOTS = True  # Generate comparison plots (reads from CSV_FILE and writes PNGs)

# Output directories (relative to this file)
# Change RUN_NAME to separate different benchmark runs, e.g.:
#   RUN_NAME = "housing_lasso"
#   RUN_NAME = "insurance_full"
# Note: Only "docs" folder has its CSV tracked in git (for documentation)
_SCRIPT_DIR = Path(__file__).parent
RUN_NAME = "docs"  # Subfolder name within results/ ("docs" CSV is git-tracked)
RESULTS_DIR = _SCRIPT_DIR / "results" / RUN_NAME
PICKLE_DIR = RESULTS_DIR / "pickles"
FIGURE_DIR = RESULTS_DIR / "figures"
CSV_FILE = RESULTS_DIR / "results.csv"

# Cache settings (shared across all runs)
DATA_CACHE_DIR = _SCRIPT_DIR / ".cache"  # Where to store the data cache
CACHE_DATA = True  # Cache data loading across library runs
CLEAR_DATA_CACHE = False  # Clear data cache before running
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
        standardize=STANDARDIZE,
    )

    return result, params


def run_all_benchmarks():
    """Run all configured benchmarks."""
    # Set up data caching
    if CACHE_DATA:
        os.environ["GLM_BENCHMARKS_CACHE"] = str(DATA_CACHE_DIR.absolute())
        if CLEAR_DATA_CACHE and DATA_CACHE_DIR.exists():
            print(f"Clearing data cache: {DATA_CACHE_DIR}")
            shutil.rmtree(DATA_CACHE_DIR)

    # Set up output directory
    if CLEAR_OUTPUT and PICKLE_DIR.exists():
        print(f"Clearing output directory: {PICKLE_DIR}")
        shutil.rmtree(PICKLE_DIR)
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
            print(f"[{current}/{total}] {library_name} / {problem_name}", end=" ")

            try:
                result, params = run_single_benchmark(problem_name, library_name)

                # Save result
                fname = params.get_result_fname() + ".pkl"
                with open(PICKLE_DIR / fname, "wb") as f:
                    pickle.dump(result, f)

                if len(result) > 0 and "runtime" in result:
                    print(f"-> {result['runtime']:.4f}s")
                else:
                    print("-> (no result)")

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

    # Extract distribution and regularization from problem name
    df = df.copy()
    df["distribution"] = df["problem_name"].apply(lambda x: x.split("-")[-1])
    df["regularization"] = df["problem_name"].apply(
        lambda x: "lasso" if "lasso" in x else "l2"
    )
    df["dataset"] = df["problem_name"].apply(
        lambda x: "-".join(x.split("-")[:-3])  # e.g., "intermediate-housing-no"
    )

    # Drop duplicates (keep latest result for each problem/library combo)
    df = df.drop_duplicates(subset=["problem_name", "library_name"], keep="last")

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

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)

            reg_label = "Lasso" if reg == "lasso" else "Ridge"
            ax.set_title(f"{dataset.replace('-', ' ').title()} - {reg_label}")
            ax.set_ylabel("Runtime (s)")
            ax.set_xlabel("Distribution")
            ax.legend(title="Library", bbox_to_anchor=(1.02, 1), loc="upper left")
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

                fig, ax = plt.subplots(figsize=(10, 5))
                pivot_norm.plot(kind="bar", ax=ax)

                ax.set_title(
                    f"{dataset.replace('-', ' ').title()} - {reg_label} (normalized)"
                )
                ax.set_ylabel("Runtime relative to glum (1.0 = glum)")
                ax.set_xlabel("Distribution")
                ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
                ax.legend(title="Library", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                fname_norm = f"{dataset}-{reg}-normalized"
                plt.savefig(FIGURE_DIR / f"{fname_norm}.png", dpi=300)
                plt.close()
                print(f"Saved: {fname_norm}.png")


def main():
    """Run benchmarks, analyze, and plot based on configuration.

    Workflow:
        1. RUN_BENCHMARKS: Execute benchmarks, save pickle files to PICKLE_DIR
        2. ANALYZE_RESULTS: Analyze pickles, print summary, write CSV_FILE
        3. GENERATE_PLOTS: Read CSV_FILE, generate figures to FIGURE_DIR

    You can run steps independently:
        - Run benchmarks once, then regenerate figures later from CSV
        - Commit CSV to repo for reproducibility
    """
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
