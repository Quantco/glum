#!/usr/bin/env python
"""
Benchmark runner script for comparing GLM libraries.

Usage:
    pixi run -e benchmark run-benchmarks

Configuration:
    Edit config.yaml to select which libraries, datasets, regularizations,
    and distributions to benchmark. You can also control which steps to run
    (run_benchmarks, analyze_results, generate_plots).

Output:
    - glum_benchmarks/results/RUN_NAME/pickles/: Pickle files with detailed results
    - glum_benchmarks/results/RUN_NAME/figures/: PNG plots comparing library performance
    - glum_benchmarks/results/RUN_NAME/results.csv: Summary CSV for reproducibility
"""

from __future__ import annotations

import pickle
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

from glum_benchmarks.problems import get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    execute_problem_library,
    get_all_libraries,
    get_params_from_fname,
)

# TODO: update README and documentation
# TODO: improve plotting to handle "not converged" and "not supported" cases better


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Steps to run
    run_benchmarks: bool
    analyze_results: bool
    generate_plots: bool

    # Output settings
    run_name: str
    clear_output: bool

    # Problem selection
    libraries: list[str]
    datasets: list[str]
    regularizations: list[str]
    distributions: list[str]

    # Benchmark settings
    num_threads: int
    reg_strength: float
    standardize: bool
    iterations: int
    num_rows: int | None
    max_iter: int

    # Derived paths (computed after init)
    script_dir: Path
    results_dir: Path
    pickle_dir: Path
    figure_dir: Path
    csv_file: Path

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        script_dir = yaml_path.parent
        results_dir = script_dir / "results" / data["run_name"]

        return cls(
            run_benchmarks=data["run_benchmarks"],
            analyze_results=data["analyze_results"],
            generate_plots=data["generate_plots"],
            run_name=data["run_name"],
            clear_output=data["clear_output"],
            libraries=data["libraries"],
            datasets=data["datasets"],
            regularizations=data["regularizations"],
            distributions=data["distributions"],
            num_threads=data["num_threads"],
            reg_strength=data["reg_strength"],
            standardize=data["standardize"],
            iterations=data["iterations"],
            num_rows=data["num_rows"],
            max_iter=data["max_iter"],
            script_dir=script_dir,
            results_dir=results_dir,
            pickle_dir=results_dir / "pickles",
            figure_dir=results_dir / "figures",
            csv_file=results_dir / "results.csv",
        )


def get_problems_to_run(config: BenchmarkConfig) -> list[str]:
    """Get list of problem names matching the configuration."""
    all_problems = get_all_problems()
    selected = []

    for name in all_problems.keys():
        # Only benchmark "-no-weights" problems (skip offset and weighted variants)
        if "-no-weights-" not in name:
            continue

        # Filter by dataset
        if config.datasets is not None:
            if not any(d in name for d in config.datasets):
                continue

        # Filter by regularization
        if config.regularizations is not None:
            if not any(reg in name for reg in config.regularizations):
                continue

        # Filter by distribution
        if config.distributions is not None:
            if not any(dist in name for dist in config.distributions):
                continue

        selected.append(name)

    return sorted(selected)


def run_single_benchmark(
    problem_name: str, library_name: str, config: BenchmarkConfig
) -> tuple[dict, BenchmarkParams]:
    """Run a single benchmark and return results."""
    # Use "auto" storage for glum (enables categorical algorithm via tabmat)
    # Use "dense" for other libraries
    storage = "auto" if library_name == "glum" else "dense"

    params = BenchmarkParams(
        problem_name=problem_name,
        library_name=library_name,
        num_rows=config.num_rows,
        storage=storage,
        threads=config.num_threads,
        regularization_strength=config.reg_strength,
    )

    result, _ = execute_problem_library(
        params,
        iterations=config.iterations,
        diagnostics_level=None,
        standardize=config.standardize,
        max_iter=config.max_iter,
    )

    return result, params


def run_all_benchmarks(config: BenchmarkConfig):
    """Run all configured benchmarks."""
    # Set up output directory
    if config.clear_output and config.results_dir.exists():
        print(f"Clearing output directory: {config.results_dir}")
        shutil.rmtree(config.results_dir)
    config.pickle_dir.mkdir(parents=True, exist_ok=True)

    problems = get_problems_to_run(config)
    available = get_all_libraries()
    libraries = [lib for lib in config.libraries if lib in available]

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
                    result, params = run_single_benchmark(
                        problem_name, library_name, config
                    )

                # Save result
                fname = params.get_result_fname() + ".pkl"
                with open(config.pickle_dir / fname, "wb") as f:
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


def analyze_results(config: BenchmarkConfig) -> pd.DataFrame:
    """Analyze benchmark results and print summary."""
    print()
    print("=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)

    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("display.precision", display_precision)

    results = []

    for fname in config.pickle_dir.glob("*.pkl"):
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

        # Check convergence (n_iter < max_iter means converged)
        converged = True
        if n_iter is not None and n_iter >= config.max_iter:
            converged = False

        results.append(
            {
                "problem_name": params.problem_name,
                "library_name": params.library_name,
                "num_rows": data.get("num_rows"),
                "regularization_strength": reg_strength,
                "storage": params.storage,
                "threads": params.threads,
                "n_iter": n_iter,
                "converged": converged,
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
    config.csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(config.csv_file, index=False)
    print(f"\nExported results to: {config.csv_file}")

    return df.reset_index()


def plot_results(config: BenchmarkConfig):
    """Generate benchmark comparison plots from CSV file.

    Reads from CSV_FILE, which allows regenerating figures without
    re-running benchmarks. The CSV can be committed to the repository.
    """
    print()
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    if not config.csv_file.exists():
        print(f"CSV file not found: {config.csv_file}")
        print("Run ANALYZE_RESULTS first to generate the CSV.")
        return

    df = pd.read_csv(config.csv_file)
    print(f"Reading results from: {config.csv_file}")

    if df["converged"].dtype == object:  # string type from CSV
        df["converged"] = df["converged"] == "True"

    if df.empty:
        print("No data to plot!")
        return

    config.figure_dir.mkdir(exist_ok=True)

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
            pivot_raw = subset.pivot(
                index="distribution",
                columns="library_name",
                values="runtime",
            )
            # Track which cells are unsupported (NaN) before filling
            unsupported = pivot_raw.isna()
            pivot = pivot_raw.fillna(0)

            # Track which cells did not converge
            pivot_converged = (
                subset.pivot(
                    index="distribution",
                    columns="library_name",
                    values="converged",
                )
                .fillna(True)
                .astype(bool)
            )
            not_converged = ~pivot_converged

            # Calculate y-axis limit (10x fastest runtime)
            min_runtime = pivot.values[pivot.values > 0].min()
            y_max = min_runtime * 10

            # Get colors for the libraries in this plot
            plot_colors = [colors.get(lib, "#999999") for lib in pivot.columns]

            # Create bar chart with clipped bars and annotations
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot_clipped = pivot.clip(upper=y_max)
            pivot_clipped.plot(kind="bar", ax=ax, color=plot_colors)

            # Draw hatched bars for unsupported library/distribution combos
            n_dists = len(pivot.index)
            bars = ax.patches
            for i, dist in enumerate(pivot.index):
                for j, lib in enumerate(pivot.columns):
                    if unsupported.loc[dist, lib]:
                        bar_idx = j * n_dists + i
                        bar = bars[bar_idx]
                        x = bar.get_x()
                        width = bar.get_width()
                        lib_color = colors.get(lib, "#999999")
                        # Draw bar with library color + hatch pattern
                        ax.bar(
                            x + width / 2,
                            y_max,
                            width=width,
                            color="white",
                            edgecolor=lib_color,
                            linewidth=2,
                            hatch="//",
                        )
                        # Add "N/A" label
                        ax.text(
                            x + width / 2,
                            y_max / 2,
                            "N/A",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#666666",
                            fontweight="bold",
                        )
                    # Add hatch overlay for non-converged results (keep the runtime bar)
                    elif not_converged.loc[dist, lib]:
                        bar_idx = j * n_dists + i
                        bar = bars[bar_idx]
                        x = bar.get_x()
                        width = bar.get_width()
                        bar_height = min(pivot.loc[dist, lib], y_max)
                        # Overlay hatch pattern on existing bar
                        ax.bar(
                            x + width / 2,
                            bar_height,
                            width=width,
                            color="none",
                            edgecolor="black",
                            linewidth=0.5,
                            hatch="//",
                            alpha=0.5,
                        )
                        # Add "NC" (not converged) label at top of bar
                        ax.text(
                            x + width / 2,
                            bar_height + y_max * 0.02,
                            "NC",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="#cc0000",
                            fontweight="bold",
                        )

            # Add annotations for clipped bars (show original value on bar with arrow)
            for i, dist in enumerate(pivot.index):
                for j, lib in enumerate(pivot.columns):
                    original_val = pivot.loc[dist, lib]
                    if original_val > y_max and not unsupported.loc[dist, lib]:
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
            # Add custom legend entries for N/A and NC
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor="white", edgecolor="gray", hatch="//"))
            labels.append("N/A (not supported)")
            handles.append(Patch(facecolor="white", edgecolor="black", hatch="//"))
            labels.append("NC (not converged)")
            ax.legend(
                handles, labels, title="", bbox_to_anchor=(1.02, 1), loc="upper left"
            )
            # Capitalize x-tick labels
            ax.set_xticklabels(
                [label.get_text().title() for label in ax.get_xticklabels()]
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save
            fname = f"{dataset}-{reg}"
            plt.savefig(config.figure_dir / f"{fname}.png", dpi=300)
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

                # Draw hatched bars for unsupported library/distribution combos
                n_dists = len(pivot_norm.index)
                bars = ax.patches
                for i, dist in enumerate(pivot_norm.index):
                    for j, lib in enumerate(pivot_norm.columns):
                        if unsupported.loc[dist, lib]:
                            bar_idx = j * n_dists + i
                            bar = bars[bar_idx]
                            x = bar.get_x()
                            width = bar.get_width()
                            lib_color = colors.get(lib, "#999999")
                            # Draw bar with library color + hatch pattern
                            ax.bar(
                                x + width / 2,
                                norm_y_max,
                                width=width,
                                color="white",
                                edgecolor=lib_color,
                                linewidth=2,
                                hatch="//",
                            )
                            # Add "N/A" label
                            ax.text(
                                x + width / 2,
                                norm_y_max / 2,
                                "N/A",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="#666666",
                                fontweight="bold",
                            )
                        # Add hatch overlay for non-converged results
                        elif not_converged.loc[dist, lib]:
                            bar_idx = j * n_dists + i
                            bar = bars[bar_idx]
                            x = bar.get_x()
                            width = bar.get_width()
                            bar_height = min(pivot_norm.loc[dist, lib], norm_y_max)
                            # Overlay hatch pattern on existing bar
                            ax.bar(
                                x + width / 2,
                                bar_height,
                                width=width,
                                color="none",
                                edgecolor="black",
                                linewidth=0.5,
                                hatch="//",
                                alpha=0.5,
                            )
                            # Add "NC" (not converged) label at top of bar
                            ax.text(
                                x + width / 2,
                                bar_height + norm_y_max * 0.02,
                                "NC",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="#cc0000",
                                fontweight="bold",
                            )

                # Add annotations for clipped bars (show original value on bar)
                for i, dist in enumerate(pivot_norm.index):
                    for j, lib in enumerate(pivot_norm.columns):
                        original_val = pivot_norm.loc[dist, lib]
                        if original_val > norm_y_max and not unsupported.loc[dist, lib]:
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
                # Add custom legend entries for N/A and NC
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Patch(facecolor="white", edgecolor="gray", hatch="//"))
                labels.append("N/A (not supported)")
                handles.append(Patch(facecolor="white", edgecolor="black", hatch="//"))
                labels.append("NC (not converged)")
                ax.legend(
                    handles,
                    labels,
                    title="",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
                # X-tick labels: show distribution name + glum runtime
                new_labels = []
                for dist in pivot.index:
                    glum_runtime = pivot.loc[dist, "glum"]
                    if glum_runtime > 0:
                        new_labels.append(
                            f"{dist.title()}\n(glum = {glum_runtime:.3f}s)"
                        )
                    else:
                        new_labels.append(dist.title())
                ax.set_xticklabels(new_labels)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                fname_norm = f"{dataset}-{reg}-normalized"
                plt.savefig(config.figure_dir / f"{fname_norm}.png", dpi=300)
                plt.close()
                print(f"Saved: {fname_norm}.png")


def main():
    # Load configuration
    script_dir = Path(__file__).parent
    config_file = script_dir / "config.yaml"
    config = BenchmarkConfig.from_yaml(config_file)

    # Run benchmark steps
    if config.run_benchmarks:
        run_all_benchmarks(config)

    if config.analyze_results:
        analyze_results(config)

    if config.generate_plots:
        plot_results(config)

    # Print summary
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    if config.run_benchmarks or config.analyze_results:
        print(f"Results saved to: {config.results_dir}/")
    if config.generate_plots:
        print(f"Figures saved to: {config.figure_dir}/")

    # Snapshot config for reproducibility
    config.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, config.results_dir / "config.yaml")
    print(f"Config snapshot saved to: {config.results_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
