#!/usr/bin/env python
"""
Benchmark runner script for comparing GLM libraries.

Usage:
    pixi run -e benchmark run-benchmarks

Configuration:
    Edit config.yaml to configure which benchmarks to run. Use param_grid to
    specify combinations of libraries, datasets, regularizations, distributions,
    and alphas. You can also control which steps to run (run_benchmarks,
    analyze_results, generate_plots, update_docs).

Output:
    - glum_benchmarks/results/RUN_NAME/pickles/: Pickle files with detailed results
    - glum_benchmarks/results/RUN_NAME/figures/: PNG plots comparing library performance
    - glum_benchmarks/results/RUN_NAME/results.csv: Summary CSV for reproducibility
"""

from __future__ import annotations

import pickle
import re
import shutil
import warnings
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from ruamel.yaml import YAML

from glum_benchmarks.problems import get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    execute_problem_library,
    get_all_libraries,
    get_params_from_fname,
)

# TODO: Update README and documentation
# TODO: Implement closed form solution for l2-gaussian and put a note in the results
# TODO: Scaling
# TODO: Determine optimal storage type for each library for a fair comparison

# Type aliases for configuration options
Library = Literal["glum", "sklearn", "h2o", "skglm", "celer", "zeros", "glmnet"]
Dataset = Literal[
    "intermediate-insurance",
    "intermediate-housing",
    "narrow-insurance",
    "wide-insurance",
    "square-simulated",
    "categorical-simulated",
]
Regularization = Literal["lasso", "l2", "net"]
Alpha = float  # Valid values: 0.0001, 0.001, 0.01
ALPHA_VALUES = (0.001, 0.01, 0.1)
Distribution = Literal["gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"]
StorageFormat = Literal["auto", "dense", "sparse", "cat"]


class ParamGridEntry(BaseModel):
    """A single entry in the parameter grid.

    Each entry specifies a set of parameter values. The Cartesian product
    is computed within each entry, but entries are unioned (not crossed).
    """

    model_config = ConfigDict(extra="forbid")

    libraries: list[Library] | None = Field(
        default=None, description="Libraries to benchmark (None = all)"
    )
    datasets: list[Dataset] | None = Field(
        default=None, description="Datasets to include (None = all)"
    )
    regularizations: list[Regularization] | None = Field(
        default=None, description="Regularization types (None = all)"
    )
    alphas: list[Alpha] | None = Field(
        default=None, description="Per-observation alpha values (None = all)"
    )
    distributions: list[Distribution] | None = Field(
        default=None, description="Distributions (None = all)"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    # Steps to run
    run_benchmarks: bool = Field(default=True, description="Whether to run benchmarks")
    analyze_results: bool = Field(
        default=True, description="Whether to analyze and print results to CSV"
    )
    generate_plots: bool = Field(
        default=True, description="Whether to generate comparison plots"
    )
    update_docs: bool = Field(
        default=False,
        description="Whether to copy figures to docs/_static and update benchmarks.rst",
    )
    docs_figures: list[str] | None = Field(
        default=None,
        description="Figure names to include in docs/benchmarks.rst (None = all )",
    )
    readme_figures: list[str] | None = Field(
        default=None,
        description="Figure names to include in README.md (None = first figure only)",
    )

    # Output settings
    run_name: str = Field(
        default="docs", description="Subfolder name within results/ directory"
    )
    clear_output: bool = Field(
        default=True, description="Clear entire run_name directory before running"
    )

    # Parameter grid for benchmark selection
    # Each entry in the list specifies a parameter set.
    # Within each entry: Cartesian product of the lists.
    # Across entries: Union (not product).
    param_grid: list[ParamGridEntry] = Field(
        default_factory=lambda: [ParamGridEntry()],
        description="List of parameter sets. Each entry defines a Cartesian product, "
        "entries are unioned. Default runs all combinations.",
    )

    # Benchmark settings
    num_threads: int = Field(
        default=16, ge=1, description="Number of threads for parallel execution"
    )
    standardize: bool = Field(
        default=True, description="Whether to standardize before fitting"
    )
    iterations: int = Field(
        default=2, ge=1, description="Run each benchmark N times, report minimum"
    )
    num_rows: int | None = Field(
        default=None, ge=1, description="Limit rows per dataset (None = full dataset)"
    )
    timeout: int = Field(
        default=100, ge=1, description="Timeout in seconds per benchmark run"
    )
    storage: dict[Library, StorageFormat] = Field(
        default_factory=dict,
        description="Storage format per library (missing libraries default to 'dense')",
    )

    # Path for computing derived paths
    script_dir: Path = Field(
        default=Path("."),
        description="Directory containing config.yaml",
    )

    @property
    def results_dir(self) -> Path:
        """Directory for all results."""
        return self.script_dir / "results" / self.run_name

    @property
    def pickle_dir(self) -> Path:
        """Directory for pickle files."""
        return self.results_dir / "pickles"

    @property
    def figure_dir(self) -> Path:
        """Directory for generated figures."""
        return self.results_dir / "figures"

    @property
    def csv_file(self) -> Path:
        """Path to results CSV file."""
        return self.results_dir / "results.csv"

    @property
    def docs_static_dir(self) -> Path:
        """Directory for docs static files (figures)."""
        return self.script_dir.parent / "docs" / "_static"

    @property
    def benchmarks_rst(self) -> Path:
        """Path to benchmarks.rst docs file."""
        return self.script_dir.parent / "docs" / "benchmarks.rst"

    @property
    def readme_file(self) -> Path:
        """Path to README.md file."""
        return self.script_dir.parent / "README.md"

    @model_validator(mode="after")
    def validate_config(self) -> BenchmarkConfig:
        """Validate cross-field constraints."""
        # Ensure run_name is not empty
        if not self.run_name or not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        yaml = YAML(typ="safe", pure=True)
        with open(yaml_path) as f:
            data = yaml.load(f)

        # Add script_dir for path computation
        data["script_dir"] = yaml_path.parent

        return cls.model_validate(data)


def _parse_problem_name(name: str) -> tuple[str, str, str]:
    """Parse problem name into (dataset, regularization, distribution).

    Problem name format: {dataset}-no-weights-{regularization}-{distribution}
    where dataset is 2 parts (e.g., "intermediate-insurance"),
    regularization is 1 part (lasso/l2/net),
    and distribution may contain hyphens (e.g., "tweedie-p=1.5").
    """
    parts = name.split("-")
    dataset = "-".join(parts[:2])
    reg = parts[4]
    dist = "-".join(parts[5:])
    return dataset, reg, dist


def get_benchmark_combinations(
    config: BenchmarkConfig,
) -> list[tuple[str, str, float]]:
    """Get list of (problem_name, library, alpha) tuples to benchmark.

    Cartesian product within each entry, union across entries.

    Returns:
        List of (problem_name, library_name, alpha) tuples to run.
    """

    all_problems = get_all_problems()
    available_libraries = list(get_all_libraries())

    # Filter to "-no-weights-" problems only
    base_problems = [name for name in all_problems.keys() if "-no-weights-" in name]

    # Build a lookup: (dataset, reg, dist) -> problem_name
    problem_lookup = {}
    for name in base_problems:
        dataset, reg, dist = _parse_problem_name(name)
        problem_lookup[(dataset, reg, dist)] = name

    # Get all valid values for each dimension
    all_datasets = sorted({_parse_problem_name(n)[0] for n in base_problems})
    all_regs = sorted({_parse_problem_name(n)[1] for n in base_problems})
    all_dists = sorted({_parse_problem_name(n)[2] for n in base_problems})
    all_alphas = list(ALPHA_VALUES)

    combinations: set[tuple[str, str, float]] = set()

    # Print configuration per parameter grid entry
    print("=" * 70)
    print("BENCHMARK CONFIGURATION")
    print("=" * 70)
    for i, entry in enumerate(config.param_grid, 1):
        # Use entry values or defaults (all)
        libraries = entry.libraries if entry.libraries else available_libraries
        datasets = entry.datasets if entry.datasets else all_datasets
        regs = entry.regularizations if entry.regularizations else all_regs
        dists = entry.distributions if entry.distributions else all_dists
        alphas = entry.alphas if entry.alphas else all_alphas

        print(f"Parameter Set {i}:")
        print(f"  Libraries: {libraries}")
        print(f"  Datasets: {datasets}")
        print(f"  Regularizations: {regs}")
        print(f"  Distributions: {dists}")
        print(f"  Alphas: {alphas}")
        print()

        # Cartesian product within this entry
        for lib, dataset, reg, dist, alpha in product(
            libraries, datasets, regs, dists, alphas
        ):
            key = (dataset, reg, dist)
            if key in problem_lookup:
                # Only add if library is actually available
                if lib in available_libraries:
                    combinations.add((problem_lookup[key], lib, alpha))

    print(f"Total benchmark runs: {len(combinations)}")
    print("=" * 70)
    print()

    return sorted(combinations)


def run_single_benchmark(
    problem_name: str, library_name: str, alpha: float, config: BenchmarkConfig
) -> tuple[dict, BenchmarkParams]:
    """Run a single benchmark and return results.

    If the benchmark exceeds the configured timeout, returns a result with
    runtime=timeout and timed_out=True.
    """
    # Get library-specific storage from config, default to "dense"
    lib_key: Library = library_name  # type: ignore[assignment]
    storage = config.storage.get(lib_key, "dense")

    params = BenchmarkParams(
        problem_name=problem_name,
        library_name=library_name,
        num_rows=config.num_rows,
        storage=storage,
        threads=config.num_threads,
        alpha=alpha,
    )

    # Pass timeout to execute_problem_library for per-iteration timeout handling
    try:
        result, _ = execute_problem_library(
            params,
            iterations=config.iterations,
            diagnostics_level=None,
            standardize=config.standardize,
            timeout=config.timeout,
        )
        result["timed_out"] = False
    except TimeoutError:
        # All iterations timed out
        result = {
            "runtime": float(config.timeout),
            "timed_out": True,
            "intercept": None,
            "coef": None,
            "n_iter": None,
        }

    return result, params


def run_all_benchmarks(config: BenchmarkConfig):
    """Run all configured benchmarks."""
    # Set up output directory
    if config.clear_output and config.results_dir.exists():
        print(f"Clearing output directory: {config.results_dir}")
        shutil.rmtree(config.results_dir)
    config.pickle_dir.mkdir(parents=True, exist_ok=True)

    # Get benchmark combinations (problem, library, alpha) from param_grid
    combinations = get_benchmark_combinations(config)

    total = len(combinations)
    current = 0

    for problem_name, library_name, alpha in combinations:
        current += 1
        label = f"{library_name} / {problem_name} (α={alpha})"
        print(f"[{current}/{total}] {label}", end=" ", flush=True)

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
                    problem_name, library_name, alpha, config
                )

            # Save result
            fname = params.get_result_fname() + ".pkl"
            with open(config.pickle_dir / fname, "wb") as f:
                pickle.dump(result, f)

            if result.get("timed_out"):
                print(f"-> TIMEOUT ({config.timeout}s)")
            elif len(result) > 0 and "runtime" in result:
                print(f"-> {result['runtime']:.4f}s")
            else:
                print("-> (skipped)")

            # Print captured warnings
            for w in caught_warnings:
                print(f"  Warning: {w.message}")

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

        if not data:
            continue

        params = get_params_from_fname(fname.name)
        if params.problem_name is None:
            continue
        problem = get_all_problems()[params.problem_name]

        # Handle timed out runs (no coef)
        timed_out = data.get("timed_out", False)
        coefs = data.get("coef")

        if coefs is None and not timed_out:
            # Skipped run (not supported by library)
            continue

        # Calculate runtime per iteration
        n_iter = data.get("n_iter")
        runtime = data.get("runtime")
        if n_iter is not None and n_iter > 0:
            runtime_per_iter = runtime / n_iter
        else:
            runtime_per_iter = runtime

        # Calculate coefficient norms (0 for timed out)
        if coefs is not None:
            l1_norm: float = np.sum(np.abs(coefs))
            l2_norm: float = np.sum(coefs**2)
            num_nonzero_coef: int = np.sum(np.abs(coefs) > 1e-8)
        else:
            l1_norm = 0.0
            l2_norm = 0.0
            num_nonzero_coef = 0

        # Get regularization strength from params or problem default
        alpha = problem.alpha if params.alpha is None else params.alpha

        # Check convergence:
        # 1. timed_out=True means we hit the benchmark timeout
        # 2. n_iter >= max_iter means the library hit its internal iteration limit
        max_iter = data.get("max_iter")
        hit_max_iter = (
            n_iter is not None and max_iter is not None and n_iter >= max_iter
        )
        converged = not timed_out and not hit_max_iter

        results.append(
            {
                "problem_name": params.problem_name,
                "library_name": params.library_name,
                "num_rows": data.get("num_rows"),
                "alpha": alpha,
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
    problem_id_cols = ["problem_name", "num_rows", "alpha"]
    df = df.set_index(problem_id_cols).sort_values("library_name").sort_index()

    # Calculate relative objective value (how far from best).
    # Use transform to preserve alignment when index has duplicates.
    best_obj = df.groupby(level=[0, 1, 2])["obj_val"].transform("min")
    df["rel_obj_val"] = df["obj_val"] - best_obj

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


def _render_bar_chart(
    pivot: pd.DataFrame,
    unsupported: pd.DataFrame,
    not_converged: pd.DataFrame,
    colors: dict,
    y_max: float,
    title: str,
    ylabel: str,
    dark_mode: bool = False,
    x_labels: list[str] | None = None,
    show_baseline: bool = False,
) -> plt.Figure:
    """Render a bar chart with support for light/dark mode."""
    # Apply dark mode style if requested
    style_context = plt.style.context("dark_background") if dark_mode else nullcontext()

    with style_context:
        plot_colors = [colors.get(lib, "#999999") for lib in pivot.columns]
        pivot_clipped = pivot.clip(upper=y_max)

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_clipped.plot(kind="bar", ax=ax, color=plot_colors)

        # Colors for dark mode
        na_bg = "#1a1a1a" if dark_mode else "white"
        na_text = "#888888" if dark_mode else "#666666"
        nc_edge = "white" if dark_mode else "black"
        nc_text = "#ff6666" if dark_mode else "#cc0000"
        arrow_color = "white" if dark_mode else "black"

        # Draw hatched bars for unsupported library/reg_combo combos
        n_combos = len(pivot.index)
        bars = ax.patches
        for i, combo in enumerate(pivot.index):
            for j, lib in enumerate(pivot.columns):
                if unsupported.loc[combo, lib]:
                    bar_idx = j * n_combos + i
                    bar = bars[bar_idx]
                    x = bar.get_x()
                    width = bar.get_width()
                    lib_color = colors.get(lib, "#999999")
                    ax.bar(
                        x + width / 2,
                        y_max,
                        width=width,
                        color=na_bg,
                        edgecolor=lib_color,
                        linewidth=2,
                        hatch="//",
                    )
                    ax.text(
                        x + width / 2,
                        y_max / 2,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=na_text,
                        fontweight="bold",
                    )
                elif not_converged.loc[combo, lib]:
                    bar_idx = j * n_combos + i
                    bar = bars[bar_idx]
                    x = bar.get_x()
                    width = bar.get_width()
                    bar_height = min(pivot.loc[combo, lib], y_max)
                    ax.bar(
                        x + width / 2,
                        bar_height,
                        width=width,
                        color="none",
                        edgecolor=nc_edge,
                        linewidth=0.5,
                        hatch="//",
                        alpha=0.5,
                    )
                    ax.text(
                        x + width / 2,
                        bar_height + y_max * 0.02,
                        "NC",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=nc_text,
                        fontweight="bold",
                    )

        # Add annotations for clipped bars
        for i, combo in enumerate(pivot.index):
            for j, lib in enumerate(pivot.columns):
                original_val = pivot.loc[combo, lib]
                if original_val > y_max and not unsupported.loc[combo, lib]:
                    bar_idx = j * n_combos + i
                    bar = bars[bar_idx]
                    x = bar.get_x() + bar.get_width() / 2
                    # Format: use .1f for normalized (ratios), .4f for absolute
                    fmt = (
                        f"{original_val:.1f}x"
                        if show_baseline
                        else f"{original_val:.4f}"
                    )
                    ax.text(
                        x,
                        y_max * 0.75,
                        fmt,
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        rotation=90,
                        color=arrow_color,
                    )
                    ax.annotate(
                        "",
                        xy=(x, y_max),
                        xytext=(x, y_max * 0.88),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                    )

        ax.set_ylim(0, y_max * 1.08)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

        if show_baseline:
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=na_bg, edgecolor="gray", hatch="//"))
        labels.append("N/A (not supported)")
        handles.append(Patch(facecolor=na_bg, edgecolor=nc_edge, hatch="//"))
        labels.append("NC (not converged)")
        ax.legend(handles, labels, title="", bbox_to_anchor=(1.02, 1), loc="upper left")

        if x_labels:
            ax.set_xticklabels(x_labels)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

    return fig


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
    df = df.copy()
    parsed = df["problem_name"].apply(_parse_problem_name)
    df["dataset"] = parsed.apply(lambda x: x[0])
    reg_map = {"lasso": "lasso", "l2": "ridge", "net": "elastic-net"}
    df["regularization"] = parsed.apply(lambda x: reg_map.get(x[1], x[1]))
    df["distribution"] = parsed.apply(lambda x: x[2])

    # Create a combined regularization column (type + strength)
    df["reg_combo"] = df.apply(
        lambda row: f"{row['regularization']} (α={row['alpha']})",
        axis=1,
    )

    # Drop duplicates (keep latest result for each unique combo)
    df = df.drop_duplicates(
        subset=[
            "dataset",
            "distribution",
            "regularization",
            "alpha",
            "library_name",
        ],
        keep="last",
    )

    # Ensure library colors are consistent across plots
    colors = {
        "glum": "#a6cee3",
        "h2o": "#fdbf6f",
        "glmnet": "#b15928",
        "sklearn": "#1f78b4",
        "skglm": "#33a02c",
        "celer": "#fb9a99",
    }

    # Generate one plot per dataset/distribution combo
    for dataset in df["dataset"].unique():
        for dist in df["distribution"].unique():
            subset = df[(df["dataset"] == dataset) & (df["distribution"] == dist)]

            if subset.empty:
                continue

            print(f"  Plotting {dataset}-{dist}: {len(subset)} rows")
            print(f"    reg_combos: {subset['reg_combo'].unique().tolist()}")

            # Pivot for plotting: reg_combo on x-axis, libraries as bars
            pivot_raw = subset.pivot(
                index="reg_combo",
                columns="library_name",
                values="runtime",
            )
            # Track which cells are unsupported (NaN) before filling
            unsupported = pivot_raw.isna()
            pivot = pivot_raw.fillna(0)

            # Track which cells did not converge
            pivot_converged = (
                subset.pivot(
                    index="reg_combo",
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

            # Title
            title_dataset = dataset.replace(" ", "-").title()
            title_dist = dist.replace(" ", "-").title()
            title = f"{title_dataset}-{title_dist}"

            # Determine if this figure needs a dark mode version (for README)
            fname = f"{dataset}-{dist}"
            readme_figs = config.readme_figures or []
            needs_dark = f"{fname}.png" in readme_figs

            # Generate light mode plot
            fig = _render_bar_chart(
                pivot=pivot,
                unsupported=unsupported,
                not_converged=not_converged,
                colors=colors,
                y_max=y_max,
                title=title,
                ylabel="run time (s)",
                dark_mode=False,
            )
            fig.savefig(config.figure_dir / f"{fname}.png", dpi=300)
            plt.close(fig)
            print(f"Saved: {fname}.png")

            # Generate dark mode version if needed for README
            if needs_dark:
                fig = _render_bar_chart(
                    pivot=pivot,
                    unsupported=unsupported,
                    not_converged=not_converged,
                    colors=colors,
                    y_max=y_max,
                    title=title,
                    ylabel="run time (s)",
                    dark_mode=True,
                )
                fig.savefig(config.figure_dir / f"{fname}_dark.png", dpi=300)
                plt.close(fig)
                print(f"Saved: {fname}_dark.png")

            # Generate normalized plot (glum = 1.0)
            if "glum" in pivot.columns:
                pivot_norm = pivot.div(pivot["glum"], axis=0)
                norm_y_max = 10.0

                # X-tick labels with glum runtime
                x_labels = []
                for combo in pivot.index:
                    glum_runtime = pivot.loc[combo, "glum"]
                    if glum_runtime > 0:
                        x_labels.append(f"{combo}\n(glum = {glum_runtime:.3f}s)")
                    else:
                        x_labels.append(combo)

                fname_norm = f"{dataset}-{dist}-normalized"
                needs_dark_norm = f"{fname_norm}.png" in readme_figs

                # Generate light mode normalized plot
                fig = _render_bar_chart(
                    pivot=pivot_norm,
                    unsupported=unsupported,
                    not_converged=not_converged,
                    colors=colors,
                    y_max=norm_y_max,
                    title=f"{title} (normalized)",
                    ylabel="run time relative to glum",
                    dark_mode=False,
                    x_labels=x_labels,
                    show_baseline=True,
                )
                fig.savefig(config.figure_dir / f"{fname_norm}.png", dpi=300)
                plt.close(fig)
                print(f"Saved: {fname_norm}.png")

                # Generate dark mode version if needed for README
                if needs_dark_norm:
                    fig = _render_bar_chart(
                        pivot=pivot_norm,
                        unsupported=unsupported,
                        not_converged=not_converged,
                        colors=colors,
                        y_max=norm_y_max,
                        title=f"{title} (normalized)",
                        ylabel="run time relative to glum",
                        dark_mode=True,
                        x_labels=x_labels,
                        show_baseline=True,
                    )
                    fig.savefig(config.figure_dir / f"{fname_norm}_dark.png", dpi=300)
                    plt.close(fig)
                    print(f"Saved: {fname_norm}_dark.png")


# Markers for auto-generated content in docs
RST_START_MARKER = ".. BENCHMARK_FIGURES_START"
RST_END_MARKER = ".. BENCHMARK_FIGURES_END"
MD_START_MARKER = "<!-- BENCHMARK_FIGURES_START -->"
MD_END_MARKER = "<!-- BENCHMARK_FIGURES_END -->"


def update_docs(config: BenchmarkConfig):
    """Copy figures to docs/_static and update benchmarks.rst and README.md.

    Uses marker comments to safely replace only the auto-generated figure
    references in each file. Figures to include can be specified via
    docs_figures and readme_figures config options.
    """
    print()
    print("=" * 60)
    print("UPDATING DOCS")
    print("=" * 60)

    if not config.figure_dir.exists():
        print(f"Figure directory not found: {config.figure_dir}")
        print("Run GENERATE_PLOTS first to create figures.")
        return

    # Find all generated figures
    all_figures = sorted(config.figure_dir.glob("*.png"))
    if not all_figures:
        print("No figures found to copy.")
        return

    # Get available figure names (without path)
    available = {f.name for f in all_figures}

    # Determine which figures to use for docs
    if config.docs_figures is not None:
        # Use explicitly specified figures
        docs_fig_names = [f for f in config.docs_figures if f in available]
        missing = set(config.docs_figures) - available
        if missing:
            print(f"Warning: docs_figures not found: {missing}")
    else:
        # Default: all generated figures
        docs_fig_names = sorted(f.name for f in all_figures)

    # Determine which figures to use for README
    if config.readme_figures is not None:
        # Use explicitly specified figures
        readme_fig_names = [f for f in config.readme_figures if f in available]
        missing = set(config.readme_figures) - available
        if missing:
            print(f"Warning: readme_figures not found: {missing}")
    else:
        # Default: first non-normalized figure only
        non_norm = sorted(f.name for f in all_figures if "normalized" not in f.name)
        readme_fig_names = [non_norm[0]] if non_norm else []

    # Collect all figures needed for copying
    all_needed = set(docs_fig_names) | set(readme_fig_names)

    # Copy figures to docs/_static
    config.docs_static_dir.mkdir(parents=True, exist_ok=True)
    for fig_name in sorted(all_needed):
        src = config.figure_dir / fig_name
        dest = config.docs_static_dir / fig_name
        shutil.copy2(src, dest)
        print(f"Copied: {fig_name} -> docs/_static/")

    # Update benchmarks.rst
    if docs_fig_names:
        rst_lines = [RST_START_MARKER, ""]
        for fig_name in docs_fig_names:
            rst_lines.append(f".. image:: _static/{fig_name}")
            rst_lines.append("   :width: 700")
            rst_lines.append("")
        rst_lines.append(RST_END_MARKER)
        rst_new_content = "\n".join(rst_lines)

        if config.benchmarks_rst.exists():
            with open(config.benchmarks_rst) as f:
                rst_content = f.read()

            if RST_START_MARKER in rst_content and RST_END_MARKER in rst_content:
                pattern = re.compile(
                    rf"{re.escape(RST_START_MARKER)}.*?{re.escape(RST_END_MARKER)}",
                    re.DOTALL,
                )
                updated_rst = pattern.sub(rst_new_content, rst_content)
                with open(config.benchmarks_rst, "w") as f:
                    f.write(updated_rst)
                print(f"\nUpdated: {config.benchmarks_rst}")
                print(f"Inserted {len(docs_fig_names)} figure references.")
            else:
                print(f"\nMarkers not found in {config.benchmarks_rst}")
                print(f"Add: {RST_START_MARKER} and {RST_END_MARKER}")
        else:
            print(f"\nbenchmarks.rst not found: {config.benchmarks_rst}")

    # Update README.md
    # Use GitHub's light/dark mode syntax with actual dark images when available
    if readme_fig_names:
        md_lines = [MD_START_MARKER]
        for fig_name in readme_fig_names:
            # Check if dark mode version exists
            base_name = fig_name.replace(".png", "")
            dark_name = f"{base_name}_dark.png"
            dark_exists = (config.figure_dir / dark_name).exists()

            if dark_exists:
                # Copy dark version too
                shutil.copy2(
                    config.figure_dir / dark_name,
                    config.docs_static_dir / dark_name,
                )
                print(f"Copied: {dark_name} -> docs/_static/")
                # Use actual dark image
                light_ref = f"docs/_static/{fig_name}#gh-light-mode-only"
                dark_ref = f"docs/_static/{dark_name}#gh-dark-mode-only"
            else:
                # Use same image for both modes
                light_ref = f"docs/_static/{fig_name}#gh-light-mode-only"
                dark_ref = f"docs/_static/{fig_name}#gh-dark-mode-only"

            md_lines.append(f"![Benchmark results]({light_ref})")
            md_lines.append(f"![Benchmark results]({dark_ref})")
        md_lines.append(MD_END_MARKER)
        md_new_content = "\n".join(md_lines)

        if config.readme_file.exists():
            with open(config.readme_file) as f:
                md_content = f.read()

            if MD_START_MARKER in md_content and MD_END_MARKER in md_content:
                pattern = re.compile(
                    rf"{re.escape(MD_START_MARKER)}.*?{re.escape(MD_END_MARKER)}",
                    re.DOTALL,
                )
                updated_md = pattern.sub(md_new_content, md_content)
                with open(config.readme_file, "w") as f:
                    f.write(updated_md)
                print(f"\nUpdated: {config.readme_file}")
                print(f"Inserted {len(readme_fig_names)} figure references.")
            else:
                print(f"\nMarkers not found in {config.readme_file}")
                print(f"Add: {MD_START_MARKER} and {MD_END_MARKER}")
        else:
            print(f"\nREADME.md not found: {config.readme_file}")


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

    if config.update_docs:
        update_docs(config)

    # Print summary
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    if config.run_benchmarks or config.analyze_results:
        print(f"Results saved to: {config.results_dir}/")
    if config.generate_plots:
        print(f"Figures saved to: {config.figure_dir}/")
    if config.update_docs:
        print(f"Docs updated: {config.benchmarks_rst}")
        print(f"README updated: {config.readme_file}")

    # Snapshot config for reproducibility
    config.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, config.results_dir / "config.yaml")
    print(f"Config snapshot saved to: {config.results_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
