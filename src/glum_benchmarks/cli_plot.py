"""Generate benchmark comparison plots from benchmark output directory."""

import os
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import pandas as pd

from .cli_analyze import (
    _extract_dict_results_to_pd_series,
    _identify_parameter_fnames,
    _load_benchmark_results,
)
from .util import BenchmarkParams


@click.command()
@click.option(
    "--output_dir",
    type=str,
    default="benchmark_output",
    help="Directory containing benchmark results (default: benchmark_output)",
)
@click.option(
    "--figure_dir",
    type=str,
    default="benchmark_figures",
    help="Directory to save generated figures (default: benchmark_figures)",
)
@click.option(
    "--problems",
    type=str,
    default=None,
    help="Comma-separated list of problem prefixes to plot "
    "(e.g., 'narrow-insurance,intermediate-housing'). If None, plots all.",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for saved PNG figures (default: 300)",
)
def cli_plot(output_dir: str, figure_dir: str, problems: Optional[str], dpi: int):
    """
    Generate benchmark comparison plots from results directory.

    Examples:
        # Plot all benchmarks from default directory
        glm_benchmarks_plot

        # Plot from specific directory
        glm_benchmarks_plot --output_dir my_benchmarks --figure_dir my_figures

        # Plot only specific problems
        glm_benchmarks_plot --problems narrow-insurance,intermediate-housing
    """
    # Create output directory
    os.makedirs(figure_dir, exist_ok=True)

    # Load benchmark results
    print(f"Loading benchmark results from {output_dir}...")
    file_names = _identify_parameter_fnames(output_dir, BenchmarkParams())

    raw_results = {
        fname: _load_benchmark_results(output_dir, fname) for fname in file_names
    }
    formatted_results = [
        _extract_dict_results_to_pd_series(name, res)
        for name, res in raw_results.items()
        if len(res) > 0
    ]

    if not formatted_results:
        print("No benchmark results found!")
        return

    df = pd.DataFrame.from_records(formatted_results)

    # Extract distribution from problem name
    df["distribution"] = (
        df["problem_name"]
        .str.split("-")
        .apply(lambda x: x[-2] if "5" in x[-1] else x[-1])
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["problem_name", "library_name"], keep="last")

    # Filter problems if specified
    if problems:
        problem_list = [p.strip() for p in problems.split(",")]
        df = df[df["problem_name"].str.contains("|".join(problem_list))]
        if df.empty:
            print(f"No results found for problems: {problems}")
            return

    # Determine which problem sets we have
    problem_prefixes = set()
    for idx, row in df.iterrows():
        parts = row["problem_name"].split("-")
        # Extract dataset name (e.g., "narrow-insurance", "intermediate-housing")
        if len(parts) >= 3:
            dataset = "-".join(
                parts[:-3]
            )  # Everything before "no-weights-{reg}-{dist}"
            problem_prefixes.add(dataset)

    print(f"Found datasets: {', '.join(sorted(problem_prefixes))}")

    # Generate plots for each problem/regularization combination
    plot_count = 0
    for prob_name in sorted(problem_prefixes):
        for reg in ["l2", "lasso"]:
            plot_df = df[
                df["problem_name"].str.contains(reg)
                & df["problem_name"].str.contains(prob_name)
            ].copy()

            if plot_df.empty:
                continue

            plot_df = plot_df.set_index(["distribution"])[["runtime", "library_name"]]

            try:
                plot_df = plot_df.pivot(columns="library_name")
                plot_df.columns = plot_df.columns.get_level_values(1)
                plot_df = plot_df.sort_index(axis=1)

                # Drop columns (libraries) that have ALL NaN values (never ran)
                plot_df = plot_df.dropna(axis=1, how="all")

                # Skip if no data remains
                if plot_df.empty:
                    print(f"Skipping {prob_name}-{reg}: No data available")
                    continue

                # Replace NaN with 0 for plotting (matplotlib will show as no bar)
                plot_df = plot_df.fillna(0)

                plot_df.index = [x.title() for x in plot_df.index]
            except ValueError as e:
                print(f"Skipping {prob_name}-{reg}: {e}")
                continue

            # Hard limit y-axis to 15 seconds
            ylim = 15

            # Clean title: just dataset name + regularization type
            clean_prob_name = prob_name.replace("-no-weights", "").replace("-no", "")
            title = (
                clean_prob_name.title() + "-" + ("Lasso" if reg == "lasso" else "Ridge")
            )

            # Create figure manually to avoid gaps for missing data
            fig, ax = plt.subplots(figsize=(6, 3))

            # Get unique libraries and assign consistent colors using Paired colormap
            libraries = plot_df.columns
            cmap = plt.cm.Paired
            n_colors = len(libraries)
            colors = {
                lib: cmap(i / max(n_colors - 1, 1)) for i, lib in enumerate(libraries)
            }

            # Plot bars distribution by distribution
            x_pos: float = 0
            x_labels: list[str] = []
            x_ticks: list[float] = []
            bar_width = 0.8 / len(libraries)

            for dist_idx, dist in enumerate(plot_df.index):
                # Only plot libraries with data for this distribution
                dist_data = plot_df.loc[dist]
                libs_with_data = dist_data[dist_data > 0]

                if len(libs_with_data) > 0:
                    # Plot bars for this distribution
                    for lib_idx, (lib, runtime) in enumerate(libs_with_data.items()):
                        color = colors[lib]
                        ax.bar(
                            x_pos + lib_idx * bar_width,
                            runtime,
                            bar_width,
                            color=color,
                            label=lib if dist_idx == 0 else "",
                        )

                    # Center tick position for this group
                    x_ticks.append(x_pos + (len(libs_with_data) - 1) * bar_width / 2)
                    x_labels.append(dist)
                    x_pos += (
                        len(libs_with_data) * bar_width + 0.2
                    )  # Gap between distributions

            ax.set_ylim([0, ylim])
            ax.set_ylabel("run time (s)")
            ax.set_title(title)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

            # Add legend (remove duplicates)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                bbox_to_anchor=(1, 1),
                loc="upper left",
                ncol=1,
            )

            ax = plt.gca()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

            # Annotate bars that exceed y-limit
            for p in ax.patches:
                x = p.get_x()
                width = p.get_width()
                y = p.get_height()
                if y > ylim + 0.1:
                    arrow_x = x + width / 2

                    # Arrow at the very top - simple arrow style like in reference
                    ax.annotate(
                        "",
                        xy=(arrow_x, ylim - 0.1),  # Arrow tip near top
                        xytext=(arrow_x, ylim - 1.0),  # Arrow starts lower
                        arrowprops=dict(
                            arrowstyle="simple",
                            fc="black",
                            ec="black",
                        ),
                    )

                    # Text below the arrow, extending downward from arrow base
                    ax.annotate(
                        f"{y:.1f}",
                        (arrow_x, ylim - 1.1),  # Start just below arrow
                        fontsize=14,
                        rotation="vertical",
                        ha="center",
                        va="top",  # Text extends downward
                    )

            plt.tight_layout()

            # Save figure
            base_name = f"{prob_name}-{reg}"
            png_path = Path(figure_dir) / f"{base_name}.png"
            pdf_path = Path(figure_dir) / f"{base_name}.pdf"

            plt.savefig(png_path, dpi=dpi)
            plt.savefig(pdf_path)
            plt.close()

            plot_count += 1
            print(f"Saved: {base_name}.png, {base_name}.pdf")

    print(f"\nGenerated {plot_count} plots in {figure_dir}/")


if __name__ == "__main__":
    cli_plot()
