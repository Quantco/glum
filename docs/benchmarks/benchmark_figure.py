# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

# %%
# !rm -r benchmark_output

# %%
base_cmd = (
    "glm_benchmarks_run --threads 6 --num_rows {n} --storage {s} "
    "--problem_name {p} --library_name {lib}"
)

problems = []
for p in ["narrow-insurance", "intermediate-insurance", "wide-insurance"]:
    for reg in ["l2", "lasso"]:
        for dist in ["tweedie-p=1.5", "poisson", "gaussian", "gamma", "binomial"]:
            problems.append(f"{p}-no-weights-{reg}-{dist}")

p = "intermediate-housing"
for reg in ["l2", "lasso"]:
    for dist in ["gaussian", "gamma", "binomial"]:
        problems.append(f"{p}-no-weights-{reg}-{dist}")

n = 500000

# %%
# run r-glmnet and h2o benchmarks, sparse storage works best.
s = "sparse"
for lib in ["r-glmnet", "h2o"]:
    for p in problems:
        if lib == "r-glmnet" and p == "wide-insurance-no-weights-l2-poisson":
            continue
        cmd = base_cmd.format(n=n, s=s, p=p, lib=lib)
        print(cmd)
        subprocess.run(cmd.split(" "))

# run glum benchmarks where auto storage works best.
lib = "glum"
s = "auto"
for p in problems:
    cmd = base_cmd.format(n=n, s=s, p=p, lib=lib)
    print(cmd)
    subprocess.run(cmd.split(" "))

analyze_cmd = "glm_benchmarks_analyze --export benchmark_data.csv"
subprocess.run(analyze_cmd.split(" "))

# %%
df = pd.read_csv("benchmark_data.csv")
df.drop(
    [
        "storage",
        "num_rows",
        "regularization_strength",
        "offset",
        "threads",
        "single_precision",
        "cv",
        "hessian_approx",
        "diagnostics_level",
    ],
    axis=1,
    inplace=True,
)
df["distribution"] = (
    df["problem_name"].str.split("-").apply(lambda x: x[-2] if "5" in x[-1] else x[-1])
)

# %%
# %config InlineBackend.figure_format='retina'

# %%
for prob_name in ["narrow-insurance", "intermediate-insurance", "intermediate-housing"]:
    for reg in ["l2", "lasso"]:
        plot_df = (
            df[
                df["problem_name"].str.contains(reg)
                & df["problem_name"].str.contains(prob_name)
            ]
            .copy()
            .set_index(["distribution"])[["runtime", "library_name"]]
        )
        plot_df = plot_df.pivot(columns="library_name")
        plot_df.columns = plot_df.columns.get_level_values(1)
        plot_df = plot_df.sort_index(axis=1).rename(columns={"r-glmnet": "glmnet"})
        plot_df.index = [x.title() for x in plot_df.index]

        title = prob_name.title() + "-" + ("Lasso" if reg == "lasso" else "Ridge")
        plot_df.plot.bar(
            ylim=[0, 4],
            title=title,
            legend=False,
            figsize=(6, 3),
            width=0.8,
            ylabel="run time (s)",
            yticks=[0, 1, 2, 3, 4],
            cmap="Paired",
        )
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        plt.xticks(rotation=45, ha="right")

        ax = plt.gca()

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        for p in ax.patches:
            x = p.get_x()  # type: ignore
            y = p.get_height()  # type: ignore
            if y > 4.1:
                text_x = x + 0.02
                text_y = 2.75 if y > 10 else 2.95
                ax.annotate(
                    f"{y:.1f}",
                    (text_x, text_y),
                    fontsize=14,
                    rotation="vertical",
                )
                arrow_x = text_x + 0.11
                arrow_y = 3.5
                ax.annotate(
                    "",
                    xy=(arrow_x, arrow_y + 0.5),
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle="->"),
                )

        plt.tight_layout()
        fp = f"../_static/{prob_name}-{reg}.png"
        plt.savefig(fp, dpi=300)
        fp = f"../_static/{prob_name}-{reg}.pdf"
        plt.savefig(fp)
        plt.show()

# %%
for prob_name in ["wide-insurance"]:
    for reg in ["l2", "lasso"]:
        plot_df = (
            df[
                df["problem_name"].str.contains(reg)
                & df["problem_name"].str.contains(prob_name)
            ]
            .copy()
            .set_index(["distribution"])[["runtime", "library_name"]]
        )
        plot_df = plot_df.pivot(columns="library_name")
        plot_df.columns = plot_df.columns.get_level_values(1)
        plot_df = plot_df.sort_index(axis=1).rename(columns={"r-glmnet": "glmnet"})
        plot_df.index = [x.title() for x in plot_df.index]

        title = prob_name.title() + "-" + ("Lasso" if reg == "lasso" else "Ridge")
        plot_df.plot.bar(
            ylim=[0, 15],
            title=title,
            legend=False,
            figsize=(6, 3),
            width=0.8,
            ylabel="run time (s)",
            yticks=[0, 3, 6, 9, 12, 15],
            cmap="Paired",
        )
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
        plt.xticks(rotation=45, ha="right")

        ax = plt.gca()

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        for p in ax.patches:
            x = p.get_x()  # type: ignore
            y = p.get_height()  # type: ignore
            if y > 15.1:
                text_x = x + 0.01
                text_y = 8 if y > 1000 else (9 if y > 100 else 10)
                ax.annotate(
                    f"{y:.1f}",
                    (text_x, text_y),
                    fontsize=14,
                    rotation="vertical",
                )
                arrow_x = text_x + 0.13
                arrow_y = 13
                ax.annotate(
                    "",
                    xy=(arrow_x, arrow_y + 2),
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle="->"),
                )

        plt.tight_layout()
        fp = f"../_static/{prob_name}-{reg}.png"
        plt.savefig(fp, dpi=300)
        fp = f"../_static/{prob_name}-{reg}.pdf"
        plt.savefig(fp)
        plt.show()

# %%
prob_name = "intermediate-insurance"
reg = "lasso"
plot_df = (
    df[
        df["problem_name"].str.contains(reg)
        & df["problem_name"].str.contains(prob_name)
    ]
    .copy()
    .set_index(["distribution"])[["runtime", "library_name"]]
)
plot_df = plot_df.pivot(columns="library_name")
plot_df.columns = plot_df.columns.get_level_values(1)
plot_df = plot_df.sort_index(axis=1).rename(columns={"r-glmnet": "glmnet"})
plot_df.index = [x.title() for x in plot_df.index]

plot_df.plot.bar(
    ylim=[0, 5],
    legend=False,
    figsize=(6, 3),
    width=0.8,
    ylabel="run time (s)",
    yticks=[0, 1, 2, 3, 4, 5],
    cmap="Paired",
)
plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1)
plt.xticks(rotation=45, ha="right")

ax = plt.gca()

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

for p in ax.patches:
    x = p.get_x()  # type: ignore
    y = p.get_height()  # type: ignore
    if y > 5.1:
        text_x = x + 0.04
        text_y = 3.2
        ax.annotate(
            f"{y:.1f}",
            (text_x, text_y),
            fontsize=14,
            rotation="vertical",
        )
        arrow_x = text_x + 0.15
        arrow_y = 4.2
        ax.annotate(
            "",
            xy=(arrow_x, arrow_y + 0.8),
            xytext=(arrow_x, arrow_y),
            arrowprops=dict(arrowstyle="->"),
        )

# Dark mode version
ax.set_facecolor((0, 0, 0))
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

plt.tight_layout()
fp = "../_static/headline_benchmark_dark.png"
plt.savefig(fp, dpi=300, facecolor=(0, 0, 0, 0))

# Light mode version
ax.set_facecolor((1, 1, 1))
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.spines["bottom"].set_color("black")
ax.spines["left"].set_color("black")
ax.tick_params(axis="x", colors="black")
ax.tick_params(axis="y", colors="black")

fp = "../_static/headline_benchmark.png"
plt.savefig(fp, dpi=300, facecolor=(0, 0, 0, 0))
fp = "../_static/headline_benchmark.pdf"
plt.savefig(fp)
plt.show()

# %%
