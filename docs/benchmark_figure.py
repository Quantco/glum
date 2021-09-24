# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
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
base_cmd = (
    "glm_benchmarks_run --threads 6 --num_rows {n} --storage {s}"
    "--problem_name {p} --library_name {lib}"
)

size = "narrow"
problems = [
    f"{size}-insurance-no-weights-lasso-tweedie-p=1.5",
    f"{size}-insurance-no-weights-lasso-poisson",
    f"{size}-insurance-no-weights-lasso-gaussian",
    f"{size}-insurance-no-weights-lasso-gamma",
    f"{size}-insurance-no-weights-lasso-binomial",
    f"{size}-insurance-no-weights-l2-tweedie-p=1.5",
    f"{size}-insurance-no-weights-l2-poisson",
    f"{size}-insurance-no-weights-l2-gaussian",
    f"{size}-insurance-no-weights-l2-gamma",
    f"{size}-insurance-no-weights-l2-binomial",
]

libraries = ["quantcore-glm", "r-glmnet", "h2o"]

n = 500000

# run r-glmnet and h2o benchmarks, sparse storage works best.
s = "sparse"
for lib in ["r-glmnet", "h2o"]:
    for p in problems:
        cmd = base_cmd.format(n=n, s=s, p=p, lib=lib)
        print(cmd)
        subprocess.run(cmd.split(" "))

# run quantcore-glm benchmarks where auto storage works best.
lib = "quantcore-glm"
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
for reg in ["l2", "lasso"]:
    plot_df = (
        df[df["problem_name"].str.contains(reg)]
        .copy()
        .set_index(["distribution"])[["runtime", "library_name"]]
    )
    plot_df = plot_df.pivot(columns="library_name")
    plot_df.columns = plot_df.columns.get_level_values(1)
    plot_df.index = [x[0:1].upper() + x[1:] for x in plot_df.index]

    reg_title = "Lasso" if reg == "lasso" else "Tikhonov"
    plot_df.plot.bar(
        ylim=[0, 4],
        title=reg_title,
        legend=False,
        figsize=(6, 3),
        width=0.8,
        ylabel="runtime (s)",
        yticks=[0, 1, 2, 3, 4],
    )
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    ax = plt.gca()
    for p in ax.patches:
        x = p.get_x()  # type: ignore
        y = p.get_height()  # type: ignore
        if y > 3.6:
            y = 3.3
            ax.annotate(
                f"{y:.1f}",
                (x + 0.03, y * 1.005),
                fontsize=14,
                rotation="vertical",
            )
    plt.show()

# %%
