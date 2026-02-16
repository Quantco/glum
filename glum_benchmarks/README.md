# glum_benchmarks

Module to benchmark glum against similar libraries.

## Benchmarked Libraries

- [glum](https://github.com/Quantco/glum)
- [scikit-learn](https://scikit-learn.org/)
- [H2O](https://h2o.ai/)
- [skglm](https://contrib.scikit-learn.org/skglm/)
- [celer](https://mathurinm.github.io/celer/)
- [glmnet (R)](https://cran.r-project.org/package=glmnet)

## Running the benchmarks

```bash
pixi run -e benchmark run-benchmarks
```

The benchmark script runs in four steps that can be controlled independently.

### Step 1: Run Benchmarks (`run_benchmarks`)

Executes each problem/library combination and saves raw results.

- **Input:** Configuration from `config.yaml`
- **Output:** Pickle files in `results/<run_name>/pickles/`
- **Set `run_benchmarks: false`** in config.yaml to skip

### Step 2: Analyze Results (`analyze_results`)

Reads pickle files, prints a summary table, and exports CSV for plotting.

- **Input:** Pickle files from `results/<run_name>/pickles/`
- **Output:** Summary printed to console + `results/<run_name>/results.csv`
- **Set `analyze_results: false`** in config.yaml to skip

### Step 3: Generate Plots (`generate_plots`)

Creates comparison charts from the CSV produced by Step 2.

- **Input:** `results/<run_name>/results.csv`
- **Output:** PNG files in `results/<run_name>/figures/`
- **Set `generate_plots: false`** in config.yaml to skip

### Step 4: Update documentation and README (`update_docs`)

Copies figures to `docs/_static/` and updates figure references in documentation files.

- **Input:** PNG files from `results/<run_name>/figures/`
- **Output:** Copies to `docs/_static/` + updates `docs/benchmarks.rst` and `README.md`
- **Set `update_docs: false`** in config.yaml to skip

Use `docs_figures` and `readme_figures` in config.yaml to control which figures are included (default: all figures for docs, first non-normalized figure for README). Figure names are usually `<dataset>-<distribution>.png`. For `simulated-glm` they include the ratio suffix, e.g. `simulated-glm-gaussian-k-over-n-0.7.png`.

### Workflow examples

**Full run (default):**

```yaml
run_benchmarks: true
analyze_results: true
generate_plots: true
update_docs: false # Set to true to update docs/_static and documentation
```

**Regenerate plots without re-running benchmarks:**

```yaml
run_benchmarks: false
analyze_results: true # Rebuild CSV from existing pickles
generate_plots: true
update_docs: false
```

**Update documentation with existing figures:**

```yaml
run_benchmarks: false
analyze_results: false
generate_plots: false
update_docs: true # Only copy figures and update docs
```

## Output structure

Results are organized by `run_name` (default: `"docs"`):

```
glum_benchmarks/
└── results/
    └── docs/              # run_name = "docs"
        ├── config.yaml    # Snapshot of config used for this run (tracked)
        ├── pickles/       # Step 1 output (gitignored)
        ├── figures/       # Step 3 output (gitignored)
        └── results.csv    # Step 2 output (tracked)
```

The configuration is automatically saved to `results/<run_name>/config.yaml` for full reproducibility. You can re-run old benchmarks by copying their config back to the main directory.

For the `docs` run, both `results.csv` and `config.yaml` are tracked in git. Change `run_name` for experiments.

## Configuration

Edit `config.yaml` to customize benchmark parameters.

### General Options

| Option            | Description                                        |
| ----------------- | -------------------------------------------------- |
| `run_benchmarks`  | Run Step 1 (execute benchmarks)                    |
| `analyze_results` | Run Step 2 (analyze pickles, write CSV)            |
| `generate_plots`  | Run Step 3 (generate figures from CSV)             |
| `update_docs`     | Run Step 4 (copy figures to docs and update files) |
| `docs_figures`    | List of figures for docs (null = all)              |
| `readme_figures`  | List of figures for README (null = first figure)   |
| `run_name`        | Subfolder in `results/` (`"docs"` is git-tracked)  |
| `clear_output`    | Clear entire `run_name` directory before running   |

### Benchmark Settings

| Option        | Description                                                           |
| ------------- | --------------------------------------------------------------------- |
| `standardize` | Standardization strategy per library (`pre`, `internal`, `none`)      |
| `iterations`  | Runs per benchmark (>=2 required for skglm)                           |
| `num_threads` | Number of threads for parallel execution                              |
| `timeout`     | Timeout in seconds (benchmarks timing out are marked "not converged") |
| `storage`     | Storage format per library: (`auto`, `dense`, `cat`, `csr`, `csc`)    |

**Notes:**

- **Standardization**: `pre` standardizes continuous columns in the data loader before OHE/format conversion; `internal` delegates to the library; `none` skips scaling.
- **glmnet dependency**: R + `glmnet` are required for the `glmnet` benchmark (via `rpy2`). If missing, the benchmark is skipped.
- **Convergence**: A benchmark is marked "not converged" if it either (1) hits the timeout, or (2) reaches the library's internal `max_iter` limit.

### Parameter Grid

The `param_grid` section defines which benchmark combinations to run using a sklearn-style parameter grid:

```yaml
param_grid:
  - libraries: ["glum", "sklearn"]
    datasets: ["intermediate-insurance"]
    regularizations: ["lasso", "l2"]
    distributions: ["gaussian", "poisson"]
    alphas: [0.001]
  - datasets: ["simulated-glm"]
    distributions: ["gaussian", "poisson"]
    num_rows: [1000, 5000]
    k_over_n_ratios: [0.5, 0.7, 1.2]
```

Each entry computes a Cartesian product. Multiple entries are unioned (not crossed).

**Available values:**

- `libraries`: `["glum", "sklearn", "h2o", "skglm", "celer", "zeros", "glmnet"]`
- `datasets`: `["intermediate-housing", "intermediate-insurance", "narrow-insurance", "wide-insurance", "simulated-glm", "categorical-simulated"]`
- `regularizations`: `["lasso", "l2", "net"]`
- `distributions`: `["gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"]`
- `alphas`: `[0.0001, 0.001, 0.01]`
- `num_rows`: list of row limits, where `null` means full dataset (e.g., `[1000, null]`)
- `k_over_n_ratios`: any positive float values (e.g., `[0.5, 0.7, 1.2]`, applies to `simulated-glm` only)

When an entry is omitted or set to `null`, all available values are used. For `libraries`, the default excludes `zeros` (include it explicitly if you want it). For `k_over_n_ratios`, the default is `[1.0]` when omitted. For `num_rows`, the default is `[null]` (full dataset). If you want to run all default combinations, leave the `param_grid` entry empty.

**Alpha note:** `alphas` are per-observation values for unweighted data, when weights are present, the benchmark runner adjusts internally to keep the penalty comparable.

See `problems.py` for available datasets and problem definitions.

## Testing

```bash
pixi run -e benchmark test-benchmarks
```
