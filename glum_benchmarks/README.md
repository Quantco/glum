# glum_benchmarks

Module to benchmark glum against similar libraries.

## Benchmarked Libraries

- [glum](https://github.com/Quantco/glum)
- [scikit-learn](https://scikit-learn.org/)
- [H2O](https://h2o.ai/)
- [skglm](https://contrib.scikit-learn.org/skglm/)
- [celer](https://mathurinm.github.io/celer/)

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

Use `docs_figures` and `readme_figures` in config.yaml to control which figures are included (default: all figures for docs, first non-normalized figure for README).

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

| Option        | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `standardize` | Whether to standardize features (glum/h2o: internal, others: StandardScaler) |
| `iterations`  | Runs per benchmark (>=2 required for skglm)                                  |
| `num_threads` | Number of threads for parallel execution                                     |
| `num_rows`    | Limit rows per dataset (`null` = full dataset)                               |
| `timeout`     | Timeout in seconds (benchmarks timing out are marked "not converged")        |
| `storage`     | Storage format per library: `{"glum": "auto", "sklearn": "dense", ...}`      |

**Notes:**

- **Standardization**: glum and h2o handle scaling internally. sklearn, skglm, and celer use `StandardScaler`. Only numerical columns are scaled (categorical/one-hot columns are not).
- **Convergence**: A benchmark is marked "not converged" if it either (1) hits the timeout, or (2) reaches the library's internal `max_iter` limit.
- **Storage formats**: `"auto"`, `"dense"`, or `"sparse"`. Configured per library for optimal performance.

### Parameter Grid

The `param_grid` section defines which benchmark combinations to run using a sklearn-style parameter grid:

```yaml
param_grid:
  - libraries: ["glum", "sklearn"]
    datasets: ["intermediate-insurance"]
    regularizations: ["lasso", "l2"]
    distributions: ["gaussian", "poisson"]
    reg_strengths: [0.001]
```

Each entry computes a Cartesian product. Multiple entries are unioned (not crossed).

**Available values:**

- `libraries`: `["glum", "sklearn", "h2o", "skglm", "celer", "zeros"]`
- `datasets`: `["intermediate-housing", "intermediate-insurance", "narrow-insurance", "wide-insurance", "square-simulated"]`
- `regularizations`: `["lasso", "l2", "net"]`
- `distributions`: `["gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"]`
- `reg_strengths`: `[0.0001, 0.001, 0.01]`

When an entry is omitted or set to `null` all available values are taken --> if you want to run all combinations possible, just leave the `param_grid` entry empty.

See `problems.py` for available datasets and problem definitions.

## Testing

```bash
pixi run -e benchmark test-benchmarks
```
