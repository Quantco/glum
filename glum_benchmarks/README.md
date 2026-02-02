# glum_benchmarks

Python module to benchmark GLM implementations.

## Benchmarked Libraries

- [glum](https://github.com/Quantco/glum) - High-performance GLMs with L1/L2 regularization
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python
- [H2O](https://h2o.ai/) - Distributed machine learning platform
- [skglm](https://contrib.scikit-learn.org/skglm/) - Fast sklearn-compatible GLM solvers
- [celer](https://mathurinm.github.io/celer/) - Fast Lasso solver with dual extrapolation

## Running the benchmarks

```bash
pixi run -e benchmark run-benchmarks
```

The benchmark script runs in three steps that can be controlled independently.

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

### Workflow examples

**Full run (default):**

```yaml
run_benchmarks: true
analyze_results: true
generate_plots: true
```

**Regenerate plots without re-running benchmarks:**

```yaml
run_benchmarks: false
analyze_results: true # Rebuild CSV from existing pickles
generate_plots: true
```

**Run benchmarks and analyze results without creating figures:**

```yaml
run_benchmarks: true
analyze_results: true
generate_plots: false
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

| Option            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `run_benchmarks`  | Run Step 1 (execute benchmarks)                   |
| `analyze_results` | Run Step 2 (analyze pickles, write CSV)           |
| `generate_plots`  | Run Step 3 (generate figures from CSV)            |
| `update_docs`     | Copy figures to docs and update documentation     |
| `run_name`        | Subfolder in `results/` (`"docs"` is git-tracked) |
| `clear_output`    | Clear entire `run_name` directory before running  |

### Benchmark Settings

| Option        | Description                                    |
| ------------- | ---------------------------------------------- |
| `standardize` | Whether to standardize features before fitting |
| `iterations`  | Runs per benchmark (>=2 required for skglm)    |
| `num_threads` | Number of threads for parallel execution       |
| `num_rows`    | Limit rows per dataset (`null` = full dataset) |
| `timeout`     | Timeout in seconds per benchmark run           |

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

When an entry is omitted or set to `null` all available values are taken --> if you want to run all combinations possiblem, just leave the `param_grid` entry empty.

See `problems.py` for available datasets and problem definitions.

## Testing

```bash
pixi run -e benchmark test-benchmarks
```
