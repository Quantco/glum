# glum_benchmarks

Python module to benchmark GLM implementations.

## Running the benchmarks

```bash
pixi run -e benchmark run-benchmarks
```

The benchmark script runs in three steps that can be controlled independently.

### Step 1: Run Benchmarks (`RUN_BENCHMARKS`)

Executes each problem/library combination and saves raw results.

- **Input:** Configuration parameters
- **Output:** Pickle files in `results/<RUN_NAME>/pickles/`
- **Set `RUN_BENCHMARKS = False`** to skip (e.g. to generate cvs from existing results)

### Step 2: Analyze Results (`ANALYZE_RESULTS`)

Reads pickle files, prints a summary table, and exports CSV for plotting.

- **Input:** Pickle files from `results/<RUN_NAME>/pickles/`
- **Output:** Summary printed to console + `results/<RUN_NAME>/results.csv`
- **Set `ANALYZE_RESULTS = False`** to skip (e.g. to generate plots from existing csv)

### Step 3: Generate Plots (`GENERATE_PLOTS`)

Creates comparison charts from the CSV produced by Step 2.

- **Input:** `results/<RUN_NAME>/results.csv`
- **Output:** PNG files in `results/<RUN_NAME>/figures/`
- **Set `GENERATE_PLOTS = False`** to skip

### Workflow examples

**Full run (default):**

```python
RUN_BENCHMARKS = True
ANALYZE_RESULTS = True
GENERATE_PLOTS = True
```

**Regenerate plots without re-running benchmarks:**

```python
RUN_BENCHMARKS = False
ANALYZE_RESULTS = True   # Rebuild CSV from existing pickles
GENERATE_PLOTS = True
```

**Run benchmarks and analyze results without creating figures:**

```python
RUN_BENCHMARKS = True
ANALYZE_RESULTS = True
GENERATE_PLOTS = False
```

## Output structure

Results are organized by `RUN_NAME` (default: `"docs"`):

```
glum_benchmarks/
└── results/
    └── docs/              # RUN_NAME = "docs"
        ├── pickles/       # Step 1 output (gitignored)
        ├── figures/       # Step 3 output (gitignored)
        └── results.csv    # Step 2 output (tracked for docs/)
```

Only `results/docs/results.csv` is tracked in git. Change `RUN_NAME` for experiments.

## Configuration

Edit the CONFIGURATION section at the top of `run_benchmarks.py`.
The script is pre-configured for the documentation benchmark run.

| Option             | Description                                    |
| ------------------ | ---------------------------------------------- |
| `RUN_BENCHMARKS`   | Run Step 1 (execute benchmarks)                |
| `ANALYZE_RESULTS`  | Run Step 2 (analyze pickles, write CSV)        |
| `GENERATE_PLOTS`   | Run Step 3 (generate figures from CSV)         |
| `RUN_NAME`         | Subfolder in `results/`                        |
| `LIBRARIES`        | Which libraries to benchmark                   |
| `DATASETS`         | Which datasets to run                          |
| `REGULARIZATIONS`  | Regularization types   |
| `DISTRIBUTIONS`    | Distribution families                          |
| `NUM_THREADS`      | Number of threads for parallel execution       |
| `REG_STRENGTH`     | Regularization strength                        |
| `STANDARDIZE`      | Whether to standardize features before fitting |
| `ITERATIONS`       | Runs per benchmark (>=2 required for skglm)    |
| `NUM_ROWS`         | Limit rows per dataset (`None` = full dataset) |
| `CLEAR_OUTPUT`     | Clear entire `RUN_NAME` directory before running |

See `problems.py` for available datasets and problem definitions.

## Testing

```bash
pixi run -e benchmark test-benchmarks
```
