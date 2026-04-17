from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ruamel.yaml import YAML

Library = Literal[
    "glum", "sklearn", "h2o", "skglm", "celer", "zeros", "glmnet", "pygam"
]
Dataset = Literal[
    "intermediate-insurance",
    "intermediate-housing",
    "narrow-insurance",
    "wide-insurance",
    "simulated-glm",
    "categorical-simulated",
]
Regularization = Literal["lasso", "l2", "net", "monotonic-l2"]
Alpha = float  # Valid values: 0.0001, 0.001, 0.01
ALPHA_VALUES = (0.001, 0.01, 0.1)
Distribution = Literal["gaussian", "gamma", "binomial", "poisson", "tweedie-p=1.5"]
StorageFormat = Literal["auto", "dense", "cat", "csr", "csc"]

ALLOWED_DISTRIBUTIONS_BY_DATASET: dict[Dataset, set[Distribution]] = {
    "intermediate-housing": {"gaussian", "gamma"},
    "intermediate-insurance": {"gamma", "poisson", "tweedie-p=1.5"},
    "narrow-insurance": {"gamma", "poisson", "tweedie-p=1.5"},
    "wide-insurance": {"gamma", "poisson", "tweedie-p=1.5"},
    "simulated-glm": {"binomial", "gaussian", "gamma", "poisson"},
    "categorical-simulated": {"binomial", "gamma", "poisson"},
}


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
    num_rows: list[int | None] | None = Field(
        default=None,
        description=(
            "Default is None; None means full dataset. "
            "Inside the list, null means full dataset."
        ),
    )
    k_over_n_ratios: list[float] | None = Field(
        default=None,
        description=(
            "Feature-to-row ratios (K/N) for simulated-glm. Ignored for other datasets."
        ),
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
    max_rel_slowdown: float = Field(
        default=0.15,
        description="Relative slowdown threshold for regression detection.",
    )
    max_abs_slowdown_sec: float = Field(
        default=0.05,
        description="Absolute slowdown threshold (seconds) for regression detection.",
    )
    max_regressed_cases: int = Field(
        default=0, description="Max allowed regressed cases before CI fails."
    )
    docs_figures: list[list[str]] | None = Field(
        default=None,
        description=(
            "Figure groups for docs/benchmarks.rst. Each inner list maps to one "
            "BENCHMARK_FIGURES_START/END block (in order). None = all figures in "
            "a single block."
        ),
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
    param_grid: list[ParamGridEntry] = Field(
        default_factory=lambda: [ParamGridEntry()],
        description="List of parameter sets. Each entry defines a Cartesian product, "
        "entries are unioned. Default runs all combinations.",
    )

    # Benchmark settings
    num_threads: int = Field(
        default=16, ge=1, description="Number of threads for parallel execution"
    )
    iterations: int = Field(
        default=3,
        ge=1,
        description=(
            "Run each benchmark N times. When >= 2, the first iteration is "
            "discarded as warmup and the median of the rest is reported."
        ),
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
        return self.script_dir / "results" / self.run_name

    @property
    def pickle_dir(self) -> Path:
        return self.results_dir / "pickles"

    @property
    def figure_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def csv_file(self) -> Path:
        return self.results_dir / "results.csv"

    @property
    def docs_static_dir(self) -> Path:
        return self.script_dir.parent / "docs" / "_static"

    @property
    def index_rst(self) -> Path:
        return self.script_dir.parent / "docs" / "index.rst"

    @property
    def benchmarks_rst(self) -> Path:
        return self.script_dir.parent / "docs" / "benchmarks.rst"

    @property
    def readme_file(self) -> Path:
        return self.script_dir.parent / "README.md"

    @model_validator(mode="after")
    def validate_config(self) -> BenchmarkConfig:
        """Validate cross-field constraints."""
        if not self.run_name or not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        yaml = YAML(typ="safe", pure=True)
        with open(yaml_path) as f:
            data = yaml.load(f)

        data["script_dir"] = yaml_path.parent

        return cls.model_validate(data)
