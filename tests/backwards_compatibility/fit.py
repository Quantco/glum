"""Fit a GLM with the installed glum version and save artifacts.

Usage:
  python fit.py X.Y.Z   # label artifacts with a release version
  python fit.py HEAD     # label artifacts with HEAD (current repo version)

Artifacts are written to:
  tests/backwards_compatibility/artifacts/<version>/model.pkl
  tests/backwards_compatibility/artifacts/<version>/predictions.npy

NOTE: This script must work with glum >= 2.0.0. It deliberately avoids
features added in 3.x: formula interface, Polars DataFrames, monotonic
constraints, closed-form solver.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from glum import GeneralizedLinearRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
        help="Artifact label: a release version string (e.g. 2.0.3) or HEAD",
    )
    args = parser.parse_args()

    import glum

    installed_version = glum.__version__

    print(f"Installed glum version: {installed_version}")
    print(f"Artifact label: {args.version}")

    output_dir = ARTIFACTS_DIR / args.version
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(str(ARTIFACTS_DIR / "X.npy"))
    y = np.load(str(ARTIFACTS_DIR / "y.npy"))

    # All keyword args: glum 3.0 made all params keyword-only; kwargs work in 2.x too.
    # alpha=1.0: explicit to avoid 2.x (default=1) vs 3.x (default=0) difference.
    # solver="irls-cd": avoids the closed-form solver added in 3.2 which may produce
    #   slightly different floating-point results against the iterative solver.
    model = GeneralizedLinearRegressor(
        family="normal",
        alpha=1.0,
        solver="irls-cd",
    )
    model.fit(X, y)

    pickle_path = output_dir / "model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {pickle_path}")

    predictions = model.predict(X)
    predictions_path = output_dir / "predictions.npy"
    np.save(str(predictions_path), predictions)
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
