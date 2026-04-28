"""Fit a GLM with the installed glum version and save artifacts.

Usage:
  python fit.py X.Y.Z   # label artifacts with a release version
  python fit.py HEAD     # label artifacts with HEAD (current repo version)

Artifacts are written to:
  tests/backwards_compatibility/artifacts/<version>/model.pkl
  tests/backwards_compatibility/artifacts/<version>/predictions.csv

NOTE: This script must work with glum >= 2.0.0. It deliberately avoids
features added in 3.x: formula interface, Polars DataFrames, monotonic
constraints, closed-form solver.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import make_regression

from glum import GeneralizedLinearRegressor


def get_data():
    X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=42)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
        help="Artifact label: a release version string (e.g. 2.0.3) or HEAD",
    )
    args = parser.parse_args()

    try:
        import importlib.metadata

        installed_version = importlib.metadata.version("glum")
    except Exception:
        installed_version = "unknown"

    print(f"Installed glum version: {installed_version}")
    print(f"Artifact label: {args.version}")

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "artifacts" / args.version
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = get_data()

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
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model to {pickle_path}")

    predictions = model.predict(X)
    predictions_path = output_dir / "predictions.csv"
    np.savetxt(str(predictions_path), predictions, delimiter=",")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
