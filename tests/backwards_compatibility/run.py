"""Backwards compatibility tests for glum.

Usage: python tests/backwards_compatibility/run_all.py
       (or via: pixi run test-backwards-compatibility)

1. Fits the current (HEAD) glum to produce reference predictions.
2. Queries conda-forge via `pixi search` to discover the latest patch release
   for each minor version of glum.
3. For each version, uses `pixi exec` to fit a model and save artifacts
   (model.pkl + predictions.csv) under artifacts/<version>/.
4. Unpickles each saved model using the current glum and verifies that
   predictions match the HEAD reference.
"""

import importlib.metadata
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from packaging.version import Version
from sklearn.datasets import make_regression

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

SKIP_VERSIONS: set[str] = set()


def get_data():
    """Return the fixed dataset used for fitting and prediction."""
    X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=42)
    return X, y


def discover_versions() -> list[str]:
    """Return the latest patch release for each minor version of glum on conda-forge."""
    result = subprocess.run(
        ["pixi", "search", "glum", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    platform = next(iter(data))
    best: dict[tuple[int, int], str] = {}
    for entry in data[platform]:
        v = Version(entry["version"])
        key = (v.major, v.minor)
        if key not in best or v > Version(best[key]):
            best[key] = entry["version"]
    return sorted(best.values(), key=Version)


def fit_version(version: str) -> bool:
    """Run fit.py for the given version and return True on success.

    Uses ``pixi run`` for HEAD and ``pixi exec`` for released versions.
    """
    if version == "HEAD":
        cmd = ["pixi", "run", "python", str(SCRIPT_DIR / "fit.py"), "HEAD"]
    else:
        v = Version(version)
        cmd = ["pixi", "exec", f"--spec=glum=={version}"]
        # glum <=2.6.0 imports pkg_resources from setuptools, which was removed
        # in setuptools 82. Pin setuptools<82 for those old versions.
        if v <= Version("2.6.0"):
            cmd += ["--spec=setuptools<82"]
        # glum <=2.3.0: sklearn 1.3 added const qualifiers to _cython_blas
        # function pointers, breaking the Cython ABI of older glum builds.
        if v <= Version("2.3.0"):
            cmd += ["--spec=scikit-learn<1.3"]
        # glum 3.0.x: sklearn 1.6 removed BaseEstimator._validate_data.
        elif v < Version("3.1.0"):
            cmd += ["--spec=scikit-learn<1.6"]
        cmd += ["python", str(SCRIPT_DIR / "fit.py"), version]
    return subprocess.run(cmd).returncode == 0


def compare_versions(versions: list[str]) -> bool:
    """Unpickle each version's model and verify its predictions match HEAD.

    Also checks that predictions match the CSV stored by fit.py to confirm
    the pickle round-trip is stable. Returns True if all versions pass.
    """
    version_dirs = [ARTIFACTS_DIR / v for v in versions if (ARTIFACTS_DIR / v).is_dir()]

    if not version_dirs:
        print("ERROR: No artifact directories found. Did fit step produce any output?")
        return False

    X, _ = get_data()
    head_predictions = np.loadtxt(
        str(ARTIFACTS_DIR / "HEAD" / "predictions.csv"), delimiter=","
    )

    try:
        current_version = importlib.metadata.version("glum")
    except Exception:
        current_version = "unknown"
    print(f"Current glum version: {current_version}")
    print(f"Testing {len(version_dirs)} version(s): {[d.name for d in version_dirs]}\n")

    failures = []

    for version_dir in version_dirs:
        version = version_dir.name
        pickle_path = version_dir / "model.pkl"
        predictions_path = version_dir / "predictions.csv"

        try:
            with open(pickle_path, "rb") as f:
                old_model = pickle.load(f)
        except Exception as e:
            failures.append(f"{version}: unpickling failed: {e}")
            continue

        try:
            old_predictions = old_model.predict(X)
        except Exception as e:
            failures.append(f"{version}: predict() failed after unpickling: {e}")
            continue

        stored_predictions = np.loadtxt(str(predictions_path), delimiter=",")

        try:
            np.testing.assert_allclose(
                old_predictions,
                stored_predictions,
                rtol=1e-5,
                err_msg=f"[{version}] Unpickled predictions do not match stored CSV",
            )
            print(f"[{version}] PASS: unpickled predictions match stored predictions")
        except AssertionError as e:
            failures.append(str(e))

        try:
            np.testing.assert_allclose(
                old_predictions,
                head_predictions,
                rtol=1e-5,
                err_msg=f"[{version}] Predictions from old model do not match HEAD",
            )
            print(f"[{version}] PASS: old model predictions match HEAD")
        except AssertionError as e:
            failures.append(str(e))

    print()
    if failures:
        print("FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return False

    print(f"All {len(version_dirs)} version(s) passed.")
    return True


def main() -> None:
    """Fit HEAD and all released minor versions, then compare predictions."""
    print("=== Fitting HEAD ===")
    if not fit_version("HEAD"):
        print("ERROR: Failed to fit HEAD model.")
        sys.exit(1)

    print("\n=== Discovering glum versions from conda-forge ===")
    versions = discover_versions()
    print(f"Found {len(versions)} minor release(s): {' '.join(versions)}")

    print("\n=== Generating compatibility artifacts ===")
    fitted_versions = []
    for version in versions:
        if version in SKIP_VERSIONS:
            print(f"--- Skipping glum=={version} (known incompatibility) ---")
            continue
        print(f"--- Fitting glum=={version} ---")
        if fit_version(version):
            fitted_versions.append(version)
        else:
            print(f"WARNING: glum=={version} failed, skipping.")

    print("\n=== Comparing against HEAD ===")
    if not compare_versions(fitted_versions):
        sys.exit(1)


if __name__ == "__main__":
    main()
