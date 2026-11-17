"""Run the code of the tutorial pages end to end.

The tutorials are Markdown pages with the saved output of their code. The
documentation build does not execute them, so this script checks that they still
run. It extracts the ```python blocks of each page and runs them in order in an
IPython shell (the pages use `display` and `%timeit`), from the page's directory
so that the helper modules next to the page can be imported.

Usage:

    pixi run -e docs run-tutorials             # all tutorials
    pixi run -e docs run-tutorials cox_model   # tutorials whose path matches
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PAGES = [DOCS_DIR / "getting_started" / "getting_started.md"] + sorted(
    p for p in (DOCS_DIR / "tutorials").glob("*/*.md")
)
CODE_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)

# Data that has to be downloaded by hand before a tutorial can run.
REQUIRED_DATA = {
    "tutorials/rossman/fixed_effects_rossman.md": (
        ["raw_data/train.csv", "raw_data/test.csv", "raw_data/store.csv"],
        "Download the data of the Kaggle Rossmann Store Sales competition "
        "(https://www.kaggle.com/c/rossmann-store-sales) into "
        "docs/tutorials/rossman/raw_data/.",
    ),
}


def code_blocks(page: Path) -> list[str]:
    return CODE_BLOCK.findall(page.read_text())


def run_page(page: Path) -> None:
    # Runs in a subprocess whose working directory is the page's directory.
    from IPython.core.interactiveshell import InteractiveShell

    sys.path.insert(0, os.getcwd())
    shell = InteractiveShell.instance(colors="nocolor")
    blocks = code_blocks(page)
    for i, block in enumerate(blocks, start=1):
        print(f"--- block {i}/{len(blocks)}", flush=True)
        result = shell.run_cell(block, silent=False)
        if not result.success:
            print(f"\nFAILED in code block {i}:\n{block}", file=sys.stderr)
            sys.exit(1)


def missing_data(page: Path) -> str | None:
    key = page.relative_to(DOCS_DIR).as_posix()
    if key not in REQUIRED_DATA:
        return None
    files, hint = REQUIRED_DATA[key]
    if all((page.parent / f).exists() for f in files):
        return None
    return hint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "patterns", nargs="*", help="only run pages whose path contains one of these"
    )
    parser.add_argument("--page", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--verbose", action="store_true", help="show the output of the tutorials"
    )
    args = parser.parse_args()

    if args.page:
        run_page(args.page)
        return 0

    pages = [
        p
        for p in PAGES
        if not args.patterns or any(s in p.as_posix() for s in args.patterns)
    ]
    if not pages:
        print("No tutorial matches the given patterns.")
        return 1

    env = {**os.environ, "MPLBACKEND": "Agg"}
    results = {}
    for page in pages:
        name = page.relative_to(DOCS_DIR).as_posix()
        if not code_blocks(page):
            results[name] = "no code"
            continue
        hint = missing_data(page)
        if hint:
            results[name] = "skipped"
            print(f"Skipping {name}: {hint}")
            continue

        print(f"Running {name} ...", flush=True)
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, "--page", str(page)],
            cwd=page.parent,
            env=env,
            capture_output=not args.verbose,
            text=True,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode == 0:
            results[name] = f"passed ({elapsed:.0f}s)"
        else:
            results[name] = f"FAILED ({elapsed:.0f}s)"
            if not args.verbose:
                # The last lines of the output contain the traceback.
                output = (proc.stdout or "") + (proc.stderr or "")
                print("\n".join(output.splitlines()[-40:]))

    print("\nSummary:")
    width = max(len(n) for n in results)
    for name, status in results.items():
        print(f"  {name:<{width}}  {status}")
    return int(any(s.startswith("FAILED") for s in results.values()))


if __name__ == "__main__":
    sys.exit(main())
