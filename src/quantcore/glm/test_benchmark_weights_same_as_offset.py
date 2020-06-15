import json

import numpy as np
from git_root import git_root


def main():
    with open(git_root("golden_master/benchmark_gm.json"), "r") as f:
        results = json.load(f)
    for k in results.keys():
        if "weights" in k:
            same_with_offset = "offset".join(k.split("weights"))
            np.testing.assert_allclose(
                results[k]["coef"], results[same_with_offset]["coef"]
            )


if __name__ == "__main__":
    main()
