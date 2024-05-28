import numpy as np
import pytest

from glum_benchmarks.cli_run import get_all_problems
from glum_benchmarks.problems import Problem
from glum_benchmarks.util import (
    BenchmarkParams,
    exposure_and_offset_to_weights,
    get_obj_val,
    get_tweedie_p,
)

all_test_problems_offset = {
    k: v
    for k, v in get_all_problems().items()
    if "offset" in k and "gaussian" not in k and "binomial" not in k
}
bench_cfg = dict(
    num_rows=10000,
    regularization_strength=0.1,
    storage="dense",
)


@pytest.mark.parametrize(
    ["Pn", "P"],
    [
        x if "wide" not in x[0] else pytest.param(x[0], x[1], marks=pytest.mark.slow)
        for x in all_test_problems_offset.items()
    ],  # mark the "wide" problems as "slow" so that we can call pytest -m "not slow"
    ids=all_test_problems_offset.keys(),
)
def test_offset_solution_matches_weights_solution(
    Pn: str,
    P: Problem,
):
    params = BenchmarkParams(
        problem_name=Pn,
        library_name="sklearn-fork",
        # storage=storage,
        **bench_cfg,
    )

    tweedie_p = get_tweedie_p(P.distribution)

    dat = P.data_loader(num_rows=params.num_rows)
    weights_dat = {"X": dat["X"]}
    weights_dat["y"], weights_dat["weights"] = exposure_and_offset_to_weights(
        tweedie_p, dat["y"], offset=dat["offset"]
    )

    np.random.seed(0)
    coefs = np.zeros(dat["X"].shape[1] + 1)
    coefs[1:] = np.random.normal(0, 0.01 / dat["X"].std(0))
    eta = dat["X"].to_numpy(dtype=float).dot(coefs[1:]) + dat["offset"]
    coefs[0] = np.log(dat["y"].mean() / np.exp(eta).mean())

    eps = 1e-7
    coefs_2 = coefs + eps

    def get_obj_val_(dat, coefs):
        if "weights" in dat.keys():
            reg_multiplier = dat["weights"].mean()
        else:
            reg_multiplier = 1

        res = get_obj_val(
            dat,
            P.distribution,
            P.regularization_strength / reg_multiplier,
            P.l1_ratio,
            coefs[0],
            coefs[1:],
        )
        if "weights" in dat.keys():
            res *= reg_multiplier
        return res

    offset_result = get_obj_val_(dat, coefs)
    offset_result_2 = get_obj_val_(dat, coefs_2)
    weights_result = get_obj_val_(weights_dat, coefs)
    weights_result_2 = get_obj_val_(weights_dat, coefs_2)

    np.testing.assert_allclose(
        offset_result_2 - offset_result,
        weights_result_2 - weights_result,
        rtol=1e-7,
        atol=1e-14,
    )
