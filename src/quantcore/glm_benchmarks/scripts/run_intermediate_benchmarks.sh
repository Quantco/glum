#!/bin/bash

set -uo pipefail

OUTPUT_DIR=$(git rev-parse HEAD)
PROBLEM_NAMES="intermediate-insurance-weights-net-poisson,intermediate-insurance-weights-l2-poisson,intermediate-insurance-weights-net-gamma,intermediate-insurance-weights-l2-gamma"
LIBRARY_NAMES="sklearn-fork,orig-sklearn-fork,glmnet-python,h2o"
THREADS=8

export GLM_BENCHMARKS_CACHE_SIZE_LIMIT=20737418240  # 20 GB
export GLM_BENCHMARKS_CACHE=.cache

for NUM_ROWS in 100000 1000000 10000000
do
    for REG_STRENGTH in 0.1 0.001 0.00001
    do
        for STORAGE in "dense" "sparse" "split0.1"
        do
            echo "---------------------------------"
            echo "NUM_ROWS = ${NUM_ROWS}"
            echo "REG_STRENGTH = ${REG_STRENGTH}"
            echo "STORAGE = ${STORAGE}"
            glm_benchmarks_run \
                --problem_name "${PROBLEM_NAMES}" \
                --library_name "${LIBRARY_NAMES}" \
                --num_rows ${NUM_ROWS} \
                --threads ${THREADS} \
                --storage ${STORAGE} \
                --regularization_strength ${REG_STRENGTH} \
                --output_dir ${OUTPUT_DIR}

        done
    done
done

glm_benchmarks_analyze \
    --problem_name "${PROBLEM_NAMES}" \
    --library_name "${LIBRARY_NAMES}" \
    --output_dir ${OUTPUT_DIR} \
    --export "intermediate_results.csv"
