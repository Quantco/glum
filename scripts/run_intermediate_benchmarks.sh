#!/bin/bash

OUTPUT_DIR=$(git rev-parse HEAD)
PROBLEM_NAMES="intermediate-insurance-offset-net-poisson,intermediate-insurance-offset-l2-poisson"
LIBRARY_NAMES="sklearn-fork,glmnet-python"
THREADS=8

export GLM_BENCHMARKS_CACHE_SIZE_LIMIT=10737418240  # 10 GB
export GLM_BENCHMARKS_CACHE=.cache

for NUM_ROWS in 100000 1000000 10000000
do
    for REG_STRENGTH in 1.0 0.1 0.01 0.0
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
    --output_dir ${OUTPUT_DIR}
