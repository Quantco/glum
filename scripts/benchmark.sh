#!/bin/bash

OUTPUT_DIR=$(git rev-parse HEAD)
PROBLEM_NAMES="intermediate_insurance_offset_net_poisson, intermediate_insurance_offset_l2_poisson"
LIBRARY_NAMES="sklearn_fork, glmnet_python"
THREADS=8

for NUM_ROWS in 100000 1000000 2000000
do
    for REG_STRENGTH in 1.0 0.1 0.001 0.00001 0.0
    do
        for STORAGE in "dense" "sparse" "split0.1"
        do
            glm_benchmarks_run \
                --problem_names "${PROBLEM_NAMES}" \
                --library_names "${LIBRARY_NAMES}" \
                --num_rows ${NUM_ROWS} \
                --threads ${THREADS} \
                --storage ${STORAGE} \
                --regularization_strength ${REG_STRENGTH} \
                --output_dir ${OUTPUT_DIR}
        done
    done
done

# analyze
glm_benchmarks_analyze \
    --problem_names "${PROBLEM_NAMES}" \
    --library_names "${LIBRARY_NAMES}" \
    --output_dir ${OUTPUT_DIR}
