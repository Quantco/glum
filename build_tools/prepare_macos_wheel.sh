#!/bin/bash

set -e
set -x

if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
    OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
else
    OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
fi

conda create -y -n build $OPENMP_URL
