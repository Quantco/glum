#!/bin/bash

set -e
set -x

if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
    CONDA_CHANNEL="conda-forge/osx-arm64"
else
    CONDA_CHANNEL="conda-forge/osx-64"
fi

conda create -n build -c $CONDA_CHANNEL 'llvm-openmp=11'

