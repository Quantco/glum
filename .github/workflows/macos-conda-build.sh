#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $*
conda activate base

mamba install -y conda-build -c conda-forge
# Don't test cross-compiled result (there is no emulation) and use the latest MacOS SDK.
if grep -q "osx-arm64" .ci_support/${CONDA_BUILD_YML}.yaml; then
  CONDA_BUILD_ARGS="--no-test"
  export CONDA_BUILD_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)
  cat <<EOF >> .ci_support/${CONDA_BUILD_YML}.yaml
CONDA_BUILD_SYSROOT:
 - "${CONDA_BUILD_SYSROOT}"
EOF
fi
conda build -m .ci_support/${CONDA_BUILD_YML}.yaml conda.recipe ${CONDA_BUILD_ARGS:-}
