#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/macos-conda-build.sh $*

mamba install -y conda-channel-client

if [[ "${GITHUB_REF}" == refs/tags/* ]]; then
  upload-conda-package $(conda render -m .ci_support/${CONDA_BUILD_YML}.yaml --output conda.recipe)
fi
