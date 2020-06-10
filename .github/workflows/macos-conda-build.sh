#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/macos-base.sh $*

conda activate base
mamba install -y conda-build
conda build --python ${PYTHON_VERSION} conda.recipe
