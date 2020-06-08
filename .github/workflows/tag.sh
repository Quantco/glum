#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $*

export GLM_ARCHITECTURE=$2

conda activate base
conda build --python ${PYTHON_VERSION} --variants "{GLM_ARCHITECTURE: ['${GLM_ARCHITECTURE}']}" conda.recipe
upload-conda-package $(conda render --python ${PYTHON_VERSION} --variants "{GLM_ARCHITECTURE: ['${GLM_ARCHITECTURE}']}" --output conda.recipe)
