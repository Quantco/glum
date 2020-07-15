#!/bin/bash

set -exo pipefail

mamba install -y conda-build conda-channel-client

conda build --python ${PYTHON_VERSION} --variants "{GLM_ARCHITECTURE: ['${GLM_ARCHITECTURE}']}" conda.recipe
upload-conda-package $(conda render --python ${PYTHON_VERSION} --variants "{GLM_ARCHITECTURE: ['${GLM_ARCHITECTURE}']}" --output conda.recipe)
