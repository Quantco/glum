#!/bin/bash

set -exo pipefail

mamba install -y conda-build
conda build --python ${PYTHON_VERSION} conda.recipe
