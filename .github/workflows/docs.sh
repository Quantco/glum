#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $*

mamba install -y yq
yq -Y '. + {dependencies: [.dependencies[], "python=3.8"] }' environment.yml > /tmp/environment.yml
mamba env create -f /tmp/environment.yml
conda activate $(yq -r .name environment.yml)
pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .

pushd $(pwd)/docs
make html
popd
