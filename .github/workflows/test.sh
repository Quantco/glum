#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $*

mamba install -y yq
yq -Y ". + {dependencies: [.dependencies[], \"python=${PYTHON_VERSION}\"] }" environment.yml > /tmp/environment.yml
mamba env create -f /tmp/environment.yml
mamba env update -n $(yq -r .name environment.yml) --file environment-benchmark.yml
conda activate $(yq -r .name environment.yml)
pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
pytest -nauto tests --doctest-modules src/

# Check that the readme example will work by running via doctest.
# We run outside the repo to make the test a bit more similar to
# a user running after installing with conda.
mkdir ../temp
cp README.md ../temp
cd ../temp
python -m doctest -v README.md
