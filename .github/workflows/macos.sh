#!/bin/bash

set -exo pipefail
source ~/.profile

mamba install -y yq jq

yq -Y ". + {dependencies: [.dependencies[], \"python=${PYTHON_VERSION}\"] }" environment.yml > /tmp/environment.yml
mamba env create -f /tmp/environment.yml
mamba env update -n $(yq -r .name environment.yml) --file environment-benchmark.yml
conda activate $(yq -r .name environment.yml)
pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
pytest -nauto tests --doctest-modules src/glum/
