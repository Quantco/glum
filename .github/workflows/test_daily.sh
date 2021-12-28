#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $1

export PANDAS_VERSION=$2
export NUMPY_VERSION=$3
export SCIKIT-LEARN_VERSION=$4

mamba install -y yq

cat environment.yml > /tmp/environment.yml
DEPENDENCIES=("python" "numpy" "pandas" "scikit")
for dependency in "${DEPENDENCIES[@]}"; do
    _dependency="${dependency^^}_VERSION"
    version=${!_dependency}
    # Delete any existing entry in the environment.yml to avoid duplicate entries when
    # appending
    yq -Y --in-place "del( .dependencies[] | select(startswith(\"${dependency}\")))" /tmp/environment.yml
    # Handle hyphenation
    # (see e.g. https://unix.stackexchange.com/questions/23659/can-shell-variable-name-include-a-hyphen-or-dash)
    if [[ "${dependency}" == "scikit" ]]; then
        dependency="scikit-learn"
    fi
    yq -Y --in-place ". + {dependencies: [.dependencies[], \"${dependency}=${version}\"] }" /tmp/environment.yml
done

mamba env create -f /tmp/environment.yml
mamba env update -n $(yq -r .name environment.yml) --file environment-benchmark.yml
conda activate $(yq -r .name environment.yml)

PRE_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
if [[ "${NUMPY_VERSION}" == "nightly" ]]; then
    echo "Installing Numpy nightly"
    conda uninstall -y --force numpy
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS numpy
fi
if [[ "${PANDAS_VERSION}" == "nightly" ]]; then
    echo "Installing Pandas nightly"
    conda uninstall -y --force pandas
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS pandas
fi
if [[ "${SCIKIT_VERSION}" == "nightly" ]]; then
    echo "Install scikit-learn nightly"
    conda uninstall -y --force scikit-learn
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS scikit-learn
fi

pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .
pytest -nauto tests --doctest-modules src/

# Check that the readme example will work by running via doctest.
# We run outside the repo to make the test a bit more similar to
# a user running after installing with conda.
mkdir ../temp
cp README.md ../temp
cd ../temp
python -m doctest -v README.md