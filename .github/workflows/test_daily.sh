#!/bin/bash

set -exo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/base.sh $1

PANDAS_VERSION=$2
NUMPY_VERSION=$3
SCIKIT_VERSION=$4
TABMAT_VERSION=$5

mamba install -y yq

cat environment.yml > /tmp/environment.yml

# pin version of some libraries, if specified
LIBRARIES=("python" "pandas" "numpy" "scikit" "tabmat")
for library in "${LIBRARIES[@]}"; do
    varname="${library^^}_VERSION"
    version=${!varname}
    if [[ -n "$version" && "$version" != "nightly" ]]; then
        if [[ "${library}" == "scikit" ]]; then
            library="scikit-learn"
        fi
        yq -Y --in-place ". + {dependencies: [.dependencies[], \"${library}=${version}\"]}" /tmp/environment.yml
    fi
done

cat /tmp/environment.yml

mamba env create -f /tmp/environment.yml
mamba env update -n $(yq -r .name environment.yml) --file environment-benchmark.yml
conda activate $(yq -r .name environment.yml)

PRE_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
if [[ "$NUMPY_VERSION" == "nightly" ]]; then
    echo "Installing Numpy nightly"
    conda uninstall -y --force numpy
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS numpy
fi
if [[ "$PANDAS_VERSION" == "nightly" ]]; then
    echo "Installing Pandas nightly"
    conda uninstall -y --force pandas
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS pandas
fi
if [[ "$SCIKIT_VERSION" == "nightly" ]]; then
    echo "Install scikit-learn nightly"
    conda uninstall -y --force scikit-learn
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS scikit-learn
fi
if [[ "$TABMAT_VERSION" == "nightly" ]]; then
    # This needs to be done before any 'uninstall --force'
    echo "Install compilation dependencies"
    mamba install -y c-compiler cxx-compiler cython jemalloc-local libgomp mako xsimd

    echo "Install tabmat nightly"
    # TODO: switch to a special channel once we have a Quetz instance up and running
    conda uninstall -y --force tabmat
    pip install git+https://github.com/Quantco/tabmat
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