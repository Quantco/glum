#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd ${SCRIPT_DIR}

set -eo pipefail
source ~/.profile

# run with latest
# mamba create -n latest 'python=3.9.15' pip
# conda activate latest
# pip install 'glum==2.5.1'
# python issue_628.py

# test conda build - latest
mamba create -n new-conda 'python=3.9.15' 'glum=2.5.1'
conda activate new-conda
python issue_628.py

# reproduce the original issue
# mamba create -n orig 'python=3.9.15' pip
# conda activate orig
# pip install 'glum==2.1.2' 'numpy==1.23.5' 'tabmat==3.1.2'
# python issue_628.py

# test conda build - original
mamba create -n orig-conda 'python=3.9.15' 'glum=2.1.2' 'numpy=1.23.5' 'tabmat=3.1.2'
conda activate orig-conda
python issue_628.py

# pypi build - conda deps - latest
mamba create -n new-conda-pip 'python=3.9.15'
conda activate new-conda-pip
mamba install --deps-only 'glum=2.5.1'
pip install 'glum==2.5.1'
python issue_628.py
