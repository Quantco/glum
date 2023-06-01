#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd ${SCRIPT_DIR}

set -exo pipefail
source ~/.profile

mamba create -n python 'python=3.9.15' pip
conda activate python
pip install 'glum==2.1.2' 'numpy==1.23.5' 'tabmat==3.1.2'
python issue_628.py
