#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd ${SCRIPT_DIR}

set -eo pipefail
source ~/.profile

# run with latest
mamba create -n latest 'python=3.9.15' pip
conda activate latest
pip install 'glum==2.5.1'
python issue_628.py

# reproduce the original issue
mamba create -n orig 'python=3.9.15' pip
conda activate orig
pip install 'glum==2.1.2' 'numpy==1.23.5' 'tabmat==3.1.2'
python issue_628.py
