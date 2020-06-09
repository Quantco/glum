#!/bin/bash

set -exo pipefail

wget https://github.com/conda-forge/miniforge/releases/download/4.8.3-4/Miniforge3-4.8.3-4-MacOSX-x86_64.sh -O miniforge.sh
/bin/bash miniforge.sh -b -p /opt/conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base
conda config --system --set auto_update_conda false
conda config --system --prepend channels quantco_main
conda config --system --set custom_channels.quantco_main https://${CONDA_CHANNEL_UPLOAD_USER}:${CONDA_CHANNEL_UPLOAD_PASSWORD}@conda.quantco.cloud
conda install -y mamba=0.2.12
