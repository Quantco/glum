#!/bin/bash

python -m venv .env
source .env/bin/activate
pip install 'glum==2.1.2' 'numpy==1.23.5' 'tabmat==3.1.2'
python issue_628.py
