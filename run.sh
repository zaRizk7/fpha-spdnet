#!/bin/bash

export PYTHONWARNINGS="ignore"

clear
rm -rf logs checkpoints
python -m fpha_spdnet fit -c configs/reproduce-uspdnet.yml --seed_everything=0
