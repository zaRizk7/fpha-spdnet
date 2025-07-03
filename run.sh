#!/bin/bash

export PYTHONWARNINGS="ignore"

MODEL="spdnet-bn"

clear
rm -rf logs/$MODEL/reproduction

python -m fpha_spdnet fit -c configs/$MODEL.yml --seed_everything=42
