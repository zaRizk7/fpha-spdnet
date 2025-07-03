#!/bin/bash

export PYTHONWARNINGS="ignore"

# Allow MODEL to be set via first argument, default to "spdnet-bn"
MODEL="${1:-spdnet-bn}"

clear
rm -rf "logs/$MODEL/reproduction"

python -m fpha_spdnet fit -c "configs/$MODEL.yml" --seed_everything=42
