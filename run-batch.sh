#!/bin/bash

export PYTHONWARNINGS="ignore"

rm -rf logs

./run.sh spdnet
./run.sh u-spdnet
./run.sh spdnet-bn
./run.sh u-spdnet-bn
