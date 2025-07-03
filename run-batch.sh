#!/bin/bash

export PYTHONWARNINGS="ignore"

rm -rf logs

./run.sh spdnet-bn
./run.sh u-spdnet-bn
./run.sh spdnet-bn
./run.sh u-spdnet-bn
