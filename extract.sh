#!/bin/bash

DATA_DIR="Hand_pose_annotation_v1"
SPLIT_FILE="data_split_action_recognition.txt"

CONFIG_NAMES=("cov" "corr" "cov-lw" "corr-lw")
CONFIG_VALUES=(
    "false emp"
    "true emp"
    "false lw"
    "true lw"
)

for i in "${!CONFIG_NAMES[@]}"; do
    name="${CONFIG_NAMES[$i]}"
    read -r standardize estimator <<<"${CONFIG_VALUES[$i]}"
    python -m fpha_spdnet.data "$DATA_DIR" \
        --split_file "$SPLIT_FILE" \
        --output_file "data/${name}.h5" \
        --standardize "$standardize" \
        --estimator "$estimator"
    sleep 2
done
