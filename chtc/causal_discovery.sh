#!/bin/bash
set -euo pipefail

run_key=$1
run_name=$2
data_file=$3
alpha=$4
sample_size=$5
depth=$6

echo "Starting $run_name ($run_key) on $data_file"

python3 causal_discovery.py \
    --run_key "$run_key" \
    --run_name "$run_name" \
    --data_path "$data_file" \
    --alpha "$alpha" \
    --sample_size "$sample_size" \
    --depth "$depth"

echo "Done."
