#!/bin/bash
# run_locally.sh

JOB_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./runs/job_${JOB_ID}"
OUTPUT_FILE="./output/slurm.${JOB_ID}.out"

echo "Train job $JOB_ID on $(hostname)" | tee -a "$OUTPUT_FILE"

mkdir -p "$RUN_DIR"

HYDRA_FULL_ERROR=1 python -m src.trainer --run_dir "$RUN_DIR" --config configs/TSE_joint/B0-TSE.json --ckpt runs/job_1079588/lightning_logs/version_0/checkpoints/epoch=19-step=150012.ckpt --test >> "$OUTPUT_FILE" 2>&1
