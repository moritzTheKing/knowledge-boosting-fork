#!/bin/bash
# run_allocation.sh

echo "Train job " $SLURM_STEP_ID " on "$(hostname)

# Hydra (Python Framework) printed den gesamten Traceback für Debugging
HYDRA_FULL_ERROR=1 
echo "wird das hier ausgeprinted"

srun -l \
    --job-name=sub_job \    
    --time=2-00:00:00 \
    --partition=gpu \
    --mem=16000M \
    --gres=gpu:1080:1 \
    --cpus-per-task=2 \
    --output=./output/slurm.%j.out \
    python -m src.trainer \
    --run_dir "./runs/job_${SLURM_STEP_ID}" \
    --config configs/baselines/BM-TSE-500.json \
    --ckpt "runs/job_1080501/lightning_logs/version_0/checkpoints/epoch=45-step=358752.ckpt" \
    #--test
# srun -l --job-name=sub_job --time=2-00:00:00 --partition=gpu --mem=16000M --gres=gpu:1080:1 --cpus-per-task=2 --output=./output/slurm.%j.out python -m src.trainer --run_dir "./runs/job_${SLURM_STEP_ID}" --config configs/baselines/BM-TSE-500.json --ckpt "runs/job_1080501/lightning_logs/version_0/checkpoints/epoch=45-step=358752.ckpt" 
echon "wird das nach srun ausgeprinted"