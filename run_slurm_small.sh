#!/bin/bash

#SBATCH --time=0-00:10:00
#SBATCH --partition=debug
#SBATCH --job-name=test_small
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1080:1
#SBATCH --cpus-per-task=2   
#SBATCH --output=./output/slurm.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.marksteller@tu-bs.de

#SBATCH --begin=now

RUN_DIR="./runs/job_${SLURM_JOB_ID}"

echo "Train job " $SLURM_JOB_ID " on "$(hostname)

HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir "$RUN_DIR" --config configs/baselines/SM-TSE-25.json #--ckpt "runs/job_1079637/lightning_logs/version_0/checkpoints/epoch=99-step=777840.ckpt" --test