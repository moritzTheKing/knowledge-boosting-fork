#!/bin/bash

#SBATCH --time=0-02:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=test_big
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1080:3
#SBATCH --cpus-per-task=2   
#SBATCH --output=./output/slurm.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.marksteller@tu-bs.de

#SBATCH --begin=now

RUN_DIR="./runs/job_${SLURM_JOB_ID}"

echo "Train job " $SLURM_JOB_ID " on "$(hostname)

HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir "$RUN_DIR" --config configs/baselines/BM-TSE-500.json --ckpt "runs/job_1080733/lightning_logs/version_0/checkpoints/epoch=99-step=958800.ckpt" --test