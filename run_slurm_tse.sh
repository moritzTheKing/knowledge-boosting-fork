#!/bin/bash

#SBATCH --time=0-00:20:00
#SBATCH --partition=gpu
#SBATCH --job-name=tse_cr_at
#SBATCH --mem=2000M
#SBATCH --gres=gpu:1080:1
#SBATCH --cpus-per-task=2   
#SBATCH --output=./output/slurm.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.marksteller@tu-bs.de

#SBATCH --begin=now

RUN_DIR="./runs/job_${SLURM_JOB_ID}"

echo "Train job " $SLURM_JOB_ID " on "$(hostname)

#HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir "$RUN_DIR" --config configs/TSE_joint/B0-TSE.json --ckpt "runs/job_1080867/lightning_logs/version_0/checkpoints/epoch=19-step=144456.ckpt" --test
HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir "$RUN_DIR" --config configs/TSE_joint/B0-TSE.json #--ckpt "runs/job_1080867/lightning_logs/version_0/checkpoints/epoch=19-step=144456.ckpt" --test
