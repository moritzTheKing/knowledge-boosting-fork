#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=too_big
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1080:3
#SBATCH --cpus-per-task=2   
#SBATCH --output=./output/slurm.%j.out

#SBATCH --begin=now

echo "Train job " $SLURM_JOB_ID " on "$(hostname)

HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir logdir --config configs/baselines/BM-TSE-500.json
