#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=KB
#SBATCH --mem=16000M
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2   
#SBATCH --output=./slurm.%j.out

#SBATCH --begin=now

echo "Train job " $SLURM_JOB_ID " on "$(hostname)


HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir logdir/test --config configs/TSE_joint/B0-TSE.json
