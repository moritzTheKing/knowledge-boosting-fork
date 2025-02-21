#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=too_big
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1080:1
#SBATCH --cpus-per-task=2   
#SBATCH --output=./output/slurm.%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marksteller@tu-bs.de

#SBATCH --begin=now

RUN_DIR="./runs/job_${SLURM_JOB_ID}"

echo "Train job " $SLURM_JOB_ID " on "$(hostname)

HYDRA_FULL_ERROR=1 srun -l python -m src.trainer --run_dir "$RUN_DIR" --config configs/baselines/BM-TSE-500.json --test