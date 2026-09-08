#!/bin/bash
#SBATCH --job-name=pred               # Job name
#SBATCH --array=660-700                       # Array of tasks
#SBATCH --partition=batch_gpu    # Partition name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=4                         # Request 8 cores
#SBATCH --mem=64G                          # Memory total in MB
#SBATCH --gres=gpu:1                          # gpu
#SBATCH --time=0-6:00                     # Runtime in D-HH:MM
#SBATCH --qos=1d                           # QoS policy
#SBATCH --output=logs/pred_%A_%a.out
#SBATCH --error=logs/pred_%A_%a.err

ml load Miniforge3
conda activate borzoi
python scripts/03_borzoi_performance/borzoi_performance.py $SLURM_ARRAY_TASK_ID