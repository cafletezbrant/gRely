#!/bin/bash
#SBATCH --job-name=eqtl_pos               # Job name
#SBATCH --array=0-37                       # Array of tasks
#SBATCH --partition=batch_cpu              # Partition name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=8                         # Request 8 cores
#SBATCH --mem=300G                          # Memory total in MB
#SBATCH --time=1-00:00                     # Runtime in D-HH:MM
#SBATCH --qos=1d                           # QoS policy
#SBATCH --output=/home/rafia1/scratch/outputs/pos_%A_%a.out
#SBATCH --error=/home/rafia1/scratch/errors/pos_%A_%a.err

ml load Miniforge3
conda activate dream
python /home/rafia1/scratch/codes/find_pos.py $SLURM_ARRAY_TASK_ID