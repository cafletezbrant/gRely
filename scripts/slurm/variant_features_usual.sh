#!/bin/bash
#SBATCH --job-name=features               # Job name
#SBATCH --array=0-999                       # Array of tasks
#SBATCH --partition=batch_cpu    # Partition name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=4                         # Request 8 cores
#SBATCH --mem=16G                          # Memory total in MB
#SBATCH --time=0-24:00                     # Runtime in D-HH:MM
#SBATCH --qos=1d                           # QoS policy
#SBATCH --output=/home/rafia1/scratch/outputs/features_%A_%a.out
#SBATCH --error=/home/rafia1/scratch/errors/features_%A_%a.err

ml load Miniforge3
conda activate dream
python /home/rafia1/scratch/codes/variant_features.py $SLURM_ARRAY_TASK_ID