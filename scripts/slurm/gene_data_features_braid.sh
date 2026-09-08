#!/bin/bash
#SBATCH --job-name=data               # Job name
#SBATCH --array=0-200                       # Array of tasks
#SBATCH --account=owner_gred_braid_gpu    # account name
#SBATCH --partition=owner_gred_braid_gpu    # Partition name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=8                         # Request 8 cores
#SBATCH --mem=64G                          # Memory total in MB
#SBATCH --gres=gpu:1                          # gpu
#SBATCH --time=0-1:00                     # Runtime in D-HH:MM
#SBATCH --output=/home/rafia1/scratch/outputs/features_%A_%a.out
#SBATCH --error=/home/rafia1/scratch/errors/features_%A_%a.err

ml load Miniforge3
conda activate borzoi
python /home/rafia1/scratch/codes/gene_data_features.py $SLURM_ARRAY_TASK_ID