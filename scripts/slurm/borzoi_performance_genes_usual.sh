#!/bin/bash
#SBATCH --job-name=gene_pred               # Job name
#SBATCH --array=0-200                       # Array of tasks
#SBATCH --partition=batch_gpu    # Partition name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=4                         # Request 8 cores
#SBATCH --mem=64G                          # Memory total in MB
#SBATCH --gres=gpu:a100:1                          # gpu
#SBATCH --time=0-12:00                     # Runtime in D-HH:MM
#SBATCH --qos=1d                           # QoS policy
#SBATCH --output=/home/rafia1/scratch/outputs/pred_%A_%a.out
#SBATCH --error=/home/rafia1/scratch/errors/pred_%A_%a.err

ml load Miniforge3
conda activate borzoi
python /home/rafia1/scratch/codes/borzoi_performance_genes.py $SLURM_ARRAY_TASK_ID