#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=IdenArch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=47:59:59
#SBATCH --output=newcsv_%j.out

# 1. Initialize Conda (required for Slurm)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Activate your new environment
conda activate fintech_gpu

# 3. Run your script
python detect_anomalies_iso.py

