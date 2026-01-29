#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=GenCSV
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=47:59:59
#SBATCH --output=newcsv_%j.out

# 1. Initialize Conda (required for Slurm)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Activate your new environment
conda activate fintech_gpu

echo "Conda environment successfully loaded"
echo "Will begin analysis now!"

# 3. Run your script
python newcsv.py
