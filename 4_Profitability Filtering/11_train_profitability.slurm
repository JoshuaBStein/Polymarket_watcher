#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ProfModel
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=47:59:59
#SBATCH --output=ProfModel_%j.out

# 1. Initialize Conda (required for Slurm)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Activate your new environment
conda activate fintech_gpu

export TMPDIR=/scratch/jbs263/tmp
mkdir -p $TMPDIR
echo "Temporary directory set to: $TMPDIR"

# 3. Run your script
python profitability_model.py
