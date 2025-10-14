#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --partition=ws-ia
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=12G
#SBATCH --output=logs/cpu_job_%j.out

source ~/.bashrc 2>/dev/null || true
cd ~/mt_project
source .venv/bin/activate
python -V
# your command here, e.g.:
python batch_eval.py --help
