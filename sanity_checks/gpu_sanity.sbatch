#!/bin/bash
#SBATCH -A faculty
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH -o logs/gpu_sanity_%j.out

echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi | head -n 20

python3 - <<'PY'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

mkdir -p logs
sbatch gpu_sanity.sbatch
squeue -u $USER
tail -f logs/gpu_sanity_*.out
a
