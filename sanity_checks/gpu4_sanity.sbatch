#!/bin/bash
#SBATCH -A faculty
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH -J gpu4_sanity
#SBATCH -o logs/gpu4_sanity_%j.out
set -euo pipefail
echo "Host: $(hostname)  Date: $(date)"
nvidia-smi -L || true
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY
