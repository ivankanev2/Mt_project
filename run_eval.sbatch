#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J eval_lora
#SBATCH -o logs/%x_%j.out
#SBATCH --export=ALL,SLURM_CPU_BIND=none

set -euo pipefail
module purge || true

cd ~/mt_project
source .venv/bin/activate

python - <<'PY'
import torch, os
print("CUDA visible:", torch.cuda.device_count(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

python batch_eval.py \
  --src data/english_ref_1.docx \
  --ref data/bulgarian_ref_1.docx \
  --models "adapters/qwen2_5_3b_bg_lora" \
  --outdir results_lora_eval --clean_outdir \
  --no_glossary --no_bg_normalize \
  --thr 0.50 --export_disagreements