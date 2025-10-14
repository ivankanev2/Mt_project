#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/gpu_job_%j.out

nvidia-smi
source ~/.bashrc 2>/dev/null || true
cd ~/mt_project
source .venv/bin/activate
python -V
# example: evaluate your LoRA adapter
python batch_eval.py \
  --src data/english_ref_1.docx \
  --ref data/bulgarian_ref_1.docx \
  --models "adapters/qwen2_5_3b_bg_lora" \
  --outdir results_lora_eval --clean_outdir \
  --no_glossary --no_bg_normalize --thr 0.50 --export_disagreements
