#!/bin/bash
#SBATCH --job-name=eval_qwen2_5_7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_qwen2_5_7b_%j.out

set -euo pipefail
cd ~/mt_project
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=true

python batch_eval.py \
  --src data/english_ref_1.docx \
  --ref data/bulgarian_ref_1.docx \
  --models "Qwen/Qwen2.5-7B-Instruct" \
  --outdir results_qwen2_5_7b --clean_outdir \
  --no_glossary --no_bg_normalize \
  --thr 0.50 --export_disagreements
