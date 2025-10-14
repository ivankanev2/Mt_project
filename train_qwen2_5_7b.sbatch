#!/bin/bash
#SBATCH --job-name=train_q7b_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_q7b_%j.out

set -euo pipefail
cd ~/mt_project
source .venv/bin/activate

python train_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --eval_file data/dev.jsonl \
  --output_dir adapters/qwen2_5_7b_bg_lora_clean \
  --epochs 1 \
  --lr 2e-4 \
  --batch_size 2 \
  --grad_accum 8
