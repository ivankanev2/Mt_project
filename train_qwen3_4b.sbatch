#!/bin/bash
#SBATCH --job-name=qwen3_4b_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --output=logs/qwen3_4b_%j.out

set -euo pipefail

cd ~/mt_project
source .venv/bin/activate

# Good CPU setup for dataloaders/tokenization
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
# Optional: keeps HF cache under your home (avoid refetching)
export HF_HOME=$HOME/.cache/huggingface

python train_lora.py \
  --model_id Qwen/Qwen3-4B-Instruct \
  --train_file data/train.jsonl \
  --eval_file  data/dev.jsonl \
  --output_dir adapters/qwen3_4b_bg_lora \
  --epochs 1 \
  --lr 2e-4 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_len 2048
