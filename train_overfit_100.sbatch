#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J qwen2_5_7b_overfit100
#SBATCH -o runs/%x_%j/logs/train.out
#SBATCH -e runs/%x_%j/logs/train.err

set -euo pipefail
echo "Host: $(hostname) GPUs:$CUDA_VISIBLE_DEVICES Date: $(date)"

# --- Cache setup ---
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

# --- W&B setup ---
export WANDB_PROJECT=mt_project_bg_en
export WANDB_ENTITY=ivankanev2-mbzuai
export WANDB_NAME="overfit100-${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false

# Define run directory BEFORE setting WANDB_DIR
RUN_DIR="runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/eval"

export WANDB_DIR="$RUN_DIR/logs"
# export WANDB_MODE=offline

# --- Environment ---
source ~/.bashrc 2>/dev/null || true
source "$HOME/mt_project/.venv/bin/activate"

# --- Training ---
python -u train/train_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.first100.jsonl \
  --eval_file  data/dev.first100.jsonl \
  --output_dir "$RUN_DIR/checkpoints" \
  --epochs 10 \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_len 512