#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J qwen2_5_7b_lora_1gpu
#SBATCH -o runs/%x_%j/logs/train.out
#SBATCH -e runs/%x_%j/logs/train.err

set -euo pipefail
echo "Host: $(hostname) GPUs:$CUDA_VISIBLE_DEVICES Date: $(date)"

export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME





# --- W&B config ---
export WANDB_PROJECT=mt_project_bg_en
export WANDB_ENTITY=ivankanev2-mbzuai     
export WANDB_NAME="qwen2.5-7b-lora-1gpu-${SLURM_JOB_ID}"

RUN_DIR="runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/eval"

export WANDB_DIR="$RUN_DIR/logs"
# export WANDB_MODE=offline


source ~/.bashrc 2>/dev/null || true
source "$HOME/mt_project/.venv/bin/activate"

python -u train/train_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --eval_file  data/dev.jsonl \
  --output_dir "$RUN_DIR/checkpoints" \
  --epochs 3 \
  --lr 1e-4 \
  --batch_size 2 \
  --grad_accum 16 \
  --max_len 1024

python -u batch_eval.py \
  --src data/english_ref_1.docx \
  --ref data/bulgarian_ref_1.docx \
  --models "$RUN_DIR/checkpoints" \
  --outdir "$RUN_DIR/eval" \
  --no_glossary --no_bg_normalize \
  --thr 0.55 --batch 16 --export_disagreements
