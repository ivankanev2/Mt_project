#!/bin/bash
#SBATCH -A faculty
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=06:00:00
#SBATCH --mem=120G
#SBATCH -J qwen2_5_7b_lora_en_bg
#SBATCH -o runs/%x_%j/logs/train.out
#SBATCH -e runs/%x_%j/logs/train.err

set -euo pipefail

echo "Host: $(hostname)  GPUs:$CUDA_VISIBLE_DEVICES  Date: $(date)"

# Caches
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export WANDB_MODE=offline   # no accidental WANDB spam

# Create a run directory with job id
RUN_DIR="runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/eval"

source ~/.bashrc 2>/dev/null || true
# Activate your project venv if you have it
source "$HOME/mt_project/.venv/bin/activate"

# Train (your train_lora.py already exists in repo)
python3 train_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl data/train.jsonl \
  --dev_jsonl data/dev.jsonl \
  --output_dir "$RUN_DIR/checkpoints" \
  --epochs 4 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_seq_len 1024 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --lr_scheduler cosine \
  --bf16 True \
  --gradient_checkpointing True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_total_limit 2 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --log_interval 20

# Optional quick eval on the whole doc after training
python3 batch_eval.py \
  --src data/english_ref_1.docx \
  --ref data/bulgarian_ref_1.docx \
  --models "$RUN_DIR/checkpoints" \
  --outdir "$RUN_DIR/eval" \
  --no_glossary --no_bg_normalize \
  --thr 0.55 --batch 16 --export_disagreements