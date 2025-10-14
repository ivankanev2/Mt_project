#!/bin/bash
#SBATCH -A faculty
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -J infer_gemma12b
#SBATCH -o logs/infer_gemma12b_%j.out

set -euo pipefail
echo "Host: $(hostname)  GPUs:$CUDA_VISIBLE_DEVICES  Date: $(date)"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

#still having problems with this code and the lab
# Multi-GPU translation
python3 mp_infer_generic.py \
  --model_id google/gemma-2-12b-it \
  --src_doc data/english_ref_1.docx \
  --out_txt results_gemma2_12b/system_gemma2_12b_dp.out \
  --gpus 4 --batch 2 --max_new_tokens 512
