#!/bin/bash
#SBATCH -A faculty
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -J infer_gemma3_12b
#SBATCH -o logs/infer_gemma3_12b_%j.out

set -euo pipefail
echo "Host: $(hostname)  GPUs:$CUDA_VISIBLE_DEVICES  Date: $(date)"

#code is still under renovation
# Hugging Face cache/env
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster downloads

# If your mp_infer_generic.py expects --model_id/--gpus/--batch etc., run it here.
# Start conservative on batch; you can bump once it works.
python3 mp_infer_generic.py \
  --model_id google/gemma-3-12b-it \
  --src_doc data/english_ref_1.docx \
  --out_txt results_gemma3_12b/system_gemma3_12b_dp.out \
  --gpus 4 --batch 2 --max_new_tokens 512
