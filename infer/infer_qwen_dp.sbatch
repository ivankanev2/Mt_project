#!/bin/bash
#SBATCH --job-name=mt_infer_qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8                # one task == one process == one GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer_%j.out

module purge 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
cd ~/mt_project
source .venv/bin/activate

# Split workload into 8 shards and run 8 identical processes.
# Each process selects its GPU via LOCAL_RANK and processes its shard.
# this shit didnt work at all

srun --cpu-bind=none bash -lc '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  python mp_infer_qwen.py \
    --model_base Qwen/Qwen2.5-3B-Instruct \
    --adapter adapters/qwen2_5_3b_bg_lora_clean \
    --src data/english_ref_1.docx \
    --ref data/bulgarian_ref_1.docx \
    --outdir results_multi \
    --rank $RANK --world_size $WORLD_SIZE --local_rank $LOCAL_RANK
'

# Merge shards into the normal files (CSV/XLSX/summary)
python merge_shards.py --outdir results_multi
