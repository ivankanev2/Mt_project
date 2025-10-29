#!/bin/bash
#SBATCH --job-name=infer_bggpt_9b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out

set -euxo pipefail

# Always start where you submitted from
cd "$SLURM_SUBMIT_DIR"

# Install deps to user site (no sudo, no venv)
python3 -m pip install --user -U pip
python3 -m pip install --user -r requirements.txt

# Make sure Python can see user-site on some clusters
USER_SITE="$(python3 -m site --user-site)"
export PYTHONPATH="${USER_SITE}:${PYTHONPATH-}"
# Sanity checks
hostname
test -f data/BGP-GW-GL-203_Revision_A_eng.docx
test -f data/BGP-GW-GL-203_Revision_A_bg.docx

# Run
python3 batch_eval.py \
  --src data/BGP-GW-GL-203_Revision_A_eng.docx \
  --ref data/BGP-GW-GL-203_Revision_A_bg.docx \
  --models "INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0" \
  --outdir results_bggpt_9b --clean_outdir \
  --no_glossary --no_bg_normalize \
  --thr 0.50 --export_disagreements
