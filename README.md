# EN→BG Machine Translation Evaluation

This project evaluates **English → Bulgarian translation models** (both MT and LLMs) against professional reference translations.  
It handles noisy DOCX/PDF/XML documents, aligns English source lines with Bulgarian references, and computes quality metrics (BLEU, chrF).  
Outputs are saved in both **CSV** and **Excel** for easy inspection.

---

## Features
- 📄 Extracts text from `.docx`, `.pdf`, `.xml` (XLIFF/TMX) references
- 📊 Aligns EN and BG documents with a **global DP aligner** (one row per English line)
- 🔍 Outputs **compare tables**, **disagreements only**, and a **summary sheet**
- ✅ Supports glossary enforcement and BG normalization (optional)
- 📈 Computes **BLEU** and **chrF** metrics using [sacreBLEU](https://github.com/mjpost/sacrebleu)
- 🔤 Writes both `.csv` (UTF-8 BOM) and `.xlsx` (Excel-ready) files

---

## Installation

Clone this repo and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Apple Silicon (M-series) GPU

PyTorch packages distributed for macOS already ship with Metal (MPS) support.  
To ensure the scripts use your MacBook’s GPU:

- Install the latest PyTorch build for macOS/arm64 (e.g. `pip install --upgrade torch torchvision torchaudio`).
- Export `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops that miss MPS kernels automatically fall back to CPU:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- Run a quick check before training/inference:
  ```bash
  python - <<'PY'
  import torch
  print("MPS available:", torch.backends.mps.is_available())
  PY
  ```
- No CLI changes are needed—`batch_eval.py` and the training scripts now auto-select `cuda` (if present), otherwise `mps`, then CPU.

### Seq2Seq LoRA fine-tuning (OPUS / Marian)

Use `train/train_lora_seq2seq.py` to add a Bulgarian adapter on top of an encoder-decoder model such as `Helsinki-NLP/opus-mt-en-bg`:

```bash
python train/train_lora_seq2seq.py \
  --model_id Helsinki-NLP/opus-mt-en-bg \
  --train_file data/train.jsonl \
  --eval_file data/dev.jsonl \
  --output_dir adapters/opus_en_bg_lora \
  --epochs 3 --batch_size 4 --grad_accum 16 \
  --max_source_len 256 --max_target_len 256 \
  --gen_max_new_tokens 128 \
  --evals_per_epoch 4 --load_best_model \
  --lora_r 256 --lora_alpha 128 --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,out_proj
```

The script reuses the chat-formatted `train.jsonl` / `dev.jsonl` produced by `make_pairs.py`, automatically picks CUDA → MPS → CPU, and logs sacreBLEU via W&B. Point `batch_eval.py` at `--models "Helsinki-NLP/opus-mt-en-bg" --lora_adapter adapters/opus_en_bg_lora` (or your `runs/...` directory) to evaluate the trained adapter. LoRA adapters are now supported for seq2seq models inside `batch_eval.py`, so evaluations reflect your fine-tunes accurately.

Use `--evals_per_epoch` to request a specific number of full-dev BLEU runs per epoch (the script derives `eval_steps` from dataset size, batch, and grad accumulation). Override with `--eval_steps`/`--save_steps` if you need absolute step counts. With `--load_best_model` the checkpoint that achieves the highest BLEU is restored at the end. Add `--no_lora` if you want to skip adapters and evaluate/train the base model directly.

### Generating train/dev JSONL splits

All training/eval commands expect the chat-style JSONL produced by `make_pairs.py`. Run it from the repo root, pointing at your DOCX pairs, for example:

```bash
python make_pairs.py \
  --pairs "data/Copy of BGP-GW-GL-203_Revision_A_eng.docx,data/Copy of BGP-GW-GL-203_Revision_A_bg.docx" \
  --outdir data/dev_tmp --thr 0.55 --dev_ratio 0.0
cat data/dev_tmp/train.jsonl data/dev_tmp/dev.jsonl > data/dev.jsonl

python make_pairs.py \
  --pairs \ 
    "data/Copy of APP-GW-G0R-004_Revision_A_eng.docx,data/Copy of APP-GW-G0R-004_Revision_A_bg.docx" \ 
    "data/Copy of APP-GW-G0R-014_Revision_A_eng.docx,data/Copy of APP-GW-G0R-014_Revision_A_bg.docx" \ 
    "data/Copy of APP-GW-GL-059_Revision_2_eng.docx,data/Copy of APP-GW-GL-059_Revision_2_bg.docx" \ 
    "data/Copy of BGP-GW-GL-200_Revision_B_eng.docx,data/Copy of BGP-GW-GL-200_Revision_B_bg.docx" \
  --outdir data/train_tmp --thr 0.55 --dev_ratio 0.1
mv data/train_tmp/train.jsonl data/train.jsonl
```

This ensures `data/dev.jsonl` mirrors the “normal” DOCX evaluation exactly, while `data/train.jsonl` aggregates the remaining documents.

Usage

Run evaluation on your English source and Bulgarian reference:
python batch_eval.py \
  --src data/english_ref.docx \
  --ref data/bulgarian_ref.docx \
  --models "Helsinki-NLP/opus-mt-en-bg" \
  --outdir results_run --clean_outdir \
  --no_glossary --no_bg_normalize \
  --thr 0.50 \
  --export_disagreements

  Arguments
  
	•	--src : English source file (.docx / .pdf)
  
	•	--ref : Bulgarian reference file (.docx / .pdf / .xml)
  
	•	--models : Comma-separated Hugging Face model IDs
  
	•	--outdir : Results folder
  
	•	--clean_outdir : Clear old results before writing new ones
  
	•	--no_glossary : Disable glossary term enforcement
  
	•	--no_bg_normalize : Disable Bulgarian variant normalization
  
	•	--thr : Similarity threshold for DP alignment (default 0.50)
  
	•	--export_disagreements : Write a file with mismatched rows only

Outputs

Each run produces:

	•	system_<model>.out
  
→ Model translations (one per English line)

	•	compare_anchored_<model>.csv / .xlsx
  
→ Side-by-side table:

English | System BG | Reference BG | en_idx | bg_idx | segment

	•	disagreements_<model>.csv / .xlsx
  
→ Only rows where system ≠ reference

	•	summary.csv / .xlsx
  
→ Overall results per model (aligned pairs, BLEU, chrF, file paths)

	•	debug/dbg_*
  
→ Debug dumps of raw and cleaned extraction

Example

Console output:
DP input sizes: EN=3046 | BG_REF=3161

Translated 3046 lines.

DP produced rows: 3046 (should be close to number of EN lines)

aligned subset: n=2895, BLEU=43.83, chrF=68.53

Important things to note:

	•	BLEU/chrF are computed case-insensitive (all lowercased before scoring).
  
	•	Segmentation is line-based (paragraphs/table cells), not sentence-segmented.
  
	•	Empty BG cells in the compare table mean no confident match (gap).
  
	•	Use --glossary (CSV file) to enforce domain-specific terms if needed.
  
	•	For large docs, evaluation may take several minutes.
