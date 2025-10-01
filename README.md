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
