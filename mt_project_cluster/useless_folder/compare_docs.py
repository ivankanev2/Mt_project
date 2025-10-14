import argparse
import csv
import os
import sys
from pathlib import Path

# ---- File readers ----
def extract_plain_text_docx(path):
    try:
        from docx import Document
    except ImportError:
        print("Please install python-docx: pip install python-docx", file=sys.stderr)
        sys.exit(1)
    doc = Document(path)
    lines = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)
    return lines

def extract_plain_text_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Please install pypdf: pip install pypdf", file=sys.stderr)
        sys.exit(1)
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def extract_text_auto(path):
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return extract_plain_text_docx(path)
    elif ext == ".pdf":
        return extract_plain_text_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .docx or .pdf")

# ---- Translation (M2M-100 418M) ----
def load_m2m_model(model_name="facebook/m2m100_418M", device=None):
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tok = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    if device is None:
        # Auto-pick cuda if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tok, model, device

def translate_lines_en_bg(lines, tok, model, device, batch_size=16):
    tok.src_lang = "en"
    out = []
    for i in range(0, len(lines), batch_size):
        batch = [s if s.strip() else "" for s in lines[i:i+batch_size]]
        # Tokenize with truncation; very long lines will be truncated
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("bg"))
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend([d.strip() for d in dec])
    return out

# ---- Normalization ----
def normalize_lines(lines):
    norm = []
    for s in lines:
        s = " ".join(s.split())  # collapse internal whitespace
        s = s.strip()
        if s:
            norm.append(s)
    return norm

# ---- Scoring (optional; will work only if lengths roughly align) ----
def try_scores(sys_out, ref_lines):
    try:
        import sacrebleu
    except ImportError:
        print("Tip: pip install sacrebleu for metrics")
        return
    m = min(len(sys_out), len(ref_lines))
    if m < 5:
        print("Not enough overlapping lines to compute metrics.")
        return
    sys_small = [s.lower().strip() for s in sys_out[:m]]
    ref_small = [[r.lower().strip() for r in ref_lines[:m]]]
    bleu = sacrebleu.corpus_bleu(sys_small, ref_small).score
    chrf = sacrebleu.corpus_chrf(sys_small, ref_small).score
    print(f"Quick metrics on first {m} lines (may be unreliable if structure differs):")
    print(f"  BLEU: {bleu:.2f}")
    print(f"  chrF: {chrf:.2f}")

def main():
    ap = argparse.ArgumentParser(description="Extract plain text, translate EN->BG, and export side-by-side CSV.")
    ap.add_argument("--src", required=True, help="English source file (.docx or .pdf)")
    ap.add_argument("--ref", required=True, help="Bulgarian reference file (.docx or .pdf)")
    ap.add_argument("--out", default="compare.csv", help="Output CSV path (default: compare.csv)")
    ap.add_argument("--save-system", default="system.out", help="Optional: save system translations to this text file")
    ap.add_argument("--model", default="facebook/m2m100_418M", help="HF model name (default: facebook/m2m100_418M)")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for translation (default: 16)")
    args = ap.parse_args()

    print("-> Extracting source (EN) text...")
    eng = extract_text_auto(args.src)
    print(f"   Source lines: {len(eng)}")

    print("-> Extracting reference (BG) text...")
    ref = extract_text_auto(args.ref)
    print(f"   Reference lines: {len(ref)}")

    # Normalize both sides
    eng = normalize_lines(eng)
    ref = normalize_lines(ref)
    print(f"   After normalization: EN={len(eng)} lines, BG_REF={len(ref)} lines")

    if len(eng) == 0:
        print("No English text found after normalization. Exiting.")
        sys.exit(1)

    print("-> Loading translation model...")
    tok, model, device = load_m2m_model(args.model)

    print("-> Translating EN -> BG ...")
    sys_out = translate_lines_en_bg(eng, tok, model, device, batch_size=args.batch)
    sys_out = normalize_lines(sys_out)
    print(f"   Produced {len(sys_out)} translated lines.")

    # Save system outputs as a plain text (one line per source)
    with open(args.save_system, "w", encoding="utf-8", newline="") as f:
        for line in sys_out:
            f.write(line + "\n")
    print(f"-> Saved system translations to: {args.save_system}")

    # Export side-by-side CSV for human review
    n = min(len(eng), len(sys_out), len(ref))
    if n == 0:
        print("Could not align any lines to export. Check your inputs.")
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["English", "System BG", "Reference BG"])
        for i in range(n):
            w.writerow([eng[i], sys_out[i], ref[i]])
    print(f"-> Side-by-side CSV saved to: {args.out} (rows: {n})")

    # Optional quick metrics (only meaningful if structures are similar)
    try_scores(sys_out, ref)

if __name__ == "__main__":
    main()
