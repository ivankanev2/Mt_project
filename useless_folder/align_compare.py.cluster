import argparse
import csv
from pathlib import Path
import sys
import numpy as np

# --- ref text extraction (docx/pdf) ---
def extract_plain_text_docx(path):
    from docx import Document
    doc = Document(path)
    lines = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)
    return lines

def extract_plain_text_pdf(path):
    from pypdf import PdfReader
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
        raise ValueError(f"Unsupported reference type: {ext} (use .docx or .pdf)")

def normalize_lines(lines):
    out = []
    for s in lines:
        s = " ".join(s.split()).strip()
        if s:
            out.append(s)
    return out

def embed_lines(lines, model):
    return model.encode(lines, convert_to_numpy=True, normalize_embeddings=True)

def main():
    ap = argparse.ArgumentParser(description="Align system BG lines to reference BG and export aligned CSV + metrics.")
    ap.add_argument("--sys", required=True, help="Path to system output (one BG line per source line), e.g., system.out")
    ap.add_argument("--ref", required=True, help="Bulgarian reference file (.docx or .pdf)")
    ap.add_argument("--out", default="aligned_compare.csv", help="Output CSV with aligned rows")
    ap.add_argument("--window", type=int, default=6, help="Search window size in reference for monotonic alignment")
    ap.add_argument("--threshold", type=float, default=0.45, help="Minimum cosine similarity to accept a match")
    args = ap.parse_args()

    # Load files
    sys_lines = [l.rstrip("\n") for l in open(args.sys, encoding="utf-8")]
    ref_lines = extract_text_auto(args.ref)

    sys_lines = normalize_lines(sys_lines)
    ref_lines = normalize_lines(ref_lines)

    if not sys_lines or not ref_lines:
        print("Empty inputs after normalization. Check your files.", file=sys.stderr)
        sys.exit(1)

    print(f"System lines: {len(sys_lines)} | Reference lines: {len(ref_lines)}")

    # Load multilingual sentence embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sys_emb = embed_lines(sys_lines, model)
    ref_emb = embed_lines(ref_lines, model)

    # Monotonic, windowed greedy alignment
    aligned = []
    j = 0
    for i, sys_vec in enumerate(sys_emb):
        start = j
        end = min(len(ref_emb), j + args.window)
        if start >= end:
            break
        window = ref_emb[start:end]
        sims = np.dot(window, sys_vec)
        k = int(np.argmax(sims))
        best_j = start + k
        best_sim = float(sims[k])
        if best_sim >= args.threshold:
            aligned.append((i, best_j, sys_lines[i], ref_lines[best_j], best_sim))
            j = best_j + 1  # move forward (monotonic)
        else:
            # no good match; still record with empty ref to inspect later
            aligned.append((i, None, sys_lines[i], "", best_sim))

    # Write aligned CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["System BG (model)", "Reference BG", "Similarity", "sys_idx", "ref_idx"])
        for (i, jj, sys_txt, ref_txt, sim) in aligned:
            w.writerow([sys_txt, ref_txt, f"{sim:.3f}", i, "" if jj is None else jj])

    print(f"Aligned CSV written to: {args.out} (rows: {len(aligned)})")

    # Optional quick metrics on the confidently-aligned subset
    try:
        import sacrebleu
        paired_sys = [s for (_, jj, s, r, sim) in aligned if jj is not None and r]
        paired_ref = [[r for (_, jj, s, r, sim) in aligned if jj is not None and r]]
        if len(paired_sys) >= 10:
            bleu = sacrebleu.corpus_bleu(paired_sys, paired_ref).score
            chrf = sacrebleu.corpus_chrf(paired_sys, paired_ref).score
            print(f"Metrics on confidently aligned subset (n={len(paired_sys)}):")
            print(f"  BLEU: {bleu:.2f}")
            print(f"  chrF: {chrf:.2f}")
        else:
            print("Not enough confident alignments for reliable metrics (need ~10+).")
    except ImportError:
        print("Tip: pip install sacrebleu to see BLEU/chrF on aligned subset.")

if __name__ == "__main__":
    main()
