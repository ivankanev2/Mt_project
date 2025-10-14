import argparse, csv, re, sys
from pathlib import Path

# ---------- Helpers ----------
def load_glossary(path):
    repl = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row["en"].strip().lower()
            bg = row["bg"].strip()
            if en and bg:
                repl.append((en, bg))
    # sort by descending length (replace longer phrases first)
    repl.sort(key=lambda x: len(x[0]), reverse=True)
    return repl

def enforce_glossary(bg_text, glossary_pairs):
    s = bg_text
    for en, bg in glossary_pairs:
        # Replace the EN phrase if it leaked into the BG output
        # Use case-insensitive, Unicode-aware word boundaries
        pattern = re.compile(rf"(?i)\b{re.escape(en)}\b")
        s = pattern.sub(bg, s)
    return " ".join(s.split())


def normalize_lines(lines):
    out = []
    for s in lines:
        s = " ".join(s.split())
        s = s.strip()
        if s:
            out.append(s)
    return out

REQ_EN = re.compile(r"\bRequirement\s+(\d+)\b", flags=re.IGNORECASE)
REQ_BG = re.compile(r"\bИзискване\s+(\d+)\b", flags=re.IGNORECASE)

def extract_plain_text_docx(path):
    from docx import Document
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        t = p.text.strip()
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
        raise ValueError(f"Unsupported type: {ext} (use .docx or .pdf)")

def drop_furniture(lines):
    # remove typical page furniture / TOC-number-only lines / copyright blocks
    cleaned = []
    for s in lines:
        if re.fullmatch(r"[\d\.\s]+", s):  # numbers/dots only (TOC page numbers)
            continue
        if len(s) <= 2:
            continue
        cleaned.append(s)
    return cleaned

def load_m2m(model_name, device=None):
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tok = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tok, model, device

def translate_batch_en_bg(lines, tok, model, device, batch_size=16):
    tok.src_lang = "en"
    out = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("bg"))
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend([d.strip() for d in dec])
    return out

# ---------- Anchor-aware alignment ----------
def anchor_map(lines, regex):
    m = {}
    for idx, s in enumerate(lines):
        mobj = regex.search(s)
        if mobj:
            n = int(mobj.group(1))
            m.setdefault(n, []).append(idx)
    return m  # number -> [indices]

def windowed_embed_align(src_chunk, ref_chunk, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", win=6, thr=0.5):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    src = normalize_lines(src_chunk)
    ref = normalize_lines(ref_chunk)
    if not src or not ref:
        return []

    st = SentenceTransformer(model_name)
    e_src = st.encode(src, convert_to_numpy=True, normalize_embeddings=True)
    e_ref = st.encode(ref, convert_to_numpy=True, normalize_embeddings=True)

    aligned = []
    j = 0
    for i, v in enumerate(e_src):
        start = j
        end = min(len(e_ref), j + win)
        if start >= end:
            aligned.append((i, None, src[i], ""))
            continue
        sims = e_ref[start:end] @ v
        k = int(sims.argmax())
        best_j = start + k
        sim = float(sims[k])
        if sim >= thr:
            aligned.append((i, best_j, src[i], ref[best_j]))
            j = best_j + 1
        else:
            aligned.append((i, None, src[i], ""))
    return aligned

def main():
    ap = argparse.ArgumentParser(description="Clean -> Translate -> Glossary -> Anchor-aware align -> CSV + metrics")
    ap.add_argument("--src", required=True, help="English DOCX/PDF")
    ap.add_argument("--ref", required=True, help="Bulgarian reference DOCX/PDF")
    ap.add_argument("--glossary", default="glossary.csv")
    ap.add_argument("--out", default="compare_anchored.csv")
    ap.add_argument("--system-out", default="system.out")
    ap.add_argument("--model", default="facebook/m2m100_418M")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    # 0) Read + clean
    eng = drop_furniture(extract_text_auto(args.src))
    bul = drop_furniture(extract_text_auto(args.ref))
    eng = normalize_lines(eng)
    bul = normalize_lines(bul)
    print(f"Cleaned: EN={len(eng)} BG_REF={len(bul)}")

    # 1) Translate EN -> BG
    tok, model, device = load_m2m(args.model)
    sys_bg = translate_batch_en_bg(eng, tok, model, device, batch_size=args.batch)

    # 2) Enforce glossary on the system output
    glossary_pairs = load_glossary(args.glossary) if Path(args.glossary).exists() else []
    if glossary_pairs:
        sys_bg = [enforce_glossary(t, glossary_pairs) for t in sys_bg]

    # Save raw system lines
    with open(args.system_out, "w", encoding="utf-8") as f:
        for t in sys_bg:
            f.write(t + "\n")
    print(f"Saved system output: {args.system_out}")

    # 3) Anchor by requirement numbers, then embed-align between anchors
    en_anchor = anchor_map(eng, REQ_EN)
    bg_anchor = anchor_map(bul, REQ_BG)

    rows = []
    anchor_numbers = sorted(set(en_anchor.keys()) & set(bg_anchor.keys()))
    last_e = last_b = 0

    for n in anchor_numbers:
        e_idx = en_anchor[n][0]        # first occurrence
        b_idx = bg_anchor[n][0]

        # align chunk before this anchor
        chunk_eng = eng[last_e:e_idx]
        chunk_sys = sys_bg[last_e:e_idx]
        chunk_bg  = bul[last_b:b_idx]

        if chunk_eng and chunk_bg:
            # align system BG to reference BG using embeddings on SYSTEM text
            aligned = windowed_embed_align(chunk_sys, chunk_bg, win=args.win, thr=args.thr)
            for (i_rel, j_rel, sys_txt, ref_txt) in aligned:
                e_abs = last_e + i_rel
                b_abs = (last_b + j_rel) if j_rel is not None else ""
                rows.append([eng[e_abs], sys_txt, ref_txt, e_abs, b_abs, "chunk"])
        # add the anchor pair itself (exact requirement number)
        rows.append([eng[e_idx], sys_bg[e_idx], bul[b_idx], e_idx, b_idx, f"anchor {n}"])

        last_e = e_idx + 1
        last_b = b_idx + 1

    # tail after the last anchor
    chunk_eng = eng[last_e:]
    chunk_sys = sys_bg[last_e:]
    chunk_bg  = bul[last_b:]
    if chunk_eng and chunk_bg:
        aligned = windowed_embed_align(chunk_sys, chunk_bg, win=args.win, thr=args.thr)
        for (i_rel, j_rel, sys_txt, ref_txt) in aligned:
            e_abs = last_e + i_rel
            b_abs = (last_b + j_rel) if j_rel is not None else ""
            rows.append([eng[e_abs], sys_txt, ref_txt, e_abs, b_abs, "tail"])

    # 4) Write CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["English", "System BG (glossary-enforced)", "Reference BG", "en_idx", "bg_idx", "segment"])
        w.writerows(rows)
    print(f"Anchored CSV written: {args.out} (rows: {len(rows)})")

    # 5) Metrics on aligned subset only
    try:
        import sacrebleu
        sys_subset, ref_subset = [], []
        for en_txt, sys_txt, ref_txt, e_i, b_i, seg in rows:
            if ref_txt:  # only pairs that found a ref
                sys_subset.append(sys_txt.lower().strip())
                ref_subset.append(ref_txt.lower().strip())
        if len(sys_subset) >= 10:
            bleu = sacrebleu.corpus_bleu(sys_subset, [ref_subset]).score
            chrf = sacrebleu.corpus_chrf(sys_subset, [ref_subset]).score
            print(f"Aligned-subset metrics: BLEU={bleu:.2f}  chrF={chrf:.2f}  (n={len(sys_subset)})")
        else:
            print("Not enough aligned pairs for metrics (need >=10).")
    except ImportError:
        print("Tip: pip install sacrebleu for BLEU/chrF.")

if __name__ == "__main__":
    main()
