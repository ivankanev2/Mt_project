# make_pairs.py
import json, argparse, random, os, re
from pathlib import Path
from lxml import etree
from batch_eval import extract_text_auto, normalize_lines, drop_furniture, dp_align

# --- paste your extract_plain_text_docx, extract_plain_text_pdf, extract_text_xml, extract_text_auto here
# You can copy them directly from your current batch_eval.py (the robust versions you ended up with).
# --- paste your dp_align() here as well (the global aligner you used for evaluation)

def normalize_lines(lines):
    out = []
    for s in lines:
        s = " ".join(s.split()).strip()
        if s:
            out.append(s)
    return out

def drop_furniture(lines):
    out = []
    for s in lines:
        t = (s or "").strip()
        if t:
            out.append(t)
    return out

def build_pairs(en_lines, bg_lines, sim_thr=0.50):
    # Align EN→BG with DP; we want only matched pairs (skip gaps)
    pairs = []
    # For training we align **reference** BG directly to EN (no system output)
    # So we just reuse dp_align with "sys_lines" = EN and "ref_lines" = BG:
    dp = dp_align(en_lines, bg_lines, sim_thr=sim_thr)
    for i, j in dp:
        if j is None:
            continue
        en = en_lines[i].strip()
        bg = bg_lines[j].strip()
        if en and bg:
            pairs.append((en, bg))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="Space-separated list of EN,BG file pairs. Example: data/eng1.docx,data/bg1.docx data/eng2.docx,data/bg2.docx")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for item in args.pairs:
        en_path, bg_path = item.split(",")
        en_raw = extract_text_auto(en_path)
        bg_raw = extract_text_auto(bg_path)
        en = normalize_lines(drop_furniture(en_raw))
        bg = normalize_lines(drop_furniture(bg_raw))
        print(f"Loaded: {en_path} ({len(en)} lines) | {bg_path} ({len(bg)} lines)")
        pairs = build_pairs(en, bg, sim_thr=args.thr)
        print(f"Aligned {len(pairs)} EN↔BG pairs")
        all_pairs.extend(pairs)

    random.seed(42)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_dev = max(100, int(n * args.dev_ratio)) if n > 1000 else max(50, int(n * args.dev_ratio))
    dev = all_pairs[:n_dev]
    train = all_pairs[n_dev:]

    def to_chat(examples):
        # TRL can consume chat messages
        return {
            "messages": [
                {"role": "system", "content": "You are a professional technical translator. Translate English to Bulgarian faithfully, preserving numbers and units. Output only the translation."},
                {"role": "user", "content": examples[0]},
                {"role": "assistant", "content": examples[1]}
            ]
        }

    train_path = Path(args.outdir) / "train.jsonl"
    dev_path = Path(args.outdir) / "dev.jsonl"
    with open(train_path, "w", encoding="utf-8") as ft:
        for en, bg in train:
            ft.write(json.dumps(to_chat((en, bg)), ensure_ascii=False) + "\n")
    with open(dev_path, "w", encoding="utf-8") as fd:
        for en, bg in dev:
            fd.write(json.dumps(to_chat((en, bg)), ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} training pairs to {train_path}")
    print(f"Wrote {len(dev)} dev pairs to {dev_path}")

if __name__ == "__main__":
    main()