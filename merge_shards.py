import argparse, glob
from pathlib import Path
from batch_eval import extract_text_auto, drop_furniture, normalize_lines, save_table_csv_and_xlsx, score_bleu_chrf, dp_align

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)

    # Concatenate shards in rank order
    shard_files = sorted(glob.glob(str(outdir / "sys_shard_*.out")))
    sys_bg = []
    for p in shard_files:
        with open(p, encoding="utf-8-sig") as f:
            sys_bg.extend([x.rstrip("\n") for x in f])

    # Reconstruct the same outputs batch_eval makes:
    src = "data/english_ref_1.docx"
    ref = "data/bulgarian_ref_1.docx"
    eng = normalize_lines(drop_furniture(extract_text_auto(src)))
    bul = normalize_lines(drop_furniture(extract_text_auto(ref)))

    # Align + write CSV/XLSX
    pairs = dp_align(sys_bg, bul, sim_thr=0.50)
    rows = []
    for i, j in pairs:
        en_txt  = eng[i]
        sys_txt = sys_bg[i]
        ref_txt = bul[j] if j is not None else ""
        rows.append([en_txt, sys_txt, ref_txt, i, ("" if j is None else j), "dp"])

    cols = ["English","System BG","Reference BG","en_idx","bg_idx","segment"]
    compare_path = outdir / "compare_anchored_qwen_multi.csv"
    save_table_csv_and_xlsx(str(compare_path), cols, rows)

    # Metrics
    sys_subset, ref_subset = [], []
    for _, s, r, *_ in rows:
        if r:
            sys_subset.append(s); ref_subset.append(r)
    bleu, chrf = score_bleu_chrf(sys_subset, ref_subset)
    print(f"Aligned subset n={len(sys_subset)} BLEU={bleu} chrF={chrf}")
    print(f"Wrote: {compare_path}")

if __name__ == "__main__":
    main()
