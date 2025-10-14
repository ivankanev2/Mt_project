import os, argparse, math, json, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from batch_eval import extract_text_auto, normalize_lines, drop_furniture  # reuse helpers

#this is hardcoded for Qwen
def load_model(base_id, adapter, device):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map={"":device})
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return tok, model

def gen_batch(tok, model, lines, device, max_new_tokens=128, batch_size=32):
    out = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        if hasattr(tok, "apply_chat_template"):
            prompts = [tok.apply_chat_template(
                [{"role":"system","content":"Translate English to Bulgarian. Output ONLY the translation."},
                 {"role":"user","content":s}],
                tokenize=False, add_generation_prompt=True
            ) for s in batch]
        else:
            prompts = [f"Translate to Bulgarian:\n{s}\n" for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        for seq, inlen in zip(gen, [enc.input_ids.shape[1]]*len(batch)):
            text = tok.decode(seq[inlen:], skip_special_tokens=True).strip()
            out.append(text)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world_size", type=int, required=True)
    ap.add_argument("--local_rank", type=int, required=True)
    args = ap.parse_args()

    # Select GPU by local rank
    torch.cuda.set_device(args.local_rank)
    device = f"cuda:{args.local_rank}"

    # Load & split English lines evenly across world_size
    eng_raw = extract_text_auto(args.src)
    eng = normalize_lines(drop_furniture(eng_raw))
    n = len(eng)
    per = math.ceil(n / args.world_size)
    start = args.rank * per
    end = min(n, start + per)
    shard = eng[start:end]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    shard_out = Path(args.outdir)/f"sys_shard_{args.rank:02d}.out"

    tok, model = load_model(args.model_base, args.adapter, device)
    sys_bg = gen_batch(tok, model, shard, device)
    with open(shard_out, "w", encoding="utf-8-sig") as f:
        for t in sys_bg: f.write(t+"\n")

    # done; merging happens later

if __name__ == "__main__":
    main()
