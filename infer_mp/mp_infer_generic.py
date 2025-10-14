import argparse, os
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#this is for any casual-LM model
def read_lines(path: str):
    p = Path(path)
    if p.suffix.lower() == ".docx":
        # requires: pip install python-docx
        from docx import Document
        doc = Document(str(p))
        lines = []
        for para in doc.paragraphs:
            t = (para.text or "").strip()
            if t:
                lines.append(t)
        return lines
    else:
        return [l.strip() for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]

def batcher(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def translate_batch(model, tok, prompts, max_new_tokens=256):
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    # strip original prompt if the model echoes it
    cleaned = []
    for src, full in zip(prompts, decoded):
        if full.startswith(src):
            cleaned.append(full[len(src):].strip())
        else:
            cleaned.append(full.strip())
    return cleaned

def make_prompts(tok, lines, sys_prompt=None):
    prompts = []
    use_chat = hasattr(tok, "apply_chat_template")
    for s in lines:
        if use_chat:
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": s})
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(txt)
        else:
            # fallback single-turn “Translate:” style
            header = (sys_prompt + "\n") if sys_prompt else ""
            prompts.append(f"{header}Translate to Bulgarian:\n{s}\nAnswer:")
    return prompts

def worker(gid, model_id, shard, outq, max_new_tokens, batch_size, sys_prompt):
    torch.cuda.set_device(gid)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map={"": gid},
    )

    outputs = []
    prompts = make_prompts(tok, shard, sys_prompt)
    for chunk in batcher(prompts, batch_size):
        outputs.extend(translate_batch(model, tok, chunk, max_new_tokens))
    outq.put(outputs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--src_doc", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--system_prompt", default="Translate English to Bulgarian as a professional technical translator.")
    args = ap.parse_args()

    set_start_method("spawn", force=True)

    lines = read_lines(args.src_doc)
    if len(lines) == 0:
        raise SystemExit("No lines found in src_doc.")

    # split work across GPUs
    n = max(1, args.gpus)
    shards = [lines[i::n] for i in range(n)]
    q = Queue()
    procs = []
    for gid in range(n):
        p = Process(target=worker, args=(
            gid, args.model_id, shards[gid], q,
            args.max_new_tokens, args.batch, args.system_prompt
        ))
        p.start()
        procs.append(p)

    merged = []
    for _ in range(n):
        out = q.get()
        merged.extend(out)

    for p in procs:
        p.join()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for t in merged:
            f.write(t + "\n")

if __name__ == "__main__":
    main()
