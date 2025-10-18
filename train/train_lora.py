# train_lora.py
import os, argparse, json, random, pathlib
import torch, wandb, sacrebleu
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig, Trainer, TrainerCallback, default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ensure we don't accidentally run W&B in offline mode
os.environ.pop("WANDB_MODE", None)

# to make the warning shut the fuck up
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_device_setup():
    if torch.cuda.is_available():
        return "cuda", True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", False  # no bitsandbytes on MPS
    return "cpu", False


def load_base(model_id: str, use_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # important for QLoRA
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # pick LoRA target modules that actually exist
    wanted = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    all_names = ",".join([n for n, _ in model.named_modules()])
    targets = [t for t in wanted if f".{t}" in all_names or all_names.startswith(t) or f"_{t}" in all_names]
    if not targets:
        raise RuntimeError("No LoRA target modules matched the model. Inspect layer names and adjust target_modules.")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=targets
    )
    model = get_peft_model(model, peft_cfg)

    # summarize trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")
    if wandb.run:
        wandb.summary["trainable_params"] = int(trainable)

    return tok, model


# ---------- helpers used in .map() ----------
def extract_src_ref(batch):
    """Add 'src' (EN) and 'ref' (BG) alongside 'messages'."""
    srcs, refs = [], []
    for msgs in batch["messages"]:
        en = ""
        bg = ""
        for m in msgs:
            if m["role"] == "user" and not en:
                en = m["content"]
            if m["role"] == "assistant":
                bg = m["content"]  # keep the last assistant as reference
        srcs.append(en)
        refs.append(bg)
    return {"src": srcs, "ref": refs}


def to_text(batch, tok, add_generation_prompt=False):
    texts = []
    for msgs in batch["messages"]:
        txt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        texts.append(txt)
    return {"text": texts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--eval_file", default="data/dev.jsonl")
    ap.add_argument("--output_dir", default="adapters/qwen2_5_3b_bg_lora")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", "--gradient_accumulation_steps",
                    type=int, default=8, dest="grad_accum")
    ap.add_argument("--eval_sample_size", type=int, default=20)
    ap.add_argument("--gen_max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_do_sample", action="store_true")
    args = ap.parse_args()

    # ---- W&B init ----
    run_name = os.environ.get("WANDB_NAME", None)
    wandb_project = os.environ.get("WANDB_PROJECT", "mt_project")
    wandb.init(project=wandb_project, name=run_name, config={
        "model_id": args.model_id,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
    })
    wandb.define_metric("eval/*", step_metric="train/global_step")

    import random, numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


    device, can_4bit = get_device_setup()
    tok, model = load_base(args.model_id, use_4bit=can_4bit)
    model.generation_config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    wandb.watch(model, log="all", log_freq=20)

    # ---- load → add src/ref → build text ----
    ds_train = load_dataset("json", data_files=args.train_file)["train"]
    ds_eval  = load_dataset("json", data_files=args.eval_file)["train"]

    # add src/ref first
    ds_train = ds_train.map(extract_src_ref, batched=True)
    ds_eval  = ds_eval.map(extract_src_ref, batched=True)

    # then add chat-formatted 'text' and drop only raw 'messages'
    def _to_text_train(b): return to_text(b, tok, add_generation_prompt=False)
    ds_train = ds_train.map(_to_text_train, batched=True, remove_columns=["messages"])
    ds_eval  = ds_eval.map(_to_text_train, batched=True, remove_columns=["messages"])

    # now columns are: text, src, ref (plus any original metadata)

    # tokenize while keeping src/ref for callbacks
    keep_cols = {"text", "src", "ref"}
    def tok_map(batch):
        enc = tok(
            batch["text"],
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        labels = []
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for ids in enc["input_ids"]:
            # copy then mask pads
            lab = ids.copy()
            labels.append([(-100 if t == pad_id else t) for t in lab])
        enc["labels"] = labels
        enc["src"] = batch["src"]
        enc["ref"] = batch["ref"]
        return enc

    remove_cols_train = [c for c in ds_train.column_names if c not in keep_cols]
    remove_cols_eval  = [c for c in ds_eval.column_names  if c not in keep_cols]

    ds_train_tok = ds_train.map(tok_map, batched=True, remove_columns=remove_cols_train)
    ds_eval_tok  = ds_eval.map(tok_map,  batched=True, remove_columns=remove_cols_eval)

    collator = default_data_collator

    # precision flags
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,

        logging_strategy="steps",
        logging_steps=20,

        # keep eval simple/compatible: we’ll call trainer.evaluate() at the end
        save_steps=500,
        save_total_limit=2,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",

        bf16=use_bf16,
        fp16=use_fp16,

        report_to="wandb",
        run_name=run_name,
    )

    # ---- BLEU + examples callback ----
    class BleuAndExamplesCallback(TrainerCallback):
        def __init__(self, tok, eval_sample_size, gen_kwargs, run_dir):
            self.tok = tok
            self.eval_sample_size = eval_sample_size
            self.gen_kwargs = gen_kwargs
            self.run_dir = pathlib.Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.trainer = None  # will be set on train begin

        def on_train_begin(self, args, state, control, **kwargs):
            # capture the Trainer instance once it exists
            self.trainer = kwargs.get("trainer", None)
            return control

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            # fallback in case on_train_begin didn't fire for some reason
            if self.trainer is None:
                self.trainer = kwargs.get("trainer", None)
            if self.trainer is None:
                # nothing we can do without the dataset reference
                return control

            trainer = self.trainer
            ds = trainer.eval_dataset
            n = min(self.eval_sample_size, len(ds))
            idxs = list(range(n))

            srcs, refs, hyps = [], [], []
            # ✅ ensure evaluation mode (no dropout etc.)
            was_training = model.training
            model.eval()

            for i in idxs:
                src = ds[i]["src"]
                ref = ds[i]["ref"]
                messages = [
                    {"role": "system", "content": "You are a translation assistant. Translate English to Bulgarian."},
                    {"role": "user", "content": src},
                ]
                prompt = self.tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():  # use this for your PyTorch version
                    attn = torch.ones_like(prompt)
                    out = model.generate(
                        input_ids=prompt,
                        attention_mask=attn,
                        max_new_tokens=self.gen_kwargs.get("max_new_tokens", 256),
                        do_sample=self.gen_kwargs.get("do_sample", False),
                        temperature=self.gen_kwargs.get("temperature", 1.0),
                        top_p=self.gen_kwargs.get("top_p", 1.0),
                        num_beams=self.gen_kwargs.get("num_beams", 1),
                        pad_token_id=self.tok.eos_token_id or self.tok.pad_token_id,
                    )
                gen_ids = out[0][prompt.shape[-1]:]
                hyp = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
                srcs.append(src); refs.append(ref); hyps.append(hyp)

            bleu = sacrebleu.corpus_bleu(hyps, [refs])
            wandb.log({"eval/sacrebleu": bleu.score, "train/global_step": state.global_step}, step=state.global_step)

            table = wandb.Table(columns=["idx", "source_en", "reference_bg", "hypothesis_bg"])
            for i, (s, r, h) in enumerate(zip(srcs, refs, hyps)):
                table.add_data(i, s, r, h)
            wandb.log({"examples": table}, step=state.global_step)

            out_path = self.run_dir / f"examples_step{state.global_step}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for s, r, h in zip(srcs, refs, hyps):
                    f.write(json.dumps({"src": s, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")
            run_id = wandb.run.id if wandb.run else "no_run"
            art = wandb.Artifact(f"eval_examples-{run_id}-step{state.global_step}", type="dataset", metadata={"n": len(srcs)})
            art.add_file(str(out_path))
            wandb.log_artifact(art, aliases=[f"step-{state.global_step}", "latest"])
            if was_training:
                model.train()
            return control
        

    def compute_and_log_bleu(trainer, tok, eval_sample_size, gen_kwargs, run_dir):
        import json, pathlib, torch, sacrebleu, wandb
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        model = trainer.model
        ds = trainer.eval_dataset
        n = min(eval_sample_size, len(ds))
        idxs = list(range(n))

        srcs, refs, hyps = [], [], []

        was_training = model.training
        model.eval()
        for i in idxs:
            src = ds[i]["src"]
            ref = ds[i]["ref"]
            messages = [
                {"role": "system", "content": "You are a translation assistant. Translate English to Bulgarian."},
                {"role": "user", "content": src},
            ]
            prompt = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                attn = torch.ones_like(prompt)
                out = model.generate(
                    input_ids=prompt,
                    attention_mask=attn,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
                    do_sample=gen_kwargs.get("do_sample", False),
                    temperature=gen_kwargs.get("temperature", 1.0),
                    top_p=gen_kwargs.get("top_p", 1.0),
                    num_beams=gen_kwargs.get("num_beams", 1),
                    pad_token_id=tok.eos_token_id or tok.pad_token_id,
                )
            gen_ids = out[0][prompt.shape[-1]:]
            hyp = tok.decode(gen_ids, skip_special_tokens=True).strip()
            srcs.append(src); refs.append(ref); hyps.append(hyp)
        if was_training:
            model.train()

        bleu = sacrebleu.corpus_bleu(hyps, [refs])

        wandb.log(
        {"eval/sacrebleu": bleu.score, "train/global_step": trainer.state.global_step},
        step=trainer.state.global_step
        )
        wandb.summary["sacrebleu_final"] = bleu.score

        table = wandb.Table(columns=["idx", "source_en", "reference_bg", "hypothesis_bg"])
        for i, (s, r, h) in enumerate(zip(srcs, refs, hyps)):
            table.add_data(i, s, r, h)
        wandb.log({"eval/examples": table})

        # Save + artifact
        out_path = run_dir / "eval_examples_final.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s, r, h in zip(srcs, refs, hyps):
                f.write(json.dumps({"src": s, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")
        art = wandb.Artifact(f"eval_examples-{wandb.run.id}-final", type="dataset", metadata={"n": len(srcs)})
        art.add_file(str(out_path))
        wandb.log_artifact(art, aliases=["final", "latest"])


    class PeriodicBleuCallback(TrainerCallback):
        def __init__(self, tok, eval_dataset, gen_kwargs, run_dir, every_n_steps=100, eval_sample_size=20, seed=42):
            self.tok = tok
            self.gen_kwargs = gen_kwargs
            self.run_dir = pathlib.Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.every_n_steps = every_n_steps
            self.eval_sample_size = eval_sample_size

            # cache a small, fixed subset once
            rng = random.Random(seed)
            idxs = list(range(len(eval_dataset)))
            rng.shuffle(idxs)
            idxs = idxs[:eval_sample_size]
            # keep just the rows we need (src/ref)
            self.cached = [{"src": eval_dataset[i]["src"], "ref": eval_dataset[i]["ref"]} for i in idxs]

        @torch.no_grad()
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
                return control

            # quick BLEU on the cached subset
            was_training = model.training
            model.eval()

            srcs, refs, hyps = [], [], []
            device = next(model.parameters()).device
            pad_id = self.tok.pad_token_id or self.tok.eos_token_id

            for item in self.cached:
                src, ref = item["src"], item["ref"]
                messages = [
                    {"role": "system", "content": "You are a translation assistant. Translate English to Bulgarian."},
                    {"role": "user", "content": src},
                ]
                prompt = self.tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(device)
                # explicit attention mask (silences the warning)
                attn = torch.ones_like(prompt)

                out = model.generate(
                    input_ids=prompt,
                    attention_mask=attn,
                    max_new_tokens=self.gen_kwargs.get("max_new_tokens", 256),
                    do_sample=self.gen_kwargs.get("do_sample", False),
                    temperature=self.gen_kwargs.get("temperature", 1.0),
                    top_p=self.gen_kwargs.get("top_p", 1.0),
                    num_beams=self.gen_kwargs.get("num_beams", 1),
                    pad_token_id=pad_id,
                )
                gen_ids = out[0][prompt.shape[-1]:]
                hyp = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

                srcs.append(src); refs.append(ref); hyps.append(hyp)

            if was_training:
                model.train()

            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            # log with the trainer step so W&B draws a curve
            wandb.log(
            {"eval/sacrebleu": bleu, "train/global_step": state.global_step},
            step=state.global_step
            )

            return control

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        data_collator=collator,
        tokenizer=tok,
    )

    run_dir = os.environ.get("WANDB_DIR", os.getcwd())
    gen_kwargs = dict(max_new_tokens=args.gen_max_new_tokens, do_sample=args.gen_do_sample)

    trainer.add_callback(PeriodicBleuCallback(
    tok=tok,
    eval_dataset=ds_eval_tok,     # use tokenized set to get src/ref we preserved
    gen_kwargs=gen_kwargs,
    run_dir=run_dir,
    every_n_steps=50,
    eval_sample_size=args.eval_sample_size
))
    
    cb = BleuAndExamplesCallback(tok, args.eval_sample_size, gen_kwargs, run_dir)
    cb.trainer = trainer
    trainer.add_callback(cb)

    trainer.train()
    trainer.evaluate()
    compute_and_log_bleu(trainer, tok, args.eval_sample_size, gen_kwargs, run_dir)


    # save adapter + tokenizer and log as artifact
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    art = wandb.Artifact(f"{wandb.run.name}-adapter", type="model")
    art.add_dir(args.output_dir)
    wandb.log_artifact(art)

    wandb.finish()


if __name__ == "__main__":
    main()