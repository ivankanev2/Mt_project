# train_lora_seq2seq.py
import os, argparse, json, pathlib, math
import random
import numpy as np
import torch, wandb, sacrebleu
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import IntervalStrategy
from peft import LoraConfig, get_peft_model
from dataclasses import fields
from types import SimpleNamespace

# keep noisy envs in check
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_base(model_id: str, lora_opts: SimpleNamespace = None):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if not (lora_opts and lora_opts.disable):
        target_names = [t.strip() for t in (lora_opts.targets.split(",") if lora_opts else []) if t.strip()]
        if not target_names:
            target_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        all_names = ",".join([n for n, _ in model.named_modules()])
        targets = [t for t in target_names if f".{t}" in all_names or all_names.startswith(t) or f"_{t}" in all_names]
        if not targets:
            raise RuntimeError("No LoRA target modules matched the model. Inspect --lora_targets.")

        peft_cfg = LoraConfig(
            r=lora_opts.r if lora_opts else 16,
            lora_alpha=lora_opts.alpha if lora_opts else 32,
            lora_dropout=lora_opts.dropout if lora_opts else 0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, peft_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.4f}%)")
    if wandb.run:
        wandb.summary["trainable_params"] = int(trainable)

    return tok, model


def extract_src_ref(batch):
    srcs, refs = [], []
    for msgs in batch["messages"]:
        en = ""
        bg = ""
        for m in msgs:
            if m["role"] == "user" and not en:
                en = m["content"]
            if m["role"] == "assistant":
                bg = m["content"]
        srcs.append(en)
        refs.append(bg)
    return {"src": srcs, "ref": refs}


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning for seq2seq EN→BG models (e.g., OPUS).")
    ap.add_argument("--model_id", default="Helsinki-NLP/opus-mt-en-bg")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--eval_file", default="data/dev.jsonl")
    ap.add_argument("--output_dir", default="adapters/opus_en_bg_lora")
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)
    ap.add_argument("--gen_num_beams", type=int, default=1)
    ap.add_argument("--eval_steps", type=int, default=0,
                    help="Absolute evaluation interval in optimizer steps (overrides --evals_per_epoch when >0).")
    ap.add_argument("--evals_per_epoch", type=int, default=4,
                    help="If --eval_steps is 0, run this many evaluations per epoch (full dev set each time).")
    ap.add_argument("--save_steps", type=int, default=0,
                    help="Checkpoint interval; defaults to eval_steps when <=0.")
    ap.add_argument("--load_best_model", action="store_true",
                    help="Track BLEU and reload the best checkpoint at the end.")
    ap.add_argument("--lora_r", type=int, default=16,
                    help="LoRA rank (set higher for larger adapters).")
    ap.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha scaling factor.")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                    help="LoRA dropout probability.")
    ap.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Comma-separated module names to wrap (attention, FFN, heads).")
    ap.add_argument("--no_lora", action="store_true",
                    help="Disable LoRA entirely (eval base model / full fine-tune).")
    args = ap.parse_args()

    run_name = os.environ.get("WANDB_NAME", None)
    wandb_project = os.environ.get("WANDB_PROJECT", "mt_project_seq2seq")
    wandb.init(project=wandb_project, name=run_name, config=vars(args))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    lora_opts = SimpleNamespace(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=args.lora_targets,
        disable=args.no_lora,
    )
    tok, model = load_base(args.model_id, lora_opts=lora_opts)
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    wandb.watch(model, log="all", log_freq=20)

    ds_train = load_dataset("json", data_files=args.train_file)["train"]
    ds_eval = load_dataset("json", data_files=args.eval_file)["train"]

    ds_train = ds_train.map(extract_src_ref, batched=True)
    ds_eval = ds_eval.map(extract_src_ref, batched=True)

    keep_cols = {"messages", "src", "ref"}
    remove_cols_train = [c for c in ds_train.column_names if c not in keep_cols]
    remove_cols_eval = [c for c in ds_eval.column_names if c not in keep_cols]

    def tokenize_fn(batch):
        model_inputs = tok(
            batch["src"],
            max_length=args.max_source_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        target_tok = tok(
            text_target=batch["ref"],
            max_length=args.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        labels = target_tok["input_ids"]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]
        model_inputs["labels"] = labels
        model_inputs["src"] = batch["src"]
        model_inputs["ref"] = batch["ref"]
        return model_inputs

    ds_train_tok = ds_train.map(tokenize_fn, batched=True, remove_columns=remove_cols_train)
    ds_eval_tok = ds_eval.map(tokenize_fn, batched=True, remove_columns=remove_cols_eval)

    def calc_steps_per_epoch(num_examples, micro_batch, grad_accum):
        denom = max(1, micro_batch * grad_accum)
        return max(1, math.ceil(num_examples / denom))

    steps_per_epoch = calc_steps_per_epoch(len(ds_train_tok), args.batch_size, args.grad_accum)
    if args.eval_steps > 0:
        eval_steps = args.eval_steps
    else:
        evals_per_epoch = max(1, args.evals_per_epoch)
        eval_steps = max(1, math.ceil(steps_per_epoch / evals_per_epoch))

    arg_field_names = {f.name for f in fields(Seq2SeqTrainingArguments)}
    save_steps = args.save_steps if args.save_steps and args.save_steps > 0 else eval_steps

    def accepts(name):
        return name in arg_field_names

    requested_args = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "label_smoothing_factor": args.label_smoothing,
        "logging_steps": 20,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": args.gen_max_new_tokens,
        "generation_num_beams": args.gen_num_beams,
        "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        "fp16": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,
        "report_to": "wandb",
        "run_name": run_name,
        "metric_for_best_model": "bleu",
        "greater_is_better": True,
    }

    def set_interval_strategy(field_names):
        set_any = False
        for name in field_names:
            if accepts(name):
                requested_args[name] = IntervalStrategy.STEPS
                set_any = True
        return set_any

    set_interval_strategy(["logging_strategy"])
    eval_strategy_set = set_interval_strategy(["evaluation_strategy", "eval_strategy"])
    if not eval_strategy_set and accepts("evaluate_during_training"):
        requested_args["evaluate_during_training"] = True
    save_strategy_set = set_interval_strategy(["save_strategy"])
    if accepts("load_best_model_at_end"):
        if args.load_best_model and eval_strategy_set and save_strategy_set:
            requested_args["load_best_model_at_end"] = True
        else:
            requested_args["load_best_model_at_end"] = False

    compatible_args = {k: v for k, v in requested_args.items() if accepts(k)}
    training_args = Seq2SeqTrainingArguments(**compatible_args)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels == -100, tok.pad_token_id, labels)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip().lower() for p in decoded_preds]
        decoded_labels = [l.strip().lower() for l in decoded_labels]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {"bleu": bleu.score}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

    art = wandb.Artifact(f"{wandb.run.name or 'seq2seq'}-adapter", type="model")
    art.add_dir(args.output_dir)
    wandb.log_artifact(art)
    wandb.finish()


if __name__ == "__main__":
    main()
