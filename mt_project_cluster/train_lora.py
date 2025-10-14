# train_lora.py
import os, argparse, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_device_setup():
    if torch.cuda.is_available():
        return "cuda", True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", False  # no bitsandbytes on MPS
    return "cpu", False

def load_base(model_id: str, use_4bit: bool):
    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # base model (4-bit if available)
    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    # LoRA on common projection modules
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)
    return tok, model

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
    ap.add_argument(
        "--grad_accum", "--gradient_accumulation_steps",
        type=int, default=8, dest="grad_accum",
        help="Gradient accumulation steps"
    )
    args = ap.parse_args()

    device, can_4bit = get_device_setup()
    tok, model = load_base(args.model_id, use_4bit=can_4bit)

    # Load JSONL datasets that contain {"messages": [{role, content}, ...]}
    ds_train = load_dataset("json", data_files=args.train_file)["train"]
    ds_eval  = load_dataset("json", data_files=args.eval_file)["train"]

    # Convert chat messages -> text with the model’s chat template
    def to_text(batch):
        texts = []
        for msgs in batch["messages"]:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(txt)
        return {"text": texts}

    ds_train = ds_train.map(to_text, batched=True, remove_columns=ds_train.column_names)
    ds_eval  = ds_eval.map(to_text,  batched=True, remove_columns=ds_eval.column_names)

    # Tokenize to fixed length & create labels
    def tok_map(batch):
        enc = tok(
            batch["text"],
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_train_tok = ds_train.map(tok_map, batched=True, remove_columns=ds_train.column_names)
    ds_eval_tok  = ds_eval.map(tok_map,  batched=True, remove_columns=ds_eval.column_names)

    # Data collator for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Mixed precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        logging_steps=20,
        save_steps=500,
        save_total_limit=2,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",

        bf16=use_bf16,
        fp16=use_fp16,

        report_to="none"
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    print(trainer.evaluate())

    # Save LoRA adapter + tokenizer
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()