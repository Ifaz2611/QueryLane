"""LoRA SFT for the local specialist model. Run on the AMD MI300X pod.

    pip install "transformers>=4.45" "trl>=0.12" "peft>=0.13" datasets accelerate
    python train.py --model Qwen/Qwen2.5-3B-Instruct --data data/train_dedup.jsonl \
        --out /workspace/ft-qwen3b --val-frac 0.05

A deterministic, category-stratified, leakage-safe validation split
(--val-frac, or --val-data for an explicit file) is held out of training.
Each epoch the trainer reports its own eval loss; after training,
finetune.eval_split reports PER-CATEGORY held-out loss (assistant-token
masked) plus deterministic, judge-free accuracy metrics (label accuracy,
span F1, constraint compliance, expression-value accuracy, token F1), which
replaces the noisy single-judge evaluation. Results land in
<out>/val_metrics.json; a provenance manifest lands in
<out>/train_manifest.json. Then merge + convert with convert.sh. Checkpoints
go under --out so an interrupted run resumes with --resume.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version as _package_version

try:
    from finetune.eval_split import evaluate_model, stratified_split
    from finetune.validate_data import load_jsonl
except ImportError:  # flat layout on the pod: `python train.py` with sibling modules
    from eval_split import evaluate_model, stratified_split
    from validate_data import load_jsonl

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_rows(path: str):
    rows, issues = load_jsonl(path)
    for issue in issues:
        print(f"warning: {path}: {issue}", file=sys.stderr)
    return rows


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(out_dir: str, args, data_path: str, n_rows: int, n_train: int, n_val: int) -> None:
    """Provenance manifest: exactly what produced this adapter, and from what."""
    arg_names = (
        "model", "data", "out", "epochs", "lr", "seed",
        "val_frac", "val_data", "max_new_tokens", "no_thinking", "skip_eval",
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "data": {"path": data_path, "sha256": _sha256(data_path), "rows": n_rows, "train": n_train, "val": n_val},
        "args": {key: getattr(args, key) for key in arg_names},
        "packages": {
            name: _package_version(name)
            for name in ("transformers", "trl", "peft", "datasets", "torch", "accelerate")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    path = os.path.join(out_dir, "train_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", path)


def _as_text_dataset(rows, tokenizer) -> Dataset:
    return Dataset.from_list(
        [
            {
                "text": tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
            }
            for row in rows
        ]
    )


def _build_sft_config(eval_dataset, config_kwargs: dict) -> "SFTConfig":
    """SFTConfig across transformers/trl versions: max_seq_length was renamed
    to max_length, evaluation_strategy to eval_strategy. Try the combinations."""
    for length_name in ("max_seq_length", "max_length"):
        for eval_name in ("eval_strategy", "evaluation_strategy", None):
            kwargs = dict(config_kwargs)
            kwargs[length_name] = 1024
            if eval_dataset is not None and eval_name is not None:
                kwargs[eval_name] = "epoch"
                kwargs["per_device_eval_batch_size"] = 8
            try:
                return SFTConfig(**kwargs)
            except TypeError:
                continue
    raise RuntimeError(
        "could not build SFTConfig for this transformers/trl version; "
        "check the max_seq_length/max_length and eval_strategy argument names"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/train_all.jsonl")
    parser.add_argument("--out", default="/workspace/ft-qwen3b")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42, help="split + trainer seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="fraction held out for validation (deterministic, stratified)",
    )
    parser.add_argument(
        "--val-data",
        default="",
        help="explicit validation JSONL (overrides --val-frac; can be the same file to eval on train rows)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="generation budget for the accuracy pass")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="pass enable_thinking=False to chat templates that support it (Qwen3 family)",
    )
    parser.add_argument("--skip-eval", action="store_true", help="skip the post-training metric pass")
    args = parser.parse_args()

    rows = load_rows(args.data)
    if args.val_data:
        val_rows = load_rows(args.val_data)
        train_rows = rows
    else:
        train_rows, val_rows = stratified_split(rows, val_frac=args.val_frac, seed=args.seed)
    print(f"{len(train_rows)} training rows / {len(val_rows)} validation rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = _as_text_dataset(train_rows, tokenizer)
    eval_dataset = _as_text_dataset(val_rows, tokenizer) if val_rows else None

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    config_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        packing=False,
        report_to=[],
    )
    config = _build_sft_config(eval_dataset, config_kwargs)
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out + "/final")
    tokenizer.save_pretrained(args.out + "/final")
    print("adapter saved to", args.out + "/final")

    write_manifest(args.out, args, args.data, n_rows=len(rows), n_train=len(train_rows), n_val=len(val_rows))

    if val_rows and not args.skip_eval:
        print("\nper-category held-out metrics (deterministic, judge-free):")
        evaluate_model(
            trainer.model,
            tokenizer,
            val_rows,
            out_path=os.path.join(args.out, "val_metrics.json"),
            max_new_tokens=args.max_new_tokens,
            no_thinking=args.no_thinking,
        )
    elif args.skip_eval:
        print("post-training metric pass skipped (--skip-eval)")
    elif not val_rows:
        print("no validation rows (--val-frac 0); skipping the metric pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())

