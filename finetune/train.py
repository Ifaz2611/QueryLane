"""LoRA SFT for the local specialist model. Run on the AMD MI300X pod.

    pip install "transformers>=4.45" "trl>=0.12" "peft>=0.13" datasets accelerate
    python train.py --model Qwen/Qwen2.5-3B-Instruct --data data/train_all.jsonl \
        --out /workspace/ft-qwen3b

Then merge + convert with convert.sh. Checkpoints go under --out so an
interrupted run resumes with --resume.
"""

import argparse
import json

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_rows(path: str):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/train_all.jsonl")
    parser.add_argument("--out", default="/workspace/ft-qwen3b")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = Dataset.from_list(
        [
            {
                "text": tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
            }
            for row in rows
        ]
    )
    print(f"{len(dataset)} training rows from {args.data}")

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
    # TRL renamed max_seq_length -> max_length; accept either version.
    try:
        config = SFTConfig(max_seq_length=1024, **config_kwargs)
    except TypeError:
        config = SFTConfig(max_length=1024, **config_kwargs)
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out + "/final")
    tokenizer.save_pretrained(args.out + "/final")
    print("adapter saved to", args.out + "/final")


if __name__ == "__main__":
    main()
