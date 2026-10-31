# QueryLane: fine-tuning on the AMD pod

Everything below runs in a terminal inside your team's Jupyter pod. Total time
on an MI300X: roughly 15-30 minutes of compute.

## 1. Upload the training assets

Upload these files through the Jupyter file browser into `/workspace/ft/`:

- `finetune/train.py`
- `finetune/convert.sh`
- `finetune/data/train_all.jsonl` (generated locally; ~3,000 rows)

## 2. Install dependencies (once per pod restart)

```bash
pip install "transformers>=4.45" "trl>=0.12" "peft>=0.13" datasets accelerate
```

Torch with ROCm is already on the pod (vllm works there, per the FAQ).

## 3. Train (LoRA, ~minutes on MI300X)

```bash
cd /workspace/ft
python train.py --model Qwen/Qwen2.5-3B-Instruct --data train_all.jsonl --out /workspace/ft-qwen3b
```

Also worth training the small variant in parallel or after (faster on the
slow grader; we will pick by local evaluation):

```bash
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --data train_all.jsonl --out /workspace/ft-qwen15b
```

If your compute window ends mid-run, restart later with `--resume`
(checkpoints are saved each epoch under `--out`, which is on /workspace
persistent storage).

## 4. Merge + convert + quantize

```bash
bash convert.sh Qwen/Qwen2.5-3B-Instruct /workspace/ft-qwen3b/final /workspace/out3b
bash convert.sh Qwen/Qwen2.5-1.5B-Instruct /workspace/ft-qwen15b/final /workspace/out15b
```

## 5. Download the artifacts

Grab via the Jupyter file browser:

- `/workspace/out3b/ft-q4_k_m.gguf` (~2.0 GB)
- `/workspace/out15b/ft-q4_k_m.gguf` (~1.1 GB)

Put them in `models/` locally as `ft-3b-q4_k_m.gguf` and `ft-1.5b-q4_k_m.gguf`.
Back on the laptop, evaluate each model per category against the dev judge,
pick the winner, extend `LOCAL_CATEGORIES`, and build the next image tag.
