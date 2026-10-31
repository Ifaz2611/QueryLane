#!/usr/bin/env bash
# Merge the LoRA adapter, convert to GGUF, quantize to Q4_K_M.
# Run on the pod after train.py finishes.
#   bash convert.sh Qwen/Qwen2.5-3B-Instruct /workspace/ft-qwen3b/final /workspace/out
set -euo pipefail

BASE_MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
ADAPTER_DIR="${2:-/workspace/ft-qwen3b/final}"
OUT_DIR="${3:-/workspace/out}"
mkdir -p "$OUT_DIR"

python - "$BASE_MODEL" "$ADAPTER_DIR" "$OUT_DIR/merged" <<'PY'
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model, adapter_dir, merged_dir = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(merged_dir)
AutoTokenizer.from_pretrained(base_model).save_pretrained(merged_dir)
print("merged ->", merged_dir)
PY

if [ ! -d /workspace/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
  pip install -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=OFF -DGGML_HIP=OFF
  cmake --build /workspace/llama.cpp/build --target llama-quantize -j
fi

python /workspace/llama.cpp/convert_hf_to_gguf.py "$OUT_DIR/merged" \
  --outfile "$OUT_DIR/ft-f16.gguf" --outtype f16

/workspace/llama.cpp/build/bin/llama-quantize \
  "$OUT_DIR/ft-f16.gguf" "$OUT_DIR/ft-q4_k_m.gguf" Q4_K_M

ls -lh "$OUT_DIR"
echo "Download $OUT_DIR/ft-q4_k_m.gguf via the Jupyter file browser."
