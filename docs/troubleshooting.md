# Troubleshooting

## Missing API key / bad cloud configuration

Symptom: the agent exits immediately with
`bad cloud configuration: ...` before answering anything.

`CloudClient.from_env()` requires all of:

- `FIREWORKS_API_KEY` — injected by the harness; never commit it.
- `FIREWORKS_BASE_URL` — the OpenAI-compatible endpoint.
- `ALLOWED_MODELS` — comma-separated model ids; at least one required, and
  every fallback must be listed here.

Fix: `set -a; source .env; set +a` (or pass `-e FIREWORKS_API_KEY` to
`docker run`). Verify with `python -c "from agent.cloud import CloudClient;
CloudClient.from_env()"`.

Per-call failures are retried only for transient errors (connection, timeout,
5xx, 429). A `400` mentioning `reasoning_effort` is retried once without it,
then the next allowed model is tried. Tasks that still fail get `""` so the
output schema stays valid — check stderr for `[task_id] cloud failed:` lines.

## Missing model files (local lanes silently off)

The local and specialist lanes degrade gracefully: if weights are absent they
return `""` and tasks escalate to cloud, so the run still succeeds — it just
costs tokens. Symptoms: `routed` is always `cloud`/`cloud-fallback` in
`AGENT_LOG_JSON=1` output, or `--dry-run` predicts `cloud` for everything.

- **ONNX specialists** (`models/ner/model.onnx`, `models/sst2/model.onnx`):
  `models/` is git-ignored. Download and verify checksums with
  `python scripts/download_models.py onnx` (or `make models-onnx`).
  `specialists.py` must stay import-safe without numpy/onnxruntime —
  install `pip install -e ".[specialists]"` if `available("sst2")` is false.
  Disable NER explicitly with `NER_SPECIALIST=off`.
- **Local GGUF** (`LOCAL_MODEL_PATH`, default `/app/models/model.gguf`):
  download with `python scripts/download_models.py gguf` and install
  `llama-cpp-python` (`pip install llama-cpp-python --extra-index-url
  https://abetlen.github.io/llama-cpp-python/whl/cpu`). The lane also requires
  `LOCAL_CATEGORIES` to list the category (default `""` = cloud-only) and the
  host to clear the per-category TPS bar after warmup.
- **Prompt too long**: prompts exceeding `LOCAL_N_CTX` (default 2048) minus
  system/few-shot/output budget are routed to cloud before local is tried —
  this is intentional, not a bug.

Use `python -m agent.main --explain "<prompt>"` (or `querylane-explain`) to see
exactly which gate sent a task to cloud.

## Slow host / local lane never engages

Warmup measures generation speed once; each category has a minimum-TPS bar
(`agent/local.py::_CATEGORY_MIN_TPS`): math 0.4, ner 0.9, sentiment 0.9,
factual 1.3 tok/s; summarization and logic are pinned at 99.0 (never local).
If the host is slower, the category silently stays on cloud.

- Check the warmup TPS in logs / `METRICS_PATH` output.
- Raise `LOCAL_MIN_TPS` is the wrong direction — the bar is per-category; a
  slow host should keep categories off local rather than burn the deadline.
- `LOCAL_TIME_BUDGET` (default 300s) caps total local generation time;
  `LOCAL_CUTOFF` (default 150s) stops starting new local tasks near
  `AGENT_DEADLINE` (480s). Lower `LOCAL_CUTOFF` on very slow hosts so cloud
  fallback still has time.
- Per-answer generation is capped at `min(_MAX_TOKENS, tps*75)` tokens so one
  slow generation cannot consume the whole budget; answers that fail
  `answer_ok()` escalate to cloud anyway.
- On CPU-only hosts, prefer the 1.5B GGUF over 3B, or keep `LOCAL_CATEGORIES`
  empty and accept cloud-only routing.

## Judge / eval issues

See [judge.md](judge.md) for grader modes, the exact judge prompt, and known
failure modes. Quick checks: `python -m eval.judge --grader deterministic`
re-grades saved answers offline; `--runs 5` measures verdict flips.
