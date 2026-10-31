# QueryLane "Cost-aware LLM query routing."

**Cost-aware LLM query routing.** QueryLane sends every task down the cheapest
lane that can answer it correctly: deterministic rules first, zero-token local
specialists second, a bundled small language model third, and a cloud model only
when the cheaper lanes are unavailable or uncertain.

The project started as a Track 1 submission where the score was the number of
Fireworks tokens used, subject to an 80% accuracy gate. The routing ideas are
general: classify the request, spend nothing when a specialist is confident, and
escalate on doubt rather than risk a wrong answer.

## Why it exists

Cloud LLM calls are the dominant cost in a task pipeline, but most benchmark
tasks are not equally hard. A sentiment label, a named-entity list, and a
grade-school word problem do not need a frontier model every time. QueryLane
turns that observation into an explicit routing policy:

| Lane | Cost | Used for |
|---|---|---|
| Rules | 0 tokens | Category classification, instruction stripping, answer normalization |
| ONNX specialists | 0 tokens | Confident sentiment classification (SST-2 + cue gates) |
| Local GGUF model | 0 tokens | Fine-tuned factual/math/NER answers when the host is fast enough |
| Cloud model | billed | Everything uncertain, plus code, logic, and summarization |

The accuracy gate is protected by **escalation**: a cheap lane that is not
confident returns an empty answer, and the task falls through to cloud.

## How routing works

`agent/router.py` classifies a prompt into one of eight categories using lexical
cues only:

`factual`, `math`, `sentiment`, `summarization`, `ner`, `code_debug`, `logic`,
`code_gen`.

The check order matters: instruction verbs (extract entities, fix this bug,
write a function, summarize) are tested before cues that can appear inside a
quoted passage. Unknown prompts fall through to `factual`, which is the safest
cloud-routed bucket.

`agent/main.py` then applies the lanes in order:

1. **Specialist** — sentiment (and optionally NER) via ONNX. Self-gating; returns
   `""` on hedges, contrast, low confidence, or shaky spans.
2. **Local model** — only for categories listed in `LOCAL_CATEGORIES`, and only
   after warm-up calibration proves the host clears the category's speed bar.
3. **Cloud** — `agent/cloud.py` uses terse per-category system prompts, disables
   reasoning where possible, and falls back across allowed models.

Math has an extra path: the cloud model may return one `EXPR: ... | UNIT: ...`
line, which `agent/pal.py` evaluates locally with an AST whitelist. A parse or
safety failure falls back to step-by-step generation.

## Quick start

### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

For the local specialists, add the extra:

```bash
pip install -e ".[specialists]"
```

For local GGUF inference, also install `llama-cpp-python` (CPU wheel shown):

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Configure

Copy `.env.example` to `.env` and fill in the values. The harness injects real
credentials at evaluation time; never commit them.

```bash
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
```

### 3. Run a batch

```bash
python -m agent.main
```

The default contract is:

- read `/input/tasks.json`: `[{"task_id": "...", "prompt": "..."}, ...]`
- write `/output/results.json`: `[{"task_id": "...", "answer": "..."}, ...]`
- exit 0 on success

For local experiments, override the paths:

```bash
# PowerShell
$env:TASKS_PATH="testio/input_dev/tasks.json"
$env:RESULTS_PATH="testio/output_dev/results.json"
python -m agent.main
```

The `querylane` console script is equivalent once the package is installed.

### 4. Evaluate against the dev set

```bash
python -m eval.judge
python -m eval.judge --category math
python -m eval.judge --answers eval/results_dev.json   # re-judge saved answers
```

The judge is a dev-only Fireworks call. Judge tokens are tracked separately and
are not part of the routed agent's token score.

### 5. Measure specialist precision (offline)

```bash
python -m eval.spec_precision
```

This replays the validated synthetic datasets through the ONNX specialists and
reports coverage and precision. No API calls are made.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TASKS_PATH` | `/input/tasks.json` | Input batch |
| `RESULTS_PATH` | `/output/results.json` | Output batch |
| `AGENT_WORKERS` | `4` | Thread pool size |
| `AGENT_DEADLINE` | `480` | Seconds before the agent stops collecting answers |
| `LOCAL_CUTOFF` | `150` | Do not start local work with less time than this |
| `LOCAL_CATEGORIES` | empty | Comma-separated categories allowed to use the local model |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | GGUF weights |
| `LOCAL_MIN_TPS` | `0.4` | Global local-inference speed floor |
| `LOCAL_TIME_BUDGET` | `300` | Total seconds of local inference |
| `LOCAL_THREADS` | `2` | llama.cpp threads |
| `NER_MODEL_DIR` | `/app/models/ner` | ONNX NER directory |
| `SST2_MODEL_DIR` | `/app/models/sst2` | ONNX sentiment directory |
| `NER_SPECIALIST` | `off` | Enable the experimental ONNX NER specialist |
| `AGENT_DEBUG` | unset | Log per-call token usage |

## Repository layout

```text
agent/            runtime: router, prompts, cloud client, local model, specialists, PAL
eval/             dev-only judge, local benchmark, specialist precision harness
finetune/         synthetic data generator, LoRA training, GGUF conversion, notebooks
models/           ONNX configs/tokenizers (weights are gitignored)
testio/           local input batches for practice/dev/stress runs
Dockerfile*       cloud-only, local, phase-2, and versioned hybrid images
SCORES.md         historical evaluation log and token/accuracy findings
TODO.md           next steps and known limitations
```

## Docker

The default `Dockerfile` is the smallest cloud-only image. It installs only
`requirements.txt` and copies `agent/`:

```bash
docker build -t querylane .
docker run --rm -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  -v "$PWD/testio/input:/input:ro" -v "$PWD/testio/output:/output" querylane
```

Hybrid images (`Dockerfile.phase2`, `Dockerfile.v10`, `Dockerfile.v13`,
`Dockerfile.v14`, `Dockerfile.v15`) bundle ONNX specialists and/or GGUF weights.
The weights are intentionally not in the repository; see
`finetune/POD_INSTRUCTIONS.md` for the build-and-convert flow.

## Design notes

- **Terse system prompts.** Every system token is billed on every call.
- **No content-based retries.** Only transient-error retries and model fallback;
  a second attempt costs tokens.
- **Precision over coverage.** A specialist that answers 38% of tasks at 100%
  precision is better than one that answers 80% at 90% precision.
- **Escalation is cheap.** Paying ~100 tokens for a cloud answer beats an
  accuracy-gate miss.
- **Calibrate, then decide.** Local inference is enabled per category from a
  measured tokens-per-second rate, not from an assumption about the host.

## Status

QueryLane is a research/competition codebase, not a hosted service. The
historical score log, including the token/accuracy trade-offs that shaped the
router, is in [`SCORES.md`](SCORES.md). Known gaps and planned work are in
[`TODO.md`](TODO.md).
