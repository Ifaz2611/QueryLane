<div align="center">
  <img src="docs/assets/logo.svg" alt="QueryLane Logo" width="500"/>
  <p><strong>Cost-aware LLM query routing: zero-token specialists, local models, and cloud fallback.</strong></p>
</div>

QueryLane routes each query through a hierarchy of processing lanes, always
choosing the cheapest option that can produce a correct answer. Deterministic
rules and zero-token local specialists run first; cloud LLMs are used only when
nothing cheaper is confident enough.

Originally built as a competition submission scored on Fireworks token usage
under an 80% accuracy gate, the methodology is general-purpose: classify the
request, spend nothing when a specialist is confident, and escalate on doubt
rather than risk a wrong answer.

## Features

- **Zero-token specialists** — ONNX models for sentiment (SST-2) and NER that
  run entirely on CPU and self-gate on uncertainty
- **Lexical routing** — deterministic, zero-cost classification into eight
  categories with a confidence score
- **Escalate on doubt** — cheap lanes return empty when unsure and the task
  falls through to a more capable lane; accuracy is never traded for tokens
- **Optional local GGUF lane** — speed-calibrated per category before a single
  token is generated
- **Token-frugal cloud path** — terse per-category system prompts,
  `reasoning_effort="none"`, model fallback across `ALLOWED_MODELS`
- **Program-aided math** — cloud emits `EXPR: ... | UNIT: ...`, evaluated
  locally through an AST whitelist
- **Explainer & web demo** — CLI and zero-dependency UI for inspecting
  routing decisions

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/lanes.md](docs/lanes.md) | Lane decision flowchart, escalation state diagram, per-lane mechanics |
| [docs/judge.md](docs/judge.md) | Exact judge prompt, deterministic checks, known failure modes |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Missing models/keys, slow hosts, eval issues |
| [docs/adding-a-category.md](docs/adding-a-category.md) | End-to-end guide for a new category |
| [docs/security.md](docs/security.md) | Prompt-injection notes, why routing is lexical |

## Architecture

Lanes run in order of increasing cost:

| Lane | Cost | Used for |
|------|------|----------|
| **Rules** | 0 tokens | Category classification, instruction stripping, answer normalization |
| **PAL math** | ~15 tokens | Single-quantity math via cloud `EXPR:` + local AST eval |
| **ONNX specialists** | 0 tokens | Confident sentiment and NER |
| **Local GGUF** | 0 tokens | Categories in `LOCAL_CATEGORIES` on hosts that clear the TPS bar |
| **Cloud** | billed | Everything uncertain, plus code, logic, and summarization |

**Escalation** is the core invariant: a cheap lane answers only when confident;
otherwise it returns `""` and the task falls through. Paying ~100 cloud tokens
always beats an accuracy-gate miss.

### Pipeline

1. **Classify** (`agent/router.py`) — lexical cues assign one of eight
   categories: `factual`, `math`, `sentiment`, `summarization`, `ner`,
   `code_debug`, `logic`, `code_gen`. Instruction verbs are checked before
   content cues that could appear inside quoted text. Unknown prompts fall
   back to `factual` (safest, cloud-routed). Each classification carries a
   confidence score; below `ROUTER_MIN_CONFIDENCE` (default `0.70`) the task
   goes straight to cloud.
2. **Specialists** (`agent/specialists.py`) — sentiment and NER run first for
   their categories when weights are present. Both are self-gating: hedges,
   low confidence, incomplete entity coverage, or uncovered capitalized spans
   all escalate.
3. **Local model** (`agent/local.py`) — only for categories in
   `LOCAL_CATEGORIES`, only after warmup calibration proves the host clears
   the category's tokens-per-second bar, and only while `LOCAL_TIME_BUDGET`
   and `LOCAL_CUTOFF` allow. Answers pass `answer_ok()` or escalate.
4. **Cloud** (`agent/cloud.py`) — terse `prompts.SPECS` system prompts,
   `temperature=0`, `reasoning_effort="none"`, double token budget
   (`max_tokens`, then `max_tokens*4`), transient-only retries, and model
   fallback across `ALLOWED_MODELS` (kimi preferred — ~54 prompt-token
   overhead vs minimax's ~163).

## Quick Start

### Prerequisites

- Python 3.10+
- A Fireworks API key (cloud lane; required for any non-dry-run)
- Optional: ONNX weights for specialists, GGUF weights for the local lane

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e .                    # core (cloud-only)
```

Optional extras:

```bash
pip install -e ".[specialists]"     # numpy, onnxruntime, tokenizers
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install -e ".[dev]"             # pytest, ruff
pip install -e ".[finetune]"        # transformers, trl, peft, datasets, accelerate
pre-commit install                  # ruff + secret scanning on each commit
```

Fine-tuning pipeline: see [`finetune/README.md`](finetune/README.md); start
with `python -m finetune.validate_data` before any training run.

### Configure

```bash
cp .env.example .env
```

```env
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
```

Never commit `.env` — the grading harness injects real credentials; `scripts/
check_secrets.py` enforces this in pre-commit and CI.

Optional local-lane variables:

```env
LOCAL_MODEL_PATH=/app/models/model.gguf
LOCAL_CATEGORIES=factual,math,ner    # empty = cloud-only (default)
NER_MODEL_DIR=/app/models/ner
SST2_MODEL_DIR=/app/models/sst2
NER_SPECIALIST=on                    # set 'off' to disable the NER specialist
```

Model weights are git-ignored. Download and verify them (SHA-256 checked)
with:

```bash
python scripts/download_models.py onnx   # NER + SST-2 specialists
python scripts/download_models.py gguf   # Qwen2.5 GGUF weights
python scripts/download_models.py all
```

### Run

Default contract: read `TASKS_PATH` (default `/input/tasks.json`), write
`RESULTS_PATH` (default `/output/results.json`).

```bash
python -m agent.main        # or: querylane
```

Local development with custom paths:

```bash
# Windows PowerShell
$env:TASKS_PATH="testio/input_dev/tasks.json"
$env:RESULTS_PATH="testio/output_dev/results.json"
python -m agent.main

# macOS/Linux
TASKS_PATH=testio/input_dev/tasks.json RESULTS_PATH=testio/output_dev/results.json python -m agent.main
```

Dry run — report the lane each task would take, spend zero tokens:

```bash
python -m agent.main --dry-run    # or AGENT_DRY_RUN=1
python -m agent.explain "Calculate 25 * 40 - 150"   # explain one prompt
```

### Input / output contract

Input is a JSON array of `{"task_id": "...", "prompt": "..."}`. Output is a
JSON array of `{"task_id": "...", "answer": "..."}` with these guarantees:

- Every input `task_id` gets exactly one output row, in input order, even on
  failure (failed tasks get `""`).
- Duplicate `task_id`s are answered once; every row receives that answer.
- Missing, `null`, or non-string prompts are treated as empty and routed to
  cloud.
- A global deadline (`AGENT_DEADLINE`, default 480s; harness kills at 600s)
  guarantees results are written in time; unfinished tasks keep `""`.

## Evaluation

```bash
python -m eval.judge                              # hybrid grader, 3 runs
python -m eval.judge --category math
python -m eval.judge --answers eval/results_dev.json   # re-judge saved answers
python -m eval.judge --answers eval/results_dev.json --grader deterministic  # offline
python -m eval.judge --grader llm --runs 5        # historical all-LLM mode
```

Grading is deterministic-first: `eval/checks_dev.json` supplies mechanical
checks (labels, sentence/word counts, entity sets, numbers, code structure)
for 35 of the 40 dev tasks; only undecided tasks reach the LLM judge.
`--grader hybrid` (default) prefers deterministic verdicts, `deterministic`
fails undecided tasks, `llm` reproduces the old all-LLM behavior. `--runs N`
reports per-task verdict flips and accuracy mean/stdev/range to
`eval/judge_variance.json` and the dashboard. The judge prompt, its failure
modes, and grader details are documented in [docs/judge.md](docs/judge.md).

Offline cost and precision checks:

```bash
python -m eval.cost_regression              # compare token budgets (offline)
python -m eval.spec_precision               # specialist precision on synthetic data
python -m eval.bench_local                  # local-lane benchmark
```

`eval/cost_regression` replays the dev set through real routing code for
`cloud-only`/`specialists`/`hybrid` configs with a simulated cloud client;
budgets live in `eval/token_budgets.json` and are enforced by
`tests/test_token_budget.py`.

## Configuration

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TASKS_PATH` | `/input/tasks.json` | Input batch file |
| `RESULTS_PATH` | `/output/results.json` | Output batch file (bare filename OK) |
| `AGENT_WORKERS` | `4` | Thread pool size |
| `AGENT_DEADLINE` | `480` | Seconds before the agent stops collecting answers |
| `LOCAL_CUTOFF` | `150` | Min seconds remaining to start local work |
| `LOCAL_CATEGORIES` | empty | Comma-separated categories allowed on local model |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | GGUF weights path |
| `LOCAL_MIN_TPS` | `0.4` | Global local speed floor (tok/s) |
| `LOCAL_TIME_BUDGET` | `300` | Total seconds for local inference |
| `LOCAL_THREADS` | `2` | llama.cpp threads |
| `NER_MODEL_DIR` | `/app/models/ner` | ONNX NER model directory |
| `SST2_MODEL_DIR` | `/app/models/sst2` | ONNX sentiment model directory |
| `NER_SPECIALIST` | `on` | Enable the NER specialist (`on`/`off`) |
| `NER_MIN_CONFIDENCE` | `0.85` | Per-token confidence floor for NER answers |
| `NER_MIN_MARGIN` | `0.05` | Per-token top-two label margin for NER |
| `NER_COMPLETENESS_GUARD` | `on` | Escalate unless every capitalized span/date is covered |
| `ROUTER_MIN_CONFIDENCE` | `0.70` | Below this, route straight to cloud |
| `CLOUD_TASK_BUDGET` | `75` | Per-task cloud time budget (s) |
| `CLOUD_BACKOFF_BASE` | `0.5` | Transient-retry base delay (s) |
| `CLOUD_BACKOFF_CAP` | `8.0` | Max backoff delay per retry (s) |
| `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` | — | Required cloud credentials |
| `AGENT_DEBUG` | unset | Human-readable per-call token logs |
| `AGENT_LOG_JSON` | unset | One JSON event per task/cloud call |
| `AGENT_DRY_RUN` | unset | Same as `--dry-run` |
| `METRICS_PATH` | unset | Write per-run metrics JSON snapshot |

### Observability

Both outputs are off by default so a graded run stays quiet.

`AGENT_LOG_JSON=1` emits one JSON object per event (`task`, `cloud_call`,
`cloud_retry`, `cloud_budget_exceeded`, `task_abandoned`,
`duplicate_task_ids`, `dry_run`, `run_summary`):

```json
{"ts": 1730000000.12, "level": "info", "event": "task", "task_id": "practice-02",
 "category": "math", "lane": "local", "confidence": 0.86, "predicted_lane": "local",
 "elapsed_seconds": 12.4, "tokens": 0, "empty_answer": false}
```

`METRICS_PATH=metrics.json` writes a post-run snapshot: task/lane/category
counts, escalations, empty answers, low-confidence routes, per-category token
deltas, and cloud call stats. Lane counts always sum to `tasks`; exact billed
tokens are under `cloud.tokens` (per-category deltas can overlap slightly
under concurrent workers).

### Tuning for your host

- **Fast host** — add categories to `LOCAL_CATEGORIES`; local engages only
  where the measured TPS clears the per-category bar.
- **Slow host** — leave `LOCAL_CATEGORIES` empty (cloud-only) rather than
  lowering `LOCAL_MIN_TPS` below what can finish inside the deadline.
- **Deadline pressure** — lower `AGENT_DEADLINE`/`CLOUD_TASK_BUDGET`;
  unfinished tasks still get `""` rows.

See [docs/troubleshooting.md](docs/troubleshooting.md) for missing-model and
slow-host diagnostics.

## Docker

```bash
docker build -t querylane .          # cloud-only, pinned digest + hashed lockfile
docker build -f Dockerfile.local -t querylane-local .   # hybrid w/ specialists+GGUF
```

```bash
docker run --rm \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  -v "$PWD/testio/input:/input:ro" -v "$PWD/testio/output:/output" \
  querylane
```

Notes:

- Images run as uid/gid 10001 — make bind-mounted `/output` writable
  (`chmod 0777 testio/output` or a named volume).
- `--dry-run` works inside the container with no credentials.
- Multi-arch (amd64/arm64) builds for the cloud image:
  `docker buildx build --platform linux/amd64,linux/arm64 -t querylane .`
- ⚠️ The llama-cpp CPU wheel index only ships x86_64 wheels — build hybrid
  images for ARM64 with `--platform linux/amd64` (emulation) or compile from
  source (see `Dockerfile.local`).
- Versioned hybrid Dockerfiles: `Dockerfile.phase2`, `.v10`, `.v13`, `.v14`,
  `.v15`. Dependencies install from `requirements.lock` /
  `requirements-hybrid.lock` with `--require-hashes`.

## Repository layout

```text
agent/       runtime: router, prompts, cloud, local, specialists, pal, obs
eval/        judge, deterministic checks, cost regression, dashboards, benches
finetune/    synthetic data gen, LoRA training, GGUF conversion, notebooks
models/      model configs/tokenizers (weights git-ignored)
testio/      local input batches for dev and stress testing
tests/       router, PAL fuzz, cloud, local calibration, deadline, secret scan
docs/        architecture, judge, troubleshooting, category guide, security
scripts/     model downloader + shared secret scanner
Dockerfile*  cloud-only + versioned hybrid images
SCORES.md    historical eval log — token/accuracy trade-off history
TODO.md      roadmap, known limitations, next steps
```

## Design principles

- **Terse system prompts** — every system token is billed on every call
- **No content-based retries** — transient errors and model fallback only; a
  second attempt costs tokens
- **Precision over coverage** — a specialist answering 38% at 100% precision
  beats one answering 80% at 90%
- **Escalation is cheap** — ~100 cloud tokens beats an accuracy-gate miss
- **Calibrate, then decide** — local lanes engage per category from measured
  tok/s, not host assumptions
- **Safety by default** — specialists escalate rather than guess

## Status & limitations

Research/competition codebase, not a hosted service. Score history lives in
[SCORES.md](SCORES.md); roadmap and known gaps in [TODO.md](TODO.md).

- Sentiment specialist has no neutral class — neutrals escalate by design
- NER specialist is precision-gated: uncertain tokens, uncovered spans, and
  incomplete coverage escalate to cloud
- Local inference only pays off above per-category TPS thresholds
- Dev judge is deterministic for 35/40 tasks; code categories still use an
  LLM — run `--runs` before trusting small accuracy deltas
- `train_all.jsonl`/`train_round2.jsonl` overlap; `train_dedup.jsonl` is the
  canonical deduplicated file (see `finetune/README.md`)

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and
coding conventions, and [TODO.md](TODO.md) for the roadmap. Good first areas:
specialist precision/coverage, new categories (see
[docs/adding-a-category.md](docs/adding-a-category.md)), routing improvements,
and test coverage.

## License

[MIT](LICENSE)
