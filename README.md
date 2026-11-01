# QueryLane

**Cost-aware LLM query routing.** QueryLane intelligently routes each query through a hierarchy of processing lanes, always choosing the cheapest option that can provide a correct answer. It prioritizes deterministic rules and zero-token local specialists before falling back to cloud LLMs only when necessary.

QueryLane was originally developed as a Track 1 submission for a competition where the scoring metric was Fireworks token usage, subject to an 80% accuracy gate. The routing methodology is general-purpose: classify the request, spend nothing when a specialist is confident, and escalate on doubt rather than risk an incorrect answer.

## Key Features

- **Zero-cost specialists**: ONNX-based models for sentiment analysis and NER that run entirely locally
- **Smart routing**: Lexical cue-based classification to determine the optimal processing lane
- **Escalation on uncertainty**: When cheap lanes aren't confident, tasks automatically fall through to more capable (but more expensive) options
- **Local model support**: Optional GGUF model for factual, math, and NER queries with automatic speed calibration
- **Cost optimization**: Terse system prompts and minimal retries to reduce token usage
- **Safety-first**: Precision-focused specialists that escalate rather than guess

## Architecture

QueryLane uses a multi-lane routing strategy that processes queries in order of increasing cost:

| Lane | Cost | Used For |
|------|------|----------|
| **Rules** | 0 tokens | Category classification, instruction stripping, answer normalization |
| **ONNX Specialists** | 0 tokens | Confident sentiment classification (SST-2 + cue gates) |
| **Local GGUF Model** | 0 tokens | Fine-tuned factual/math/NER answers when host is fast enough |
| **Cloud Model** | Billed | Everything uncertain, plus code, logic, and summarization |

### How Escalation Works

The accuracy gate is protected by **escalation**: when a cheap lane is not confident in its answer, it returns an empty response and the task automatically falls through to the next lane. This ensures that QueryLane never risks an incorrect answer just to save tokens—when in doubt, it escalates.

## Routing Pipeline

### Category Classification

The router in `agent/router.py` classifies each prompt into one of eight categories using only lexical cues:

- `factual` - General knowledge questions
- `math` - Mathematical problems and calculations
- `sentiment` - Sentiment analysis and opinion extraction
- `summarization` - Text summarization tasks
- `ner` - Named entity recognition
- `code_debug` - Debugging and code analysis
- `logic` - Logical reasoning problems
- `code_gen` - Code generation tasks

The check order is intentional: instruction verbs (e.g., "extract entities", "fix this bug", "write a function", "summarize") are tested before content cues that might appear within quoted passages. Unknown prompts default to `factual`, which is the safest cloud-routed category.

### Lane Selection

After classification, `agent/main.py` applies the lanes in sequence:

1. **Specialist Lane** — Sentiment (and optionally NER) via ONNX models. These specialists are self-gating and return empty strings when they detect hedges, contrast, low confidence, or uncertain spans.

2. **Local Model Lane** — Only for categories listed in `LOCAL_CATEGORIES`, and only after warm-up calibration proves the host clears the category's speed threshold. This ensures local inference is only used when it's actually faster than cloud.

3. **Cloud Lane** — The fallback in `agent/cloud.py` uses terse per-category system prompts, disables reasoning features where possible, and falls back across allowed models if one fails.

### Math Processing

Math queries have an additional optimization: the cloud model may return a structured response with `EXPR: ... | UNIT: ...`, which `agent/pal.py` evaluates locally using an AST whitelist for safety. If parsing or safety checks fail, the system falls back to step-by-step generation.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- A Fireworks API key (for cloud LLM access)
- (Optional) ONNX model weights for local specialists
- (Optional) GGUF model for local inference

### Installation

#### 1. Clone and Set Up Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

#### 2. Install Optional Components

**For ONNX specialists (sentiment/NER):**
```bash
pip install -e ".[specialists]"
```

**For local GGUF inference:**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**For development/testing:**
```bash
pip install -e ".[dev]"
pre-commit install  # enable ruff + secret scanning before each commit
```

**For fine-tuning capabilities:**
```bash
pip install -e ".[finetune]"
```

### Configuration

#### 1. Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Fireworks credentials:

```env
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
```

**Security Note:** Never commit `.env` to version control. The grading harness injects real credentials at evaluation time; your local development file should stay private.

#### 2. Model Paths (Optional)

If using local specialists or GGUF models, set these environment variables:

```env
LOCAL_MODEL_PATH=/app/models/model.gguf
LOCAL_CATEGORIES=factual,math,ner
NER_MODEL_DIR=/app/models/ner
SST2_MODEL_DIR=/app/models/sst2
NER_SPECIALIST=off  # Set to 'on' to enable experimental NER specialist
```

### Running Queries

#### Basic Usage

The default contract reads from `/input/tasks.json` and writes to `/output/results.json`:

```bash
python -m agent.main
```

Or use the installed console script:

```bash
querylane
```

#### Local Development

Override default paths for local testing:

```bash
# PowerShell
$env:TASKS_PATH="testio/input_dev/tasks.json"
$env:RESULTS_PATH="testio/output_dev/results.json"
python -m agent.main

# macOS/Linux
export TASKS_PATH="testio/input_dev/tasks.json"
export RESULTS_PATH="testio/output_dev/results.json"
python -m agent.main
```

#### Input Format

Tasks should be provided as a JSON array:

```json
[
  {
    "task_id": "task_001",
    "prompt": "What is the capital of France?"
  },
  {
    "task_id": "task_002",
    "prompt": "Classify the sentiment: I love this product!"
  }
]
```

#### Output Format

Results are returned as a JSON array:

```json
[
  {
    "task_id": "task_001",
    "answer": "Paris"
  },
  {
    "task_id": "task_002",
    "answer": "positive"
  }
]
```

### Evaluation

#### Run the Development Judge

```bash
python -m eval.judge
python -m eval.judge --category math
python -m eval.judge --answers eval/results_dev.json   # re-judge saved answers
```

The judge uses a Fireworks LLM call (tokens tracked separately from the routed agent's score).

#### Measure Specialist Precision (Offline)

```bash
python -m eval.spec_precision
```

This replays validated synthetic datasets through ONNX specialists and reports coverage and precision without making any API calls.

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TASKS_PATH` | `/input/tasks.json` | Input batch file path |
| `RESULTS_PATH` | `/output/results.json` | Output batch file path |
| `AGENT_WORKERS` | `4` | Thread pool size for parallel processing |
| `AGENT_DEADLINE` | `480` | Seconds before the agent stops collecting answers |
| `LOCAL_CUTOFF` | `150` | Minimum seconds required to start local work |
| `LOCAL_CATEGORIES` | empty | Comma-separated categories allowed to use local model |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | Path to GGUF model weights |
| `LOCAL_MIN_TPS` | `0.4` | Global local-inference speed floor (tokens/sec) |
| `LOCAL_TIME_BUDGET` | `300` | Total seconds allocated for local inference |
| `LOCAL_THREADS` | `2` | Number of llama.cpp threads |
| `NER_MODEL_DIR` | `/app/models/ner` | Directory for ONNX NER model |
| `SST2_MODEL_DIR` | `/app/models/sst2` | Directory for ONNX sentiment model |
| `NER_SPECIALIST` | `off` | Enable experimental ONNX NER specialist (`on`/`off`) |
| `AGENT_DEBUG` | unset | Enable per-call token usage logging |

### Fine-Tuning Performance

Adjust these parameters based on your hardware and workload:

- **Fast hosts**: Increase `LOCAL_CATEGORIES` and `LOCAL_MIN_TPS` to use local model more
- **Slow hosts**: Decrease `LOCAL_MIN_TPS` or leave `LOCAL_CATEGORIES` empty to rely on cloud
- **Cost-sensitive**: Lower `AGENT_DEADLINE` to prevent expensive long-running cloud calls
- **Accuracy-critical**: Increase `LOCAL_CUTOFF` to give local specialists more time

## Repository Structure

```text
agent/            Runtime components: router, prompts, cloud client, local model, specialists, PAL
eval/             Development tools: judge, local benchmark, specialist precision harness
finetune/         Fine-tuning pipeline: synthetic data generator, LoRA training, GGUF conversion, notebooks
models/           ONNX configs and tokenizers (weights are gitignored)
testio/           Local input batches for practice, development, and stress testing
tests/            Router, PAL fuzz, cloud, local calibration, deadline, and secret-scan tests
scripts/          Secret-scanning hook shared by pre-commit and CI
requirements.lock Fully pinned, hash-verified cloud image dependencies
Dockerfile*       Various Docker images: cloud-only, local, phase-2, and versioned hybrid builds
SCORES.md         Historical evaluation log with token/accuracy findings
TODO.md           Roadmap, next steps, and known limitations
```

## 🐳 Docker

### Cloud-Only Image

The default `Dockerfile` creates a minimal cloud-only image containing only the core runtime:

```bash
docker build -t querylane .
# The image runs as uid/gid 10001. Make the bind-mounted output directory
# writable by that uid (or use a named volume) before starting the container.
mkdir -p testio/output && chmod 0777 testio/output
docker run --rm \
  -e FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS \
  -v "$PWD/testio/input:/input:ro" \
  -v "$PWD/testio/output:/output" \
  querylane
```

The base image is pinned by manifest digest and runtime dependencies are
installed from `requirements.lock` with `--require-hashes`; hybrid images use
`requirements-hybrid.lock`. This makes image rebuilds reproducible.

### Hybrid Images

Versioned Dockerfiles bundle ONNX specialists and/or GGUF weights for local inference:

- `Dockerfile.phase2` - Phase 2 hybrid build
- `Dockerfile.v10` - Version 10 hybrid build
- `Dockerfile.v13` - Version 13 hybrid build
- `Dockerfile.v14` - Version 14 hybrid build
- `Dockerfile.v15` - Version 15 hybrid build

**Note:** Model weights are intentionally not included in the repository. See `finetune/POD_INSTRUCTIONS.md` for the complete build-and-convert workflow.

### Local Development Image

For local development with specialist support:

```bash
docker build -f Dockerfile.local -t querylane-local .
mkdir -p testio/output && chmod 0777 testio/output
docker run --rm \
  -e FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS \
  -v "$PWD/testio/input:/input:ro" \
  -v "$PWD/testio/output:/output" \
  querylane-local
```

## Design Principles

QueryLane's design prioritizes cost efficiency without sacrificing accuracy:

- **Terse system prompts** - Every system token is billed on every call, so prompts are kept minimal
- **No content-based retries** - Only transient-error retries and model fallback; a second attempt costs tokens
- **Precision over coverage** - A specialist that answers 38% of tasks at 100% precision is better than one that answers 80% at 90% precision
- **Escalation is cheap** - Paying ~100 tokens for a cloud answer beats an accuracy-gate miss
- **Calibrate, then decide** - Local inference is enabled per category from a measured tokens-per-second rate, not from assumptions about the host
- **Safety by default** - Specialists escalate rather than guess when uncertain

## Project Status

QueryLane is a research/competition codebase, not a hosted service. The historical score log, including the token/accuracy trade-offs that shaped the router, is available in [`SCORES.md`](SCORES.md). Known gaps and planned improvements are documented in [`TODO.md`](TODO.md).

### Current Limitations

- The sentiment specialist has no neutral class; neutral reviews are escalated by design
- The ONNX NER specialist is disabled by default due to precision concerns (~22% exact-match)
- Local generative inference is only useful on hosts above per-category speed thresholds
- The development judge uses an LLM and is not fully deterministic
- Training datasets `train_all.jsonl` and `train_round2.jsonl` have overlap that needs deduplication

## Contributing

Contributions are welcome! QueryLane is actively being improved. Key areas for contribution include:

- Improving specialist precision and coverage
- Adding new task categories
- Enhancing the routing algorithm
- Expanding the test suite
- Documentation improvements

See [`TODO.md`](TODO.md) for a detailed roadmap of planned features and known issues.

## Additional Resources

- [`SCORES.md`](SCORES.md) - Historical evaluation results and performance metrics
- [`TODO.md`](TODO.md) - Roadmap, known issues, and planned improvements
- [`finetune/POD_INSTRUCTIONS.md`](finetune/POD_INSTRUCTIONS.md) - Model training and conversion guide

## License

Released under the [MIT License](LICENSE).
