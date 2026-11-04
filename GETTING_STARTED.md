# Getting Started with QueryLane

This guide will walk you through setting up and using QueryLane from scratch. Whether you want to run a simple query batch, set up local specialists, or dive into fine-tuning, this guide has you covered.

## Table of Contents

1. [Installation](#installation)
2. [Basic Setup](#basic-setup)
3. [Your First Query Batch](#your-first-query-batch)
4. [Understanding the Output](#understanding-the-output)
5. [Setting Up Local Specialists](#setting-up-local-specialists)
6. [Using the Local GGUF Model](#using-the-local-gguf-model)
7. [Running Evaluations](#running-evaluations)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

## Installation

### Step 1: Check Prerequisites

Ensure you have the following installed:

- **Python 3.10 or higher**
  ```bash
  python --version
  ```

- **pip** (comes with Python)
  ```bash
  pip --version
  ```

- **Git** (if cloning from repository)
  ```bash
  git --version
  ```

### Step 2: Clone or Download the Repository

If you haven't already:

```bash
git clone https://github.com/Ifaz2611/QueryLane
cd QueryLane
```

Or download and extract the repository zip file.

### Step 3: Create a Virtual Environment

Creating a virtual environment is highly recommended to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt when activated.

### Step 4: Install Core Dependencies

Install the base package:

```bash
pip install -e .
```

This installs QueryLane in editable mode with its core dependencies (OpenAI SDK).

### Step 5: Install Optional Components

Depending on your use case, you may want to install additional components:

#### For ONNX Specialists (Sentiment/NER)

```bash
pip install -e ".[specialists]"
```

This installs:
- `numpy` - Numerical computing
- `onnxruntime` - ONNX model runtime
- `tokenizers` - Hugging Face tokenizers

#### For Local GGUF Inference

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**Note:** For GPU acceleration, replace `cpu` with `cu117` (CUDA 11.7), `cu118` (CUDA 11.8), or `cu121` (CUDA 12.1) depending on your CUDA version.

#### For Development

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` / `pytest-cov` - Testing framework and coverage
- `pre-commit` - Ruff and secret-scanning hooks
- `ruff` - Python linter

#### For Fine-Tuning

```bash
pip install -e ".[finetune]"
```

This installs machine learning libraries for training and converting models.

## Basic Setup

### Step 1: Configure Environment Variables

Copy the example environment file:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` with your text editor and add your Fireworks API credentials:

```env
FIREWORKS_API_KEY=fw_your_actual_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
```

**Important:** Never commit `.env` to version control. It's already in `.gitignore`.

### Step 2: Verify Your Setup

Test that QueryLane is installed correctly:

```bash
python -c "import agent; print('QueryLane imported successfully')"
```

Or try the console script:

```bash
querylane --help
```

## Your First Query Batch

### Step 1: Create Input Data

Create a file named `tasks.json` with your queries:

```json
[
  {
    "task_id": "001",
    "prompt": "What is the capital of France?"
  },
  {
    "task_id": "002",
    "prompt": "Classify the sentiment: I absolutely love this product!"
  },
  {
    "task_id": "003",
    "prompt": "Calculate: 15 * 7"
  }
]
```

### Step 2: Run QueryLane

Run the agent with default paths:

```bash
# Create input directory
mkdir input
move tasks.json input\tasks.json  # Windows
# mv tasks.json input/tasks.json  # macOS/Linux

# Create output directory
mkdir output

# Run QueryLane
python -m agent.main
```

Or override paths without moving files:

```bash
# Windows PowerShell
$env:TASKS_PATH="tasks.json"
$env:RESULTS_PATH="results.json"
python -m agent.main

# macOS/Linux
export TASKS_PATH="tasks.json"
export RESULTS_PATH="results.json"
python -m agent.main
```

### Step 3: Check the Results

Open `results.json` (or your specified output path):

```json
[
  {
    "task_id": "001",
    "answer": "Paris"
  },
  {
    "task_id": "002",
    "answer": "positive"
  },
  {
    "task_id": "003",
    "answer": "105"
  }
]
```

## Understanding the Output

### Output Format

Each result contains:
- `task_id` - Matches your input task identifier
- `answer` - The processed answer from the appropriate lane

### What Just Happened?

QueryLane routed each query through its lane hierarchy:

1. **Task 001 (factual)** - Routed to cloud LLM (no specialist handles general knowledge)
2. **Task 002 (sentiment)** - Routed to ONNX sentiment specialist (if installed) or cloud
3. **Task 003 (math)** - Routed to cloud LLM with possible PAL evaluation

### Enable Debug Logging

To see which lane was used for each task:

```bash
# Windows PowerShell
$env:AGENT_DEBUG="1"
python -m agent.main

# macOS/Linux
export AGENT_DEBUG=1
python -m agent.main
```

This logs token usage and routing decisions for each task.

### Structured Logs and Metrics

For machine-readable output, ask for JSON events and a metrics snapshot:

```bash
# macOS/Linux
AGENT_LOG_JSON=1 METRICS_PATH=metrics.json python -m agent.main

# Windows PowerShell
$env:AGENT_LOG_JSON="1"; $env:METRICS_PATH="metrics.json"; python -m agent.main
```

`AGENT_LOG_JSON` prints one JSON object per task (`task_id`, `category`,
`lane`, `confidence`, `elapsed_seconds`, `tokens`) plus cloud retry, budget, and
run-summary events. `METRICS_PATH` writes a single JSON file with tokens by
category, lane counts, escalations, empty answers, and duplicate counts — check
it after each run so a regression is visible without reading logs.

### Dry Run Before Spending Tokens

```bash
python -m agent.main --dry-run        # or: AGENT_DRY_RUN=1 python -m agent.main
```

Every task is routed and its predicted lane (`cloud`, `specialist`, `local`) is
reported, but no cloud or local model call is made. Results contain empty
answers, so use it to sanity-check router or `LOCAL_CATEGORIES` changes.

## Setting Up Local Specialists

Local specialists provide zero-token inference for specific task types. Currently supported:

- **Sentiment analysis** (SST-2 model, negation- and quote-aware cue gates)
- **Named Entity Recognition** (enabled by default when the model is present)

### Step 1: Download ONNX Models

Model weights are not included in the repository. You'll need to obtain them:

1. Check `finetune/POD_INSTRUCTIONS.md` for training instructions
2. Or use pre-trained models if available
3. Place models in appropriate directories:
   - Sentiment: `/app/models/sst2/` (or set `SST2_MODEL_DIR`)
   - NER: `/app/models/ner/` (or set `NER_MODEL_DIR`)

### Step 2: Configure Paths

Add to your `.env`:

```env
SST2_MODEL_DIR=/path/to/your/sst2/model
NER_MODEL_DIR=/path/to/your/ner/model
NER_SPECIALIST=on  # default; set to 'off' to disable NER
ROUTER_MIN_CONFIDENCE=0.70  # low-confidence routes go straight to cloud
```

### Step 3: Test Specialists

```bash
python -m eval.spec_precision
```

This runs specialists on synthetic datasets and reports precision/coverage without API calls.

### Expected Performance

- **Sentiment specialist**: ~38% coverage at ~100% precision (designed to escalate on uncertainty)
- **NER specialist**: enabled by default with per-token confidence/margin gates and a completeness guard. It escalates unless every capitalized span and date in the passage is covered, so re-run `eval.spec_precision` with your model weights to measure the current precision/coverage on your host.

## Using the Local GGUF Model

The local GGUF model can handle factual, math, and NER queries on fast hosts.

### Step 1: Download or Train a GGUF Model

1. Train a model using the fine-tuning pipeline (see `finetune/POD_INSTRUCTIONS.md`)
2. Convert to GGUF format
3. Place in a directory accessible to QueryLane

### Step 2: Configure Local Model

Add to your `.env`:

```env
LOCAL_MODEL_PATH=/path/to/your/model.gguf
LOCAL_CATEGORIES=factual,math,ner  # Categories to use local model for
LOCAL_MIN_TPS=0.4  # Minimum tokens/second threshold
LOCAL_TIME_BUDGET=300  # Total seconds for local inference
LOCAL_THREADS=2  # Number of llama.cpp threads
```

### Step 3: Calibration

QueryLane automatically calibrates local model speed on startup:

1. Runs a warm-up prompt
2. Measures tokens/second
3. Enables local inference only if speed exceeds `LOCAL_MIN_TPS`

On slow hosts, the local model will be skipped even if configured.

### Step 4: Test with Debug Logging

```bash
export AGENT_DEBUG=1
python -m agent.main
```

Look for log messages indicating local model usage or skipping due to speed.

## Running Evaluations

### Development Judge

Evaluate your results against the development set:

```bash
python -m eval.judge
```

Filter by category:

```bash
python -m eval.judge --category math
python -m eval.judge --category sentiment
```

Re-judge saved results:

```bash
python -m eval.judge --answers eval/results_dev.json
```

### Specialist Precision

Measure specialist performance offline:

```bash
python -m eval.spec_precision
```

This reports:
- Coverage: Percentage of tasks the specialist attempts
- Precision: Percentage of attempted tasks answered correctly

### Running Tests

If you installed dev dependencies:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py -v

# Run the secret-scanning hook over every tracked file
python scripts/check_secrets.py --all
```

## Troubleshooting

### Common Issues

#### "Module not found" Errors

**Problem:** Import errors for `agent` or dependencies.

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

#### "FIREWORKS_API_KEY not found"

**Problem:** API key not configured.

**Solution:**
```bash
# Ensure .env file exists
ls .env  # macOS/Linux
dir .env  # Windows

# Check contents (be careful not to expose)
cat .env  # macOS/Linux
type .env  # Windows

# Ensure variable name matches exactly (case-sensitive)
```

#### Local Model Not Being Used

**Problem:** Local model configured but queries still going to cloud.

**Solution:**
1. Enable debug logging: `export AGENT_DEBUG=1`
2. Check logs for speed calibration results
3. If speed is below `LOCAL_MIN_TPS`, increase threshold or use a faster host
4. Verify `LOCAL_CATEGORIES` includes the task category

#### ONNX Import Errors

**Problem:** Errors importing ONNX models or `onnxruntime`.

**Solution:**
```bash
# Ensure specialists extra is installed
pip install -e ".[specialists]"

# Verify installation
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
```

#### Slow Performance

**Problem:** Queries taking too long.

**Solution:**
1. Reduce `AGENT_WORKERS` if CPU-bound
2. Decrease `AGENT_DEADLINE` to stop early
3. Disable local model if host is slow
4. Check network latency to Fireworks API

### Getting Help

If you encounter issues not covered here:

1. Check [`TODO.md`](TODO.md) for known limitations
2. Review logs with `AGENT_DEBUG=1` enabled
3. Check Fireworks API status
4. Open an issue on the project repository

## Next Steps

Now that you have QueryLane running, explore further:

- **Fine-tune your own models**: See [`finetune/POD_INSTRUCTIONS.md`](finetune/POD_INSTRUCTIONS.md)
- **Optimize for your workload**: Adjust configuration parameters based on your task distribution
- **Contribute improvements**: See [`TODO.md`](TODO.md) for areas needing work
- **Experiment with routing**: Try different `LOCAL_CATEGORIES` and thresholds
- **Monitor costs**: Use `AGENT_DEBUG=1` to track token usage per category

### Advanced Topics

- **Adding new categories**: Modify `agent/router.py` and `agent/prompts.py`
- **Custom specialists**: Implement new specialists in `agent/specialists.py`
- **Batch optimization**: Tune `AGENT_WORKERS` and category-specific budgets
- **Multi-model deployment**: Configure multiple `ALLOWED_MODELS` for fallback

## Summary

You now have:
- QueryLane installed and configured
- Successfully run your first query batch
- Understood the routing pipeline
- Optional local specialists set up
- Optional local GGUF model configured
- Evaluation tools ready
- troubleshooting skills

Happy routing! 
