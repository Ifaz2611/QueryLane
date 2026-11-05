# Changelog

All notable changes to the **QueryLane** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.16.0] - 2026-11-05

### Added
- **Interactive Route Explainer**: Added `agent/explain.py` CLI and `querylane-explain` console script to inspect category signals, confidence scores, runtime lane decisions, and cost estimates.
- **Zero-Dependency Web Demo**: Added `agent/web_demo.py` featuring an interactive web UI with preset prompts, live lane pipeline visualization, and token cost breakdown.
- **Automated Model Downloader**: Added `scripts/download_models.py` with cryptographic SHA-256 verification for ONNX specialists (NER, SST-2) and GGUF models.
- **Developer Task Runners**: Added `Makefile` and `justfile` providing recipes for testing, coverage, linting, formatting, secret scanning, building, and Docker.
- **Multi-Architecture Docker Builds**: Configured GitHub Actions CI for multi-arch `linux/amd64` and `linux/arm64` container builds.
- **Architecture Documentation & Wordmark**: Created SVG wordmark logo and `docs/lanes.md` detailing the complete multi-lane decision flowchart and escalation state machine.
- **Packaging in CI**: Added automated sdist and wheel build job in GitHub Actions with `twine check` validation.
- **Deterministic Dev-Set Evaluator**: Added `eval/deterministic.py` evaluating 35/40 tasks with zero API calls.
- **Structured JSON Observability**: Added `agent/obs.py` supporting `AGENT_LOG_JSON=1` and `METRICS_PATH` counters.

### Changed
- Complete project rebrand from `LLM-Query-Route` to **QueryLane**.
- Updated `pyproject.toml` with PEP 621 metadata (authors, documentation and issue tracker URLs, classifiers, entrypoints).
- Documented the prebuilt CPU wheel limitation of `llama-cpp-python` on ARM64 platforms across `README.md`, `GETTING_STARTED.md`, and Dockerfiles.
- Hardened router confidence scoring (`route_with_confidence`) to immediately short-circuit routes below 0.70 confidence to cloud.
- Hardened PAL math evaluation with AST operator whitelists, exponent bounds, and rejecting non-numeric results.
- Added equal-jitter exponential backoff and a 75-second per-task cloud deadline (`CLOUD_TASK_BUDGET`).

---

## [0.15.0] - 2026-07-13

### Added
- Per-category speed bars (math 0.4 tok/s, NER 0.9 tok/s, factual 1.3 tok/s) allowing categories to degrade gracefully to cloud individually.
- Queue prioritization by savings-per-second (math -> NER -> factual).

---

## [0.14.0] - 2026-07-13

### Added
- Fine-tuned Qwen2.5-1.5B guarded NER with zero-token completeness verification (every capitalized span and date must be present in the output).

---

## [0.13.0] - 2026-07-12

### Added
- Fine-tuned Qwen2.5-1.5B (Unsloth LoRA on 2,753 validated rows) serving factual and math PAL queries locally.
- Normalizers for document and treaty entity names.

---

## [0.12.0] - 2026-07-12

### Added
- Cloud-PAL (Program-Aided Language) math: kimi emits structured formulas evaluated locally via Python AST parser (~15 completion tokens).
- Paraphrase test suite and router hardening (64/64 dev paraphrase prompts).
- Sentence-count constraints for summarization queries.

---

## [0.10.0] - 2026-07-12

### Added
- Non-generative int8 ONNX specialists for BERT-base-NER and DistilBERT-SST2 sentiment classification (~1s CPU execution for 0 tokens).
- Instruction stripping for sentiment and NER cloud fallbacks.

---

## [0.7.0] - 2026-07-12

### Added
- 150-second deadline cutoff (`LOCAL_CUTOFF_SECONDS`) to flip in-progress local work to cloud before harness timeout.
- Cheapest-first local queue scheduling.

---

## [0.5.0] - 2026-07-12

### Added
- Adaptive hybrid mode: background model preload and warm-up speed calibration (measuring tokens/second before activating local execution).
- Global 300-second local time budget with cloud fallback.

---

## [0.2.0] - 2026-07-11

### Added
- Initial hybrid local inference with bundled Qwen2.5-3B-Instruct GGUF via `llama-cpp-python`.

---

## [0.1.0] - 2026-07-11

### Added
- Initial cloud-only release with lexical regex router, terse system prompts, and primary/fallback model routing across Fireworks AI.
