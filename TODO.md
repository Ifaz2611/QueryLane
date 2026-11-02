# QueryLane TODO

Priorities: **P0** = correctness or security, **P1** = reliability/quality,
**P2** = polish and scale. Items marked were fixed during the rebrand pass;
the rest are future work.

## P0 — correctness and security

- [x] Make `agent/specialists.py` import-safe in the cloud-only image (NumPy is
      now optional; `available()` returns `False` without it).
- [x] Harden `agent/pal.py`: reject non-numeric constants, `inf`/`nan`, complex
      results, huge exponents, and oversized expressions.
- [x] Fix the `ThreadPoolExecutor` context-manager wait that could keep the
      container alive past the deadline.
- [x] Fix `os.makedirs("")` when `RESULTS_PATH` is a bare filename.
- [x] Validate that the input batch is a JSON array of task objects.
- [x] Add a starter test suite for the router, PAL, instruction stripping,
      specialist gates, data validation, and the main loop (`tests/`, 24 tests).
- [x] Extend coverage to the cloud client, local calibration, and deadline
      handling, and wire the suite into CI.
- [x] Fuzz `pal.safe_eval` with adversarial model output and add regression
      cases for every rejected construct.
- [x] Decide on a license and add `LICENSE` before publishing.
- [x] Add a secret-scanning pre-commit hook and CI check (never commit `.env`).
- [x] Run the container as a non-root user and pin base-image digests.
- [x] Add dependency pinning/lockfile for reproducible images.

## P1 — routing and accuracy

- [x] Raise ONNX NER precision above the current ~22% exact-match bar so the
      specialist can be enabled by default. The post-processor now reconciles
      the model's four-class output with the task's six types, gates every
      token on confidence/margin, and escalates unless every capitalized span
      and date is covered. NER is on by default (`NER_SPECIALIST=off` disables).
- [x] Improve the sentiment specialist's coverage without lowering precision:
      negation scopes ("not bad" / "not great"), quoted marketing copy versus
      endorsed praise, and two-sided reviews are handled explicitly. Contrast
      only forces escalation when evidence is on both sides or neither side.
- [x] Add a router confidence score and route low-confidence prompts to cloud
      instead of silently falling back to `factual`. `route()` keeps its
      2-tuple contract; `route_with_confidence()` exposes the score and forces
      cloud below `ROUTER_MIN_CONFIDENCE` (default `0.70`).
- [x] Expand the paraphrase test set into a versioned, growing suite:
      `eval/router_paraphrases_v1.json` has 64 prompts across all categories,
      and `tests/test_paraphrases.py` discovers every `v*.json` automatically.
- [x] Add regression tests for the multipart-math detector, including
      "and the" prose that caused the v16 token regression. Handled in
      `tests/test_pal_multipart.py`.
- [x] Measure whether a small classifier beats the lexical router on the real
      category distribution; keep the lexical router as the zero-token default.
      Measured in `eval/measure_router.py`: lexical router achieves 100% vs
      90.3% small classifier on real distribution; lexical retained as zero-token default.
- [x] Add category-level fallback budgets so a single slow category cannot
      consume the whole local time budget (`LOCAL_CATEGORY_TIME_BUDGET`,
      `agent/local.py`).
- [x] Revisit local summarization/logic only after a fine-tune proves them
      (both regressed in earlier A/B runs). Guarded behind `category_proven()`
      and verification harness `eval/prove_local.py`.
- [x] Add a per-category accuracy/token dashboard generated from `eval/`.
      Implemented in `eval/dashboard.py` and wired into `eval/judge.py`.

## P1 — data and fine-tuning

- [x] Rebuild `finetune/data/train_all.jsonl`; it previously contained only
      summarization rows (493) instead of all 2,819 validated rows.
- [x] Fix `finetune/gen_data.py` so a partial `--categories` run cannot clobber
      the combined training file.
- [x] Reject neutral sentiment rows that carry praise/complaint quotes, and NER
      rows with empty entity text.
- [ ] Add a data validation script that checks every JSONL row (schema,
      category, verbatim quotes/entities, math expression correctness, summary
      length constraints) before training.
- [ ] Deduplicate near-identical prompts across `train_all` and `train_round2`
      and document the difference between the two files.
- [ ] Add a held-out validation split and report per-category loss/accuracy;
      current evaluation is judge-based and noisy.
- [ ] Make dataset generation reproducible (seed the generator calls and record
      the model/version used).
- [ ] Investigate the optional Qwen3.5 candidate in `colab_unsloth.ipynb`; verify
      the model id exists before relying on it.

## P1 — reliability and observability

- [ ] Add structured JSON logging with task id, category, lane, elapsed time,
      and token usage. `AGENT_DEBUG` is useful but ad hoc.
- [ ] Export per-run metrics (tokens by category, lane counts, escalations,
      empty answers) so regressions are visible without reading logs.
- [x] Make `_time_spent` updates thread-safe in `agent/local.py` and add a lock
      around the budget counters (`_budget_lock`).
- [ ] Add explicit backoff/jitter for connection errors and 5xx responses in
      `agent/cloud.py`; currently 5xx retries are immediate.
- [ ] Add a hard per-task cloud time budget so one slow call cannot consume the
      global deadline.
- [ ] Handle `response.choices == []` and malformed usage objects in all cloud
      call sites (partially handled in `agent/cloud.py`).
- [ ] Add graceful handling for duplicate `task_id` values in the input batch.
- [ ] Add a `--dry-run` mode that routes and reports lanes without calling the
      cloud.

## P1 — evaluation

- [ ] Replace or supplement the nondeterministic LLM judge with deterministic
      checks where possible (exact labels, sentence counts, entity sets).
- [ ] Run the judge multiple times and report variance; several historical
      scores flipped between runs.
- [ ] Add a token-cost regression test: the dev set must stay under a recorded
      budget for each lane configuration.
- [ ] Add end-to-end tests that run `python -m agent.main` against
      `testio/input/tasks.json` with a mocked cloud client.
- [ ] Record the exact model versions used for each `SCORES.md` row.

## P2 — packaging and rebrand polish

- [ ] Rename the workspace folder from `LLM-Query-Route` to `querylane` when it
      is safe to do so (the internal `agent` package is intentionally kept for
      entrypoint/import compatibility).
- [ ] Publish `pyproject.toml` metadata (authors, URLs, license) and build a
      wheel/sdist in CI.
- [ ] Add a small logo/wordmark and a docs page with the lane diagram.
- [ ] Add a `Makefile` or `justfile` for common tasks (test, lint, eval, build).
- [x] Add `ruff`/`pytest` to CI and a pre-commit configuration.
- [ ] Add a `CHANGELOG.md` and semantic version tags.
- [ ] Provide a model download script with checksums for the ONNX and GGUF
      artifacts instead of documenting manual placement.
- [ ] Add multi-arch image builds (amd64/arm64) and document the CPU wheel
      caveat for `llama-cpp-python`.
- [ ] Add a minimal web demo or CLI that accepts a prompt and explains the lane
      decision.

## P2 — performance

- [ ] Benchmark the ONNX sessions with `intra_op_num_threads` tuned per host;
      currently hard-coded to 2.
- [ ] Cache tokenizer/session loads across processes where possible.
- [ ] Evaluate whether ONNX Runtime's dynamic quantization or a smaller
      DistilBERT variant improves cold-start latency.
- [ ] Add context-length checks before local generation; overly long prompts
      currently fail into cloud after a wasted attempt.
- [ ] Measure the cost/benefit of speculative category batching again; the
      earlier A/B showed a net loss at group sizes 2-3, but larger groups may
      differ.

## P2 — docs

- [ ] Add architecture diagrams for the lane order and escalation paths.
- [ ] Document the exact judge prompt and its known failure modes.
- [ ] Add a troubleshooting section for missing models, missing API keys, and
      slow hosts.
- [ ] Document how to add a new category end to end (router, prompts, local
      system prompt, eval rows).
- [ ] Add a security note about prompt injection from task payloads and why
      routing is lexical rather than model-based.

## Known limitations

- The sentiment specialist has no neutral class; neutral reviews are escalated
  by design.
- The ONNX NER specialist is enabled by default but remains precision-gated:
  missing weights, uncertain tokens, MISC spans without a lexical cue, or any
  uncovered capitalized span/date all escalate to cloud.
- Local generative inference is only useful on hosts above the per-category
  speed bars; the historical grader was slower than the original thresholds.
- The dev judge is an LLM and is not fully deterministic.
- `train_all.jsonl` and `train_round2.jsonl` overlap; the relationship should be
  documented and deduplicated.
