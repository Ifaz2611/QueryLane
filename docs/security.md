# Security Notes

## Prompt injection from task payloads

Task prompts are untrusted input: the harness (or anyone who can write
`tasks.json`) controls the full text sent to the cloud model. A malicious
payload could try to make the model ignore its system prompt, exfiltrate data,
or emit content outside the expected format.

Mitigations already in place:

- **No tools, no side effects.** The agent only reads `tasks.json` and writes
  `results.json`. There is no shell, no file access beyond those paths, no
  network beyond the configured Fireworks endpoint, and no agentic loop — a
  successful injection can at worst produce a wrong *answer string*.
- **Terse system prompts + `temperature=0`** reduce but do not eliminate
  instruction-following risk; treat model output as data, never execute it.
- **`pal.safe_eval`** is an AST whitelist (`+ - * / // % **`, `ceil floor
  round abs min max sqrt`), rejects bools, non-finite values, huge exponents
  (>100), long expressions (>300 chars), and out-of-range results (>1e15).
  It compiles with `{"__builtins__": {}}` — math payloads cannot reach
  `os`/`open`/etc. Fuzzed in `tests/test_pal_fuzz.py`.
- **`strip_instruction()`** only unwraps exact sentiment/NER response
  wrappers; it does not execute or interpret payload content.
- **Secrets hygiene.** `FIREWORKS_API_KEY` comes from the environment only;
  `scripts/check_secrets.py` (pre-commit + CI) blocks `.env`, `*.gguf`,
  `*.onnx`, `testio/output*/`, `eval/results_*.json`, and `source/` from
  being committed. Never log or echo the key.
- **Container hardening.** The Docker images run as uid 10001 (non-root) on
  pinned, hash-verified lockfiles.

Residual risk: a crafted prompt can still degrade *answer quality* (e.g. make
the cloud model refuse or ramble within `max_tokens`). That is an accuracy
problem, bounded by the 80% gate and the empty-answer fallback — not a
code-execution or data-exfiltration path.

## Why routing is lexical, not model-based

`agent/router.py` classifies with regexes and keyword cues — zero tokens, zero
latency, fully deterministic. This is a deliberate security and cost choice:

1. **A model-based router is itself an injection surface.** A small classifier
   model reading the same untrusted prompt could be steered into the wrong
   category (e.g. force a logic puzzle into `factual`), and its decision would
   be probabilistic and hard to audit. Regexes are inspectable: every category
   has explicit cue patterns with word boundaries, and `route_with_confidence`
   exposes the score.
2. **Determinism enables regression testing.** Lexical rules give stable
   PASS/FAIL behavior under `eval/router_paraphrases_v1.json`; a learned
   router's drift would be invisible until eval.
3. **Cheap lanes self-gate.** Specialists and the local model return `""` on
   doubt, so a misclassification costs tokens (escalation to cloud), never
   correctness — the accuracy gate is protected by construction.
4. **`ROUTER_MIN_CONFIDENCE` (default 0.70)** sends ambiguous prompts straight
   to cloud, so edge-case phrasing can't sneak into a specialist/local lane.

Trade-off accepted: lexical routing has lower ceiling coverage than a learned
router — that's why `factual` is the safe fallback and paraphrase suites exist
to measure (not hide) the gap.
