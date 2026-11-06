# Adding a New Category End to End

A category touches five layers: lexical router, cloud prompt spec, optional
local-lane support, eval fixtures, and tests. Work top to bottom; the router
change alone is enough to ship (the task just goes to cloud via `factual`-like
handling until a `Spec` exists — actually a `Spec` is required, so add both
before testing).

## 1. Router — `agent/router.py`

- Add the constant (e.g. `MY_CAT = "my_cat"`) and append it to `ALL_CATEGORIES`.
- Add a cue regex. Conventions: `re.IGNORECASE`, **word boundaries** on every
  alternation (the `uncomfortable` bug was a substring match firing inside a
  longer word), and keep prose-ambiguous cues out of `_STRONG` scores.
- Wire the signal in `_signals()`. Check **order matters**: instruction verbs
  (extract, fix, write, summarize) are tested before content cues that can
  appear inside quoted passages — add your check in the right position.
- Add an entry to `_CATEGORY_PRIORITY` (earlier wins ties) and `_STRONG`
  (confidence score). If the category is numeric-heavy, gate it with
  `_looks_numeric()` the way `logic`/`math` do.
- Decide the fallback: unknown prompts land on `factual` (confidence 0.30,
  always cloud). New categories should only win on positive evidence.

Verify: `python -c "from agent.router import classify, route_with_confidence;
print(classify('...'), route_with_confidence('...'))"`.

## 2. Cloud prompt — `agent/prompts.py`

Add a `Spec` to `SPECS` keyed by the new constant:

- `system`: terse — every system token is billed on every call. Spell out the
  exact output shape the judge expects (label line, sentence count, code-only).
- `max_tokens`: generous enough that a misrouted task is never truncated;
  `factual` carries 500 for exactly this reason.
- `effort`: keep `"none"` unless dev-eval shows the category needs reasoning.

If the answer needs post-processing, extend `local.normalize()` (it runs on
cloud answers too).

## 3. Local lane (optional) — `agent/local.py` + `agent/main.py`

Only if the category is proven accurate on-device:

- `_LOCAL_SYSTEM[cat]`: the local system prompt (few-shot turns go in
  `_LOCAL_FEWSHOT` — free locally, use them to steady small models).
- `_MAX_TOKENS[cat]`: output cap; math-style structured output should be short.
- `_CATEGORY_MIN_TPS[cat]`: the speed bar. Use `99.0` to keep the category
  cloud-only until proven.
- `category_enabled()` / `PROVEN_CATEGORIES`: `summarization` and `logic` are
  hard-blocked behind `PROVE_LOCAL_<CAT>` env flags — follow that pattern for
  anything risky.
- `answer_ok()`: add a plausibility gate. Cheap lanes must return `""` (or a
  failing check) to escalate — never guess.
- `main.py::_LOCAL_COST_ORDER`: add the category with a priority matching
  savings-per-second (math-PAL first, longest outputs last). Cloud-routed
  tasks are scheduled before local ones automatically.
- `LOCAL_CATEGORIES` env decides whether the router targets `local` at all;
  document the new value.

## 4. Eval fixtures — `eval/`

- `eval/tasks_dev.json`: add rows `{"task_id", "prompt", "expect",
  "category"}` covering the category's shapes and edge cases.
- `eval/checks_dev.json`: add a deterministic `check` wherever the rubric is
  mechanical (`constraints`, `sentiment_label`, `entities`, `number`,
  `requires_code`). Anything left unchecked falls to the LLM judge and adds
  variance — see [judge.md](judge.md).
- New check type? Add a handler in `eval/deterministic.py::_HANDLERS` that
  returns a tri-state `Verdict` (`authoritative=True` only when it fully
  decides).
- `eval/router_paraphrases_v1.json` (or a new versioned suite): add paraphrase
  rows so router regressions are measured, not anecdotal.

## 5. Tests — `tests/`

- `test_core.py` / `test_router_confidence.py`: classify + confidence cases,
  including a near-miss that must NOT classify as the new category.
- `test_local.py` if local behavior changed; `test_paraphrases.py` picks up
  new paraphrase rows automatically.
- `eval/spec_precision.py` measures prompt-spec precision offline — run it
  after touching `prompts.py`.

## Checklist before merging

- [ ] `pytest -v` and `ruff check .` clean
- [ ] `python -m eval.judge --category <cat>` at/above gate; `--runs 3` stable
- [ ] `python -m agent.main --dry-run` shows expected lane mix
- [ ] Token cost per task measured (`AGENT_DEBUG=1`) and recorded in SCORES.md
- [ ] Docs updated: README category list, `.env.example` if a new env var was
      added, `docs/lanes.md` lane table if cost profile changed
