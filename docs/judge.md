# Evaluation Judge

QueryLane's dev evaluation (`eval/judge.py`) answers `eval/tasks_dev.json`
through the real agent pipeline, then grades each answer. Grading is
deterministic-first: mechanical constraints (labels, sentence/word counts,
entity sets, exact numbers, code structure) come from `eval/checks_dev.json`
via `eval/deterministic.py`; only tasks those checks cannot decide (mostly the
code categories) go to the Fireworks LLM judge.

```bash
set -a; source .env; set +a
python -m eval.judge                                  # hybrid grader, 3 runs
python -m eval.judge --answers eval/results_dev.json  # re-judge saved answers
python -m eval.judge --grader deterministic           # offline, no API key
python -m eval.judge --grader llm --runs 5            # historical all-LLM mode
python -m eval.judge --category math                  # one category only
python -m eval.judge --judge-model <id-in-ALLOWED_MODELS>
```

Exit code is `0` when the mean accuracy across runs is `>= 80%` (the accuracy
gate), `1` otherwise. Judge tokens are dev spend only and never counted against
the agent's Fireworks token score.

## Grader modes (`--grader`)

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Deterministic check decides when `authoritative`; the LLM judge handles the rest. |
| `deterministic` | Deterministic checks only. Undecided tasks fail. Works offline when re-grading saved answers. |
| `llm` | Every task goes to the LLM judge — the historical behavior, kept for comparison. |

## Exact judge prompt

`eval/judge.py` sends one call per undecided task, `temperature=0`,
`max_tokens=1500`, to the judge model. The judge model defaults to the first
`ALLOWED_MODELS` entry containing `minimax` (strongest generalist), or
`ALLOWED_MODELS[0]`; `--judge-model` must name a model already in
`ALLOWED_MODELS`.

System prompt (`_JUDGE_SYSTEM`):

```
You are a strict grader. Decide if the answer correctly accomplishes the
task and matches the key points. Formatting differences are fine; factual or
logical errors, missing required parts, and violated format constraints (like
sentence counts) are failures. First line: PASS or FAIL. Second line: one
short reason.
```

User template (`_JUDGE_TEMPLATE`):

```
Task given to the model:
{prompt}

Key points a correct answer must satisfy:
{expect}

Answer to grade:
{answer}
```

Verdict parsing: the first line is uppercased, stripped of non-letters
(tolerating markdown like `**PASS**`), and passes iff it starts with `PASS`.
The second line is the recorded reason. An empty answer is auto-failed
without a judge call.

## Deterministic check types

`eval/deterministic.py` supports `check.type` values: `constraints`
(sentence/word/line counts, required/forbidden terms and numbers, ordering,
`final_only` scoping), `sentiment_label`, `entities`, `number`, and
`requires_code` (AST-parse check; correctness still needs the judge). Checks
live in `eval/checks_dev.json` keyed by `task_id`; inline `check` objects in
`tasks_dev.json` win. Tasks with no check, an unknown type, or a handler error
return `UNDECIDED` and fall through to the LLM judge.

## Known failure modes

- **Judge nondeterminism.** Even at `temperature=0`, provider-side
  nondeterminism flips borderline verdicts between runs. Mitigation: run with
  `--runs N` (default 3); the dashboard verdict is the majority vote (ties
  fail), and `eval/judge_variance.json` lists unstable tasks.
- **Borderline-format answers.** The judge tolerates formatting differences
  but is an LLM: a near-miss on sentence counts or a missing key point can
  flip between PASS/FAIL across runs. Prefer deterministic checks in
  `checks_dev.json` wherever the constraint is mechanical — it removes judge
  variance from that task entirely.
- **Markdown-decorated verdicts** (`**PASS**`, `PASS.`) are normalized by the
  parser, but a judge that buries the verdict on a later line fails the parse
  (`verdict.startswith("PASS")` on line 1 only).
- **Weak judge model.** The default prefers `minimax` in `ALLOWED_MODELS`;
  pointing `--judge-model` at a weaker allowed model raises flip rates and can
  bias scores. Keep the judge model fixed when comparing SCORES.md entries.
- **Empty answers are free fails** (`"empty answer"`, no judge call) — check
  agent logs before blaming the judge.
- **`--grader deterministic` with saved answers** marks every undecided task
  as failed, so it is a lower bound on accuracy, not comparable to hybrid
  numbers.
