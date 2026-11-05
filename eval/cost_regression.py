"""Offline token-cost regression harness for the dev set.

Real Fireworks usage depends on the provider's tokenizer and on how verbose a
particular model happens to be, so it cannot serve as a stable CI assertion.
This harness instead models the part of cost that QueryLane controls:

- the exact system prompt and stripped user prompt sent to each cloud call,
- how many calls each lane configuration makes, and
- a fixed per-category completion baseline (the expected answer length).

Prompt tokens are estimated as ``ceil(characters / 4)``. That is deliberately a
model, not a billing quote: the point is to catch a lost specialist, a prompt
that grew, or a lane that silently stopped answering, all of which move the
call count and the prompt volume. ``eval/token_budgets.json`` records the
result for every supported configuration; ``tests/test_token_budget.py`` fails
when a configuration exceeds its recorded budget.

Usage:
    python -m eval.cost_regression              # measure and compare to budget
    python -m eval.cost_regression --update     # intentionally re-record budgets
"""

import argparse
import contextlib
import json
import math
import os
import time
from collections import Counter
from unittest import mock

from agent import local as local_model
from agent import main as main_module
from agent import obs, prompts, router, specialists
from agent.cloud import strip_instruction
from eval import deterministic

EVAL_DIR = os.path.dirname(__file__)
DEV_TASKS_PATH = os.path.join(EVAL_DIR, "tasks_dev.json")
BUDGETS_PATH = os.path.join(EVAL_DIR, "token_budgets.json")

CONFIGURATIONS = ("cloud-only", "specialists", "hybrid")
LOCAL_CATEGORIES = {"math", "factual", "ner"}

# Expected completion length per category, in tokens. These mirror the tuned
# system prompts and max_tokens caps in agent/prompts.py. They are intentionally
# conservative enough to notice a verbose-answer regression but not so tight
# that unrelated wording changes break the test.
COMPLETION_TOKENS = {
    "factual": 48,
    "math": 20,
    "sentiment": 36,
    "summarization": 62,
    "ner": 58,
    "code_debug": 90,
    "logic": 95,
    "code_gen": 90,
}

_ESTIMATOR = (
    "ceil(characters/4) for the exact system+user prompt plus a fixed "
    "per-category completion baseline; specialists and local answers are "
    "zero-token by construction"
)


def estimate_tokens(text: str) -> int:
    """Deterministic prompt-token estimate (about four characters per token)."""
    return max(1, math.ceil(len(str(text)) / 4))


def load_tasks(path: str = DEV_TASKS_PATH) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_budgets(path: str = BUDGETS_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------- reference lanes

def _format_number(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _local_math_answer(task: dict) -> str:
    check = deterministic.check_for(task) or {}
    value = _format_number(check.get("value", 0))
    units = check.get("unit_any", [])
    unit = units[0] if units else ""
    if unit == "$":
        return f"Answer: ${value}"
    if unit:
        return f"Answer: {value} {unit}"
    return f"Answer: {value}"


def _ner_answer(task: dict) -> str:
    check = deterministic.check_for(task) or {}
    lines = []
    for entity in check.get("entities", []):
        types = entity.get("types") or ["misc"]
        lines.append(f"{entity.get('name', '')}: {str(types[0]).title()}")
    return "\n".join(lines)


def _sentiment_answer(task: dict) -> str:
    check = deterministic.check_for(task) or {}
    labels = check.get("labels") or ["neutral"]
    return (
        f"{str(labels[0]).capitalize()}. The review contains both the praise "
        "and the complaint described in the task."
    )


def _lane_answers(tasks: list) -> tuple[dict, dict]:
    """Build prompt-keyed canned answers for the zero-token lanes."""
    specialist_answers = {}
    local_answers = {}
    for task in tasks:
        prompt = task.get("prompt", "")
        category = task.get("category", "")
        if category == "sentiment":
            specialist_answers[prompt] = _sentiment_answer(task)
        elif category == "ner":
            specialist_answers[prompt] = _ner_answer(task)
            local_answers[prompt] = specialist_answers[prompt]
        elif category == "math":
            local_answers[prompt] = _local_math_answer(task)
        elif category == "factual":
            local_answers[prompt] = task.get("expect", "Mocked local answer.")
    return specialist_answers, local_answers


# ------------------------------------------------------------------ fake cloud

class SimulatedCloud:
    """CloudClient-compatible stub that models provider token accounting."""

    def __init__(self):
        self._total = 0
        self._calls = 0
        self._category_tokens = Counter()
        self._category_calls = Counter()

    def answer(self, category: str, prompt: str) -> str:
        spec = prompts.SPECS[category]
        user = strip_instruction(category, prompt)
        prompt_tokens = estimate_tokens(spec.system) + estimate_tokens(user)
        completion_tokens = min(
            COMPLETION_TOKENS.get(category, 60), spec.max_tokens
        )
        total = prompt_tokens + completion_tokens
        self._total += total
        self._calls += 1
        self._category_tokens[category] += total
        self._category_calls[category] += 1
        return f"Mocked cloud answer for {category}."

    def total_tokens(self) -> int:
        return self._total

    def call_count(self) -> int:
        return self._calls

    def tokens_for_category(self, category: str) -> int:
        return self._category_tokens.get(category, 0)

    def calls_for_category(self, category: str) -> int:
        return self._category_calls.get(category, 0)

    def tokens_by_category(self) -> dict:
        return dict(self._category_tokens)

    def calls_by_category(self) -> dict:
        return dict(self._category_calls)

    def retry_count(self) -> int:
        return 0

    def empty_response_count(self) -> int:
        return 0

    def budget_exceeded_count(self) -> int:
        return 0


@contextlib.contextmanager
def _configuration(name: str, tasks: list):
    """Patch lane availability for one measured configuration."""
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name!r}")

    local_enabled = name == "hybrid"
    specialists_enabled = name in ("specialists", "hybrid")
    specialist_answers, local_answers = _lane_answers(tasks)
    local_categories = LOCAL_CATEGORIES if local_enabled else set()

    def specialist_answer(prompt):
        return specialist_answers.get(prompt, "")

    def local_answer(category, prompt):
        if category not in local_categories:
            return ""
        return local_answers.get(prompt, "")

    patches = [
        mock.patch.object(router, "LOCAL_CATEGORIES", set(local_categories)),
        mock.patch.object(local_model, "available", return_value=local_enabled),
        mock.patch.object(local_model, "wait_ready", return_value=local_enabled),
        mock.patch.object(
            local_model,
            "category_enabled",
            side_effect=lambda category: category in local_categories,
        ),
        mock.patch.object(local_model, "answer", side_effect=local_answer),
        mock.patch.object(
            specialists, "available", side_effect=lambda kind: specialists_enabled
        ),
        mock.patch.object(
            specialists, "ner_enabled", return_value=specialists_enabled
        ),
        mock.patch.object(
            specialists, "sentiment_answer", side_effect=specialist_answer
        ),
        mock.patch.object(specialists, "ner_answer", side_effect=specialist_answer),
        mock.patch.object(main_module, "_START", time.time()),
    ]

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield


def run_configuration(name: str, tasks: list = None) -> dict:
    """Measure one lane configuration over the dev set, sequentially."""
    tasks = tasks if tasks is not None else load_tasks()
    obs.metrics.reset()
    with _configuration(name, tasks):
        cloud = SimulatedCloud()
        entries = [main_module.answer_task(cloud, task) for task in tasks]

    return {
        "configuration": name,
        "tasks": len(tasks),
        "tokens": cloud.total_tokens(),
        "calls": cloud.call_count(),
        "avg_tokens_per_task": round(cloud.total_tokens() / max(len(tasks), 1), 2),
        "lanes": dict(Counter(entry["routed"] for entry in entries)),
        "tokens_by_category": cloud.tokens_by_category(),
        "calls_by_category": cloud.calls_by_category(),
    }


def run_all(tasks: list = None) -> dict:
    tasks = tasks if tasks is not None else load_tasks()
    return {name: run_configuration(name, tasks) for name in CONFIGURATIONS}


def write_budgets(results: dict, path: str = BUDGETS_PATH) -> dict:
    """Record measured results as the new regression budgets."""
    payload = {
        "schema_version": 1,
        "estimator": _ESTIMATOR,
        "configurations": {
            name: {
                "budget_tokens": result["tokens"],
                "calls": result["calls"],
                "tasks": result["tasks"],
                "avg_tokens_per_task": result["avg_tokens_per_task"],
                "lanes": result["lanes"],
                "tokens_by_category": result["tokens_by_category"],
            }
            for name, result in results.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return payload


def check_budgets(results: dict, budgets: dict) -> list:
    """Return ``[(configuration, measured, budget, delta), ...]`` overruns."""
    overruns = []
    recorded = budgets.get("configurations", {})
    for name, result in results.items():
        budget = recorded.get(name, {}).get("budget_tokens")
        if budget is None:
            overruns.append((name, result["tokens"], None, None))
            continue
        if result["tokens"] > int(budget):
            overruns.append(
                (name, result["tokens"], int(budget), result["tokens"] - int(budget))
            )
    return overruns


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="intentionally re-record eval/token_budgets.json",
    )
    parser.add_argument("--json", help="write measured results to this path")
    args = parser.parse_args(argv)

    results = run_all()
    print(f"{'configuration':<14} {'tasks':>5} {'calls':>6} {'tokens':>8} {'lanes'}")
    for name in CONFIGURATIONS:
        result = results[name]
        print(
            f"{name:<14} {result['tasks']:>5} {result['calls']:>6} "
            f"{result['tokens']:>8} {result['lanes']}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"measurements written to: {args.json}")

    if args.update:
        write_budgets(results)
        print(f"budgets updated: {BUDGETS_PATH}")
        return 0

    try:
        budgets = load_budgets()
    except FileNotFoundError:
        print(f"no budgets at {BUDGETS_PATH}; run with --update")
        return 1

    overruns = check_budgets(results, budgets)
    if overruns:
        for name, measured, budget, delta in overruns:
            if budget is None:
                print(f"OVER BUDGET {name}: {measured} tokens (no recorded budget)")
            else:
                print(
                    f"OVER BUDGET {name}: {measured} tokens "
                    f"(budget {budget}, +{delta})"
                )
        return 1

    for name in CONFIGURATIONS:
        budget = budgets["configurations"][name]["budget_tokens"]
        print(f"OK {name}: {results[name]['tokens']} <= {budget} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
