"""Dev-only evaluation harness. Never ships in the image.

Answers eval/tasks_dev.json through the real agent pipeline, then grades each
answer. Grading is deterministic-first: labels, sentence/word counts, entity
sets, exact numbers, and other mechanical constraints come from
``eval/checks_dev.json``; only tasks the deterministic checks cannot decide (the
code categories) go to the Fireworks LLM judge. That removes judge variance from
most of the dev set and saves judge tokens.

The judge can also be run several times (``--runs``) so any remaining
nondeterminism is measured instead of being hidden by a single sample.

Usage:
    set -a; source .env; set +a
    python -m eval.judge                       # deterministic + LLM, 3 runs
    python -m eval.judge --answers eval/results_dev.json   # re-judge saved answers
    python -m eval.judge --answers eval/results_dev.json --grader deterministic
                                                 # offline re-grade, no API key
    python -m eval.judge --grader llm --runs 5   # historical all-LLM behavior
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from agent.cloud import CloudClient, response_text, usage_totals
from agent.main import answer_task
from eval import deterministic
from eval.dashboard import build_dashboard

DEV_TASKS = os.path.join(os.path.dirname(__file__), "tasks_dev.json")
DEFAULT_RUNS = 3
DEFAULT_VARIANCE_PATH = os.path.join(os.path.dirname(__file__), "judge_variance.json")

_JUDGE_SYSTEM = (
    "You are a strict grader. Decide if the answer correctly accomplishes the "
    "task and matches the key points. Formatting differences are fine; factual "
    "or logical errors, missing required parts, and violated format constraints "
    "(like sentence counts) are failures. First line: PASS or FAIL. Second "
    "line: one short reason."
)

_JUDGE_TEMPLATE = """Task given to the model:
{prompt}

Key points a correct answer must satisfy:
{expect}

Answer to grade:
{answer}"""


class Judge:
    def __init__(self, model: str = ""):
        self._client = OpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
            timeout=30.0,
        )
        allowed = [
            m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()
        ]
        if not allowed:
            raise ValueError("ALLOWED_MODELS must name at least one model")
        if model:
            if model not in allowed:
                raise ValueError(
                    f"judge model {model!r} is not in ALLOWED_MODELS: {allowed}"
                )
            self.model = model
        else:
            # Prefer the strongest generalist as judge by default.
            self.model = next(
                (m for m in allowed if "minimax" in m.lower()), allowed[0]
            )
        self._lock = threading.Lock()
        self.tokens = 0

    def grade(self, prompt: str, expect: str, answer: str):
        if not answer.strip():
            return False, "empty answer"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _JUDGE_TEMPLATE.format(
                        prompt=prompt, expect=expect, answer=answer
                    ),
                },
            ],
            temperature=0,
            max_tokens=1500,
        )
        usage_total = usage_totals(getattr(response, "usage", None))["total"]
        if usage_total:
            with self._lock:
                self.tokens += usage_total
        text = response_text(response)
        if not text:
            return False, "empty judge response"
        lines = text.splitlines()
        first = lines[0].strip().upper() if lines else ""
        # Tolerate markdown decoration such as "**PASS**".
        verdict = re.sub(r"[^A-Z]", "", first)
        reason = lines[1].strip() if len(lines) > 1 else ""
        return verdict.startswith("PASS"), reason


def answer_all(tasks):
    cloud = CloudClient.from_env()
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda t: answer_task(cloud, t), tasks))
    return answers, cloud.total_tokens(), cloud.call_count()


def _det_verdicts(tasks, answers):
    by_id = {entry["task_id"]: entry for entry in answers}
    return {
        task["task_id"]: deterministic.evaluate(
            task, by_id.get(task["task_id"], {}).get("answer", "")
        )
        for task in tasks
    }


def _grade_once(judge, det_verdicts, by_id, entries, grader):
    """Grade every answer once and return ``{task_id: result}``.

    ``result`` carries the verdict, reason, and provenance so the variance
    report can show whether a flip came from the LLM judge or from an
    actual answer change.
    """

    def grade(entry):
        task = by_id[entry["task_id"]]
        det = det_verdicts[entry["task_id"]]

        if grader == "llm":
            passed, reason = judge.grade(task["prompt"], task["expect"], entry["answer"])
            return entry, passed, reason, "llm"

        if grader == "deterministic":
            if det.authoritative:
                return entry, det.passed, det.reason, "deterministic"
            return entry, False, det.reason or "no deterministic check", "undecided"

        # hybrid: deterministic decides when authoritative, LLM handles the rest
        if det.authoritative:
            return entry, det.passed, det.reason, "deterministic"
        passed, reason = judge.grade(task["prompt"], task["expect"], entry["answer"])
        return entry, passed, reason, "llm"

    with ThreadPoolExecutor(max_workers=4) as pool:
        graded = list(pool.map(grade, entries))

    return {
        entry["task_id"]: {"passed": passed, "reason": reason, "source": source}
        for entry, passed, reason, source in graded
    }


def summarize_variance(task_runs: dict, tasks, runs: int) -> dict:
    """Summarize per-task and per-run verdicts.

    ``task_runs`` maps task_id -> list of ``{"passed", "reason", "source"}`` in
    run order. The return value is JSON-ready and is also embedded in the
    dashboard.
    """
    task_order = [str(task["task_id"]) for task in tasks]
    categories = {str(task["task_id"]): task.get("category", "") for task in tasks}

    per_task = {}
    unstable = []
    deterministic_tasks = 0
    llm_tasks = 0
    undecided_tasks = 0

    for task_id in task_order:
        outcomes = task_runs.get(task_id, [])
        passed_count = sum(1 for item in outcomes if item["passed"])
        total = len(outcomes)
        sources = sorted({item.get("source", "llm") for item in outcomes})
        source = sources[0] if len(sources) == 1 else "mixed"

        if source == "deterministic":
            deterministic_tasks += 1
        elif source == "undecided":
            undecided_tasks += 1
        else:
            llm_tasks += 1

        stable = total > 0 and passed_count in (0, total)
        item = {
            "task_id": task_id,
            "category": categories.get(task_id, ""),
            "passes": passed_count,
            "runs": total,
            "stable": stable,
            "source": source,
        }
        per_task[task_id] = item
        if not stable:
            unstable.append(item)

    accuracies = []
    for run_index in range(max(1, runs)):
        passed = sum(
            1
            for task_id in task_order
            if run_index < len(task_runs.get(task_id, []))
            and task_runs[task_id][run_index]["passed"]
        )
        accuracies.append(100.0 * passed / len(task_order) if task_order else 0.0)

    mean = sum(accuracies) / len(accuracies) if accuracies else 0.0
    stdev = (
        (sum((value - mean) ** 2 for value in accuracies) / len(accuracies)) ** 0.5
        if accuracies
        else 0.0
    )

    return {
        "runs": max(1, runs),
        "deterministic_tasks": deterministic_tasks,
        "llm_tasks": llm_tasks,
        "undecided_tasks": undecided_tasks,
        "accuracy_per_run": [round(value, 2) for value in accuracies],
        "accuracy_mean": round(mean, 2),
        "accuracy_stdev": round(stdev, 2),
        "accuracy_min": round(min(accuracies), 2) if accuracies else 0.0,
        "accuracy_max": round(max(accuracies), 2) if accuracies else 0.0,
        "stable_tasks": len(task_order) - len(unstable),
        "unstable_tasks": unstable,
        "per_task": per_task,
    }


def _majority_verdicts(task_runs: dict) -> dict:
    """Collapse multiple runs into one dashboard verdict (ties fail)."""
    verdicts = {}
    for task_id, outcomes in task_runs.items():
        passed = sum(1 for item in outcomes if item["passed"])
        verdicts[task_id] = passed * 2 > len(outcomes) if outcomes else False
    return verdicts


def _print_variance(summary: dict) -> None:
    runs = summary["runs"]
    print()
    print(f"grader variance ({runs} run{'s' if runs != 1 else ''})")
    print(
        f"  deterministic verdicts: {summary['deterministic_tasks']} task(s); "
        f"LLM-judged: {summary['llm_tasks']} task(s)"
        + (
            f"; undecided: {summary['undecided_tasks']} task(s)"
            if summary.get("undecided_tasks")
            else ""
        )
    )
    if runs == 1:
        print("  single run; pass --runs N to measure flip rate")
        return
    per_run = " ".join(f"{value:.1f}%" for value in summary["accuracy_per_run"])
    print(f"  accuracy per run: {per_run}")
    print(
        f"  mean {summary['accuracy_mean']:.1f}%  "
        f"stdev {summary['accuracy_stdev']:.1f}  "
        f"range {summary['accuracy_min']:.1f}-{summary['accuracy_max']:.1f}%"
    )
    unstable = summary["unstable_tasks"]
    if not unstable:
        print("  all task verdicts were stable across runs")
        return
    print(f"  unstable: {len(unstable)} task(s)")
    for item in unstable:
        print(
            f"    {item['task_id']:<10} {item['passes']}/{item['runs']} "
            f"[{item['category']}] ({item['source']})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", help="re-judge a saved answers file")
    parser.add_argument(
        "--save", default="eval/results_dev.json", help="where to save answers"
    )
    parser.add_argument("--category", help="only run one category")
    parser.add_argument(
        "--grader",
        choices=("hybrid", "deterministic", "llm"),
        default="hybrid",
        help=(
            "hybrid (default): deterministic checks first, LLM only when "
            "undecided; deterministic: offline when re-grading saved answers, "
            "undecided tasks fail; llm: historical all-LLM behavior"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"judge passes to run (default {DEFAULT_RUNS}); variance is reported",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="exact model id for the LLM judge (must be in ALLOWED_MODELS)",
    )
    parser.add_argument(
        "--out-md", default="eval/dashboard.md", help="path to write markdown dashboard"
    )
    parser.add_argument(
        "--out-json", default="eval/dashboard.json", help="path to write JSON dashboard"
    )
    parser.add_argument(
        "--out-variance",
        default=DEFAULT_VARIANCE_PATH,
        help="path to write the multi-run variance report",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    with open(DEV_TASKS, encoding="utf-8") as fh:
        tasks = json.load(fh)
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]

    if args.answers:
        with open(args.answers, encoding="utf-8") as fh:
            saved = json.load(fh)
        saved_by_id = {
            str(item.get("task_id", "")): item
            for item in saved
            if isinstance(item, dict)
        }
        # Keep one row per task in dev-set order, even when the saved file is
        # incomplete; a missing answer is graded as empty rather than silently
        # shrinking the denominator.
        answers = [
            {
                "task_id": task["task_id"],
                "answer": saved_by_id.get(task["task_id"], {}).get("answer", ""),
                "category": task.get("category", ""),
            }
            for task in tasks
        ]
        agent_tokens, agent_calls = None, None
    else:
        answers, agent_tokens, agent_calls = answer_all(tasks)
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(answers, fh, ensure_ascii=False, indent=1)

    by_id = {t["task_id"]: t for t in tasks}
    det_verdicts = _det_verdicts(tasks, answers)

    needs_llm = args.grader == "llm" or (
        args.grader == "hybrid"
        and any(not verdict.authoritative for verdict in det_verdicts.values())
    )
    judge = None
    if needs_llm:
        try:
            judge = Judge(args.judge_model)
        except (KeyError, ValueError) as exc:
            print(f"bad judge configuration: {exc}", file=sys.stderr, flush=True)
            return 1
        print(f"judge model: {judge.model}", flush=True)
    else:
        print("deterministic-only grading: no judge model needed", flush=True)

    task_runs = {task["task_id"]: [] for task in tasks}
    representative = {}
    for run_index in range(args.runs):
        run_results = _grade_once(
            judge, det_verdicts, by_id, answers, args.grader
        )
        for task_id, result in run_results.items():
            task_runs[task_id].append(result)
        if run_index == 0:
            representative = run_results

    # Print one representative run, then the variance report.
    per_category = {}
    for entry in answers:
        task = by_id[entry["task_id"]]
        result = representative.get(entry["task_id"], {})
        passed = bool(result.get("passed"))
        cat = task.get("category", "")
        per_category.setdefault(cat, []).append(passed)
        mark = "PASS" if passed else "FAIL"
        source = result.get("source", "")
        print(f"{mark} {entry['task_id']} [{cat}] ({source}) {result.get('reason', '')}")

    print()
    total_pass, total = 0, 0
    for cat in sorted(per_category):
        outcomes = per_category[cat]
        total_pass += sum(outcomes)
        total += len(outcomes)
        print(f"{cat:15s} {sum(outcomes)}/{len(outcomes)}")

    variance = summarize_variance(task_runs, tasks, args.runs)
    _print_variance(variance)

    mean_pct = variance["accuracy_mean"]
    print(f"{'OVERALL (mean)':15s} {mean_pct:.1f}%  gate is 80%")
    if args.runs > 1:
        print(
            f"{'per-run':15s} "
            + " ".join(f"{value:.1f}%" for value in variance["accuracy_per_run"])
        )
    if agent_tokens is not None:
        print(f"agent fireworks tokens: {agent_tokens} across {agent_calls} calls")
    print(f"judge tokens (dev spend only): {judge.tokens if judge else 0}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_variance)), exist_ok=True)
    with open(args.out_variance, "w", encoding="utf-8") as fh:
        json.dump(variance, fh, ensure_ascii=False, indent=1)
    print(f"variance report saved to: {args.out_variance}")

    # Generate and display the per-category accuracy/token dashboard.
    verdicts = _majority_verdicts(task_runs)
    report = build_dashboard(
        tasks,
        answers,
        graded_verdicts=verdicts,
        judge_variance=variance,
    )
    print("\n" + report.format_table())
    report.save_markdown(args.out_md)
    report.save_json(args.out_json)
    print(f"\nDashboard saved to: {args.out_md} and {args.out_json}")

    return 0 if mean_pct >= 80.0 else 1


if __name__ == "__main__":
    sys.exit(main())
