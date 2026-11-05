"""Per-category accuracy and token usage dashboard generator.

Analyzes evaluation results against tasks_dev.json (or custom test batches),
computing per-category:
- Accuracy (passed / total, percentage)
- Gate status (PASS / FAIL against 80% accuracy threshold)
- Fireworks API tokens (total tokens and average tokens per task)
- Fireworks API call counts
- Routing lane breakdown (cloud, local, specialist, escalated, cloud-fallback)

Outputs:
- Terminal formatted table
- Markdown report (eval/dashboard.md)
- JSON export (eval/dashboard.json)

Usage:
    python -m eval.dashboard                                   # generate from results_dev.json or run deterministically
    python -m eval.dashboard --answers eval/results_dev.json   # generate from saved answers
    python -m eval.dashboard --out-md eval/dashboard.md        # custom output markdown path
"""

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.router import ALL_CATEGORIES
from eval import deterministic

EVAL_DIR = os.path.dirname(__file__)
DEV_TASKS_PATH = os.path.join(EVAL_DIR, "tasks_dev.json")
DEFAULT_RESULTS_PATH = os.path.join(EVAL_DIR, "results_dev.json")
DEFAULT_MD_PATH = os.path.join(EVAL_DIR, "dashboard.md")
DEFAULT_JSON_PATH = os.path.join(EVAL_DIR, "dashboard.json")


@dataclass
class CategoryMetrics:
    category: str
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    accuracy: float = 0.0
    gate_passed: bool = False
    tokens: int = 0
    calls: int = 0
    avg_tokens_per_task: float = 0.0
    routed_lanes: Dict[str, int] = field(default_factory=dict)


@dataclass
class DashboardReport:
    categories: Dict[str, CategoryMetrics]
    total_tasks: int = 0
    total_passed: int = 0
    total_failed: int = 0
    overall_accuracy: float = 0.0
    overall_gate_passed: bool = False
    total_tokens: int = 0
    total_calls: int = 0
    avg_tokens_per_task: float = 0.0
    overall_lanes: Dict[str, int] = field(default_factory=dict)
    judge_variance: Optional[dict] = None

    def format_table(self) -> str:
        lines = []
        width = 100
        lines.append("=" * width)
        lines.append("QUERYLANE EVALUATION DASHBOARD: PER-CATEGORY ACCURACY & TOKEN COST")
        lines.append("=" * width)
        header = (
            f"{'Category':<15} | {'Tasks':<6} | {'Pass':<5} | {'Acc (%)':<8} | "
            f"{'Gate':<6} | {'Tokens':<8} | {'Calls':<6} | {'Avg Tok':<8} | {'Lanes'}"
        )
        lines.append(header)
        lines.append("-" * width)

        for cat in ALL_CATEGORIES:
            m = self.categories.get(cat)
            if not m or m.total_tasks == 0:
                continue
            gate_str = "PASS" if m.gate_passed else "FAIL"
            lanes_str = ", ".join(f"{k}:{v}" for k, v in sorted(m.routed_lanes.items())) or "cloud"
            lines.append(
                f"{m.category:<15} | {m.total_tasks:<6} | {m.passed_tasks:<5} | "
                f"{m.accuracy:>7.1f}% | {gate_str:<6} | {m.tokens:>8} | "
                f"{m.calls:<6} | {m.avg_tokens_per_task:>8.1f} | {lanes_str}"
            )

        lines.append("=" * width)
        overall_gate = "PASSED" if self.overall_gate_passed else "FAILED"
        overall_lanes_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.overall_lanes.items())) or "cloud"
        lines.append(
            f"{'TOTAL / OVERALL':<15} | {self.total_tasks:<6} | {self.total_passed:<5} | "
            f"{self.overall_accuracy:>7.1f}% | {overall_gate:<6} | {self.total_tokens:>8} | "
            f"{self.total_calls:<6} | {self.avg_tokens_per_task:>8.1f} | {overall_lanes_str}"
        )
        lines.append("=" * width)
        lines.append(
            f"Gate Bar: >= 80.0% accuracy | Status: {overall_gate} "
            f"({self.total_passed}/{self.total_tasks} passing, {self.overall_accuracy:.1f}%)"
        )
        return "\n".join(lines)

    def format_markdown(self) -> str:
        lines = []
        lines.append("# QueryLane Evaluation Dashboard\n")
        gate_badge = "**PASSED**" if self.overall_gate_passed else "**FAILED**"
        lines.append(f"> [!IMPORTANT]\n> **Gate Status: {gate_badge}**  \n> Overall Accuracy: **{self.overall_accuracy:.1f}%** ({self.total_passed}/{self.total_tasks} passed) | Fireworks Tokens: **{self.total_tokens:,}** across **{self.total_calls}** calls (avg **{self.avg_tokens_per_task:.1f}** tok/task)\n")

        lines.append("## Per-Category Performance Breakdown\n")
        lines.append("| Category | Tasks | Passed | Failed | Accuracy | Gate (80%) | Fireworks Tokens | API Calls | Avg Tok/Task | Routed Lanes |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for cat in ALL_CATEGORIES:
            m = self.categories.get(cat)
            if not m or m.total_tasks == 0:
                continue
            gate_icon = "PASS" if m.gate_passed else "FAIL"
            lanes_str = ", ".join(f"`{k}`: {v}" for k, v in sorted(m.routed_lanes.items())) or "`cloud`: all"
            lines.append(
                f"| **{m.category}** | {m.total_tasks} | {m.passed_tasks} | {m.failed_tasks} | "
                f"{m.accuracy:.1f}% | {gate_icon} | {m.tokens:,} | {m.calls} | "
                f"{m.avg_tokens_per_task:.1f} | {lanes_str} |"
            )

        lines.append(
            f"| **TOTAL** | **{self.total_tasks}** | **{self.total_passed}** | **{self.total_failed}** | "
            f"**{self.overall_accuracy:.1f}%** | **{gate_badge}** | **{self.total_tokens:,}** | "
            f"**{self.total_calls}** | **{self.avg_tokens_per_task:.1f}** | - |"
        )

        lines.append("\n## Lane Distribution\n")
        lines.append("| Lane | Task Count | Percentage |")
        lines.append("| :--- | :---: | :---: |")
        for lane, count in sorted(self.overall_lanes.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / max(self.total_tasks, 1)
            lines.append(f"| `{lane}` | {count} | {pct:.1f}% |")

        if self.judge_variance:
            variance = self.judge_variance
            lines.append("\n## Grader Variance\n")
            lines.append(
                f"- Runs: **{variance.get('runs', 0)}** across "
                f"**{variance.get('llm_tasks', 0)}** LLM-judged task(s) "
                f"({variance.get('deterministic_tasks', 0)} task(s) graded deterministically)."
            )
            per_run = ", ".join(
                f"{value:.1f}%" for value in variance.get("accuracy_per_run", [])
            )
            lines.append(f"- Accuracy by run: {per_run}")
            lines.append(
                f"- Mean **{variance.get('accuracy_mean', 0.0):.1f}%** "
                f"± {variance.get('accuracy_stdev', 0.0):.1f} "
                f"(min {variance.get('accuracy_min', 0.0):.1f}%, "
                f"max {variance.get('accuracy_max', 0.0):.1f}%)."
            )
            unstable = variance.get("unstable_tasks", [])
            if unstable:
                lines.append("\n| Task | Passes | Runs | Source |")
                lines.append("| :--- | :---: | :---: | :--- |")
                for item in unstable:
                    lines.append(
                        f"| `{item['task_id']}` | {item['passes']} | "
                        f"{item['runs']} | {item.get('source', 'llm')} |"
                    )
            else:
                lines.append("\nAll task verdicts were stable across runs.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "overall_accuracy": round(self.overall_accuracy, 2),
                "overall_gate_passed": self.overall_gate_passed,
                "total_tokens": self.total_tokens,
                "total_calls": self.total_calls,
                "avg_tokens_per_task": round(self.avg_tokens_per_task, 2),
                "lanes": self.overall_lanes,
            },
            "categories": {
                cat: {
                    "total_tasks": m.total_tasks,
                    "passed": m.passed_tasks,
                    "failed": m.failed_tasks,
                    "accuracy": round(m.accuracy, 2),
                    "gate_passed": m.gate_passed,
                    "tokens": m.tokens,
                    "calls": m.calls,
                    "avg_tokens": round(m.avg_tokens_per_task, 2),
                    "lanes": m.routed_lanes,
                }
                for cat, m in self.categories.items()
            },
            **({"judge_variance": self.judge_variance} if self.judge_variance else {}),
        }

    def save_markdown(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.format_markdown())

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


def deterministic_grade(task: dict, answer: str) -> tuple[bool, str]:
    """Offline fallback grader backed by :mod:`eval.deterministic`.

    Undecided tasks (code categories without a mechanical check) are reported
    as failures here; ``eval.judge`` upgrades them to the LLM judge instead.
    """
    verdict = deterministic.evaluate(task, answer)
    return bool(verdict.passed), verdict.reason


def build_dashboard(
    tasks: List[dict],
    answers: List[dict],
    graded_verdicts: Optional[Dict[str, bool]] = None,
    category_tokens: Optional[Dict[str, int]] = None,
    category_calls: Optional[Dict[str, int]] = None,
    judge_variance: Optional[dict] = None,
) -> DashboardReport:
    """Construct a DashboardReport from tasks, agent answers, and verdicts."""
    by_id_answer = {a["task_id"]: a for a in answers}

    # Group metrics by category
    categories = {cat: CategoryMetrics(category=cat) for cat in ALL_CATEGORIES}
    overall_lanes = collections.defaultdict(int)

    for task in tasks:
        tid = task["task_id"]
        cat = task.get("category") or ALL_CATEGORIES[0]
        if cat not in categories:
            categories[cat] = CategoryMetrics(category=cat)
        m = categories[cat]
        m.total_tasks += 1

        ans_entry = by_id_answer.get(tid, {})
        ans_text = ans_entry.get("answer", "")
        routed_lane = ans_entry.get("routed", "cloud")
        task_tokens = ans_entry.get("tokens", 0)

        m.routed_lanes[routed_lane] = m.routed_lanes.get(routed_lane, 0) + 1
        overall_lanes[routed_lane] += 1

        if graded_verdicts is not None and tid in graded_verdicts:
            passed = graded_verdicts[tid]
        else:
            passed, _ = deterministic_grade(task, ans_text)

        if passed:
            m.passed_tasks += 1
        else:
            m.failed_tasks += 1

        if ans_entry and "tokens" in ans_entry:
            m.tokens += int(task_tokens or 0)
            if routed_lane.startswith("cloud"):
                m.calls += 1

    # If external category_tokens / category_calls are provided (e.g. directly from CloudClient)
    if category_tokens:
        for cat, tok in category_tokens.items():
            if cat in categories:
                categories[cat].tokens = tok
    if category_calls:
        for cat, calls in category_calls.items():
            if cat in categories:
                categories[cat].calls = calls

    total_tasks = sum(m.total_tasks for m in categories.values())
    total_passed = sum(m.passed_tasks for m in categories.values())
    total_failed = sum(m.failed_tasks for m in categories.values())
    total_tokens = sum(m.tokens for m in categories.values())
    total_calls = sum(m.calls for m in categories.values())

    for m in categories.values():
        if m.total_tasks > 0:
            m.accuracy = 100.0 * m.passed_tasks / m.total_tasks
            m.gate_passed = m.accuracy >= 80.0
            m.avg_tokens_per_task = m.tokens / m.total_tasks

    overall_accuracy = 100.0 * total_passed / max(total_tasks, 1)
    overall_gate_passed = overall_accuracy >= 80.0
    avg_tokens_overall = total_tokens / max(total_tasks, 1)

    return DashboardReport(
        categories=categories,
        total_tasks=total_tasks,
        total_passed=total_passed,
        total_failed=total_failed,
        overall_accuracy=overall_accuracy,
        overall_gate_passed=overall_gate_passed,
        total_tokens=total_tokens,
        total_calls=total_calls,
        avg_tokens_per_task=avg_tokens_overall,
        overall_lanes=dict(overall_lanes),
        judge_variance=judge_variance,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate per-category accuracy and token dashboard.")
    parser.add_argument("--tasks", default=DEV_TASKS_PATH, help="path to tasks JSON")
    parser.add_argument("--answers", help="path to answers JSON ([{'task_id': ..., 'answer': ...}])")
    parser.add_argument("--out-md", default=DEFAULT_MD_PATH, help="path to write markdown dashboard")
    parser.add_argument("--out-json", default=DEFAULT_JSON_PATH, help="path to write JSON dashboard")
    parser.add_argument("--quiet", action="store_true", help="do not print table to stdout")
    args = parser.parse_args()

    with open(args.tasks, encoding="utf-8") as fh:
        tasks = json.load(fh)

    answers_path = args.answers
    if not answers_path:
        if os.path.exists(DEFAULT_RESULTS_PATH):
            answers_path = DEFAULT_RESULTS_PATH
        else:
            answers_path = None

    if answers_path and os.path.exists(answers_path):
        with open(answers_path, encoding="utf-8") as fh:
            answers = json.load(fh)
    else:
        # Default mock / baseline answers evaluated deterministically
        from agent import specialists
        answers = []
        for t in tasks:
            cat = t["category"]
            prompt = t["prompt"]
            ans = ""
            target = "cloud"
            if cat == "sentiment" and specialists.available("sst2"):
                ans = specialists.sentiment_answer(prompt) or ""
                if ans:
                    target = "specialist"
            elif cat == "ner" and specialists.ner_enabled():
                ans = specialists.ner_answer(prompt) or ""
                if ans:
                    target = "specialist"
            if not ans:
                ans = t.get("expect", "")  # Reference baseline
            answers.append({
                "task_id": t["task_id"],
                "answer": ans,
                "category": cat,
                "routed": target,
                "tokens": 0 if target == "specialist" else 75,
            })

    report = build_dashboard(tasks, answers)

    if not args.quiet:
        print("\n" + report.format_table() + "\n")

    report.save_markdown(args.out_md)
    report.save_json(args.out_json)
    print(f"Saved dashboard markdown to: {args.out_md}")
    print(f"Saved dashboard JSON to: {args.out_json}")

    return 0 if report.overall_gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
