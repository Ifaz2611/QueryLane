"""Dev-only evaluation harness. Never ships in the image.

Answers eval/tasks_dev.json through the real agent pipeline, then grades each
answer with a Fireworks LLM judge against the task's expected intent. Reports
per-category accuracy plus the agent's Fireworks token total (the leaderboard
tiebreaker). Judge tokens are tracked separately; they are dev spend, not part
of the score.

Usage:
    set -a; source .env; set +a
    python -m eval.judge                 # answer + judge everything
    python -m eval.judge --answers eval/results_dev.json   # re-judge saved answers
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
from eval.dashboard import build_dashboard

DEV_TASKS = os.path.join(os.path.dirname(__file__), "tasks_dev.json")

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
    def __init__(self):
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
        # Prefer the strongest generalist as judge.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", help="re-judge a saved answers file")
    parser.add_argument(
        "--save", default="eval/results_dev.json", help="where to save answers"
    )
    parser.add_argument("--category", help="only run one category")
    parser.add_argument(
        "--out-md", default="eval/dashboard.md", help="path to write markdown dashboard"
    )
    parser.add_argument(
        "--out-json", default="eval/dashboard.json", help="path to write JSON dashboard"
    )
    args = parser.parse_args()

    with open(DEV_TASKS, encoding="utf-8") as fh:
        tasks = json.load(fh)
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]

    if args.answers:
        with open(args.answers, encoding="utf-8") as fh:
            answers = json.load(fh)
        task_ids = {t["task_id"] for t in tasks}
        answers = [a for a in answers if a.get("task_id") in task_ids]
        agent_tokens, agent_calls = None, None
    else:
        answers, agent_tokens, agent_calls = answer_all(tasks)
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(answers, fh, ensure_ascii=False, indent=1)

    by_id = {t["task_id"]: t for t in tasks}
    judge = Judge()

    def grade(entry):
        task = by_id[entry["task_id"]]
        passed, reason = judge.grade(task["prompt"], task["expect"], entry["answer"])
        return entry, passed, reason

    with ThreadPoolExecutor(max_workers=4) as pool:
        graded = list(pool.map(grade, answers))

    per_category = {}
    for entry, passed, reason in graded:
        cat = entry.get("category", by_id[entry["task_id"]]["category"])
        per_category.setdefault(cat, []).append(passed)
        mark = "PASS" if passed else "FAIL"
        print(f"{mark} {entry['task_id']} [{cat}] {reason}")

    print()
    total_pass, total = 0, 0
    for cat in sorted(per_category):
        outcomes = per_category[cat]
        total_pass += sum(outcomes)
        total += len(outcomes)
        print(f"{cat:15s} {sum(outcomes)}/{len(outcomes)}")
    pct = 100.0 * total_pass / total if total else 0.0
    print(f"{'OVERALL':15s} {total_pass}/{total}  ({pct:.1f}%)  gate is 80%")
    if agent_tokens is not None:
        print(f"agent fireworks tokens: {agent_tokens} across {agent_calls} calls")
    print(f"judge tokens (dev spend only): {judge.tokens}")

    # Generate and display the per-category accuracy/token dashboard
    verdicts = {entry["task_id"]: passed for entry, passed, _ in graded}
    report = build_dashboard(tasks, answers, graded_verdicts=verdicts)
    print("\n" + report.format_table())
    report.save_markdown(args.out_md)
    report.save_json(args.out_json)
    print(f"\nDashboard saved to: {args.out_md} and {args.out_json}")

    return 0 if pct >= 80.0 else 1


if __name__ == "__main__":
    sys.exit(main())
