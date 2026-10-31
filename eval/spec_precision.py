"""Offline specialist precision harness. Zero API calls.

Replays the validated synthetic datasets (ground-truth labels/entities)
through the ONNX specialists and reports, per specialist:
- coverage: fraction of tasks it answers (rest escalate to cloud)
- precision: fraction of answered tasks whose answer matches ground truth

The gate-tuning goal is precision >= ~0.97 on the answered subset; coverage
is worth tokens but never at the price of gate misses.
"""

import json
import os
import re
import sys

from agent import specialists

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "finetune", "data")


def _load_rows(name: str):
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sentiment_report():
    rows = _load_rows("sentiment")
    answered = correct = 0
    wrong = []
    for row in rows:
        prompt = row["messages"][1]["content"]
        truth = row["messages"][2]["content"].split(",")[0].strip().lower()
        answer = specialists.sentiment_answer(prompt)
        if not answer:
            continue
        answered += 1
        predicted = answer.split(",")[0].strip().lower()
        if predicted == truth:
            correct += 1
        elif len(wrong) < 12:
            wrong.append((truth, predicted, prompt[:90]))
    total = len(rows)
    print(f"sentiment: {total} tasks, answered {answered} "
          f"({100*answered/max(total,1):.0f}% coverage), "
          f"precision {100*correct/max(answered,1):.1f}%")
    for truth, predicted, snippet in wrong:
        print(f"  WRONG truth={truth} pred={predicted}: {snippet}")


_LINE_RE = re.compile(r"^(.+?):\s*(\w+)\s*$")


def _parse_entities(text: str):
    pairs = set()
    for line in text.splitlines():
        match = _LINE_RE.match(line.strip())
        if match:
            pairs.add((match.group(1).strip().lower(), match.group(2).strip().lower()))
    return pairs


def ner_report():
    rows = _load_rows("ner")
    answered = exact = 0
    wrong = []
    for row in rows:
        prompt = row["messages"][1]["content"]
        truth = _parse_entities(row["messages"][2]["content"])
        answer = specialists.ner_answer(prompt)
        if not answer:
            continue
        answered += 1
        predicted = _parse_entities(answer)
        # Score on entity text match (types judged separately: judges care
        # most about missing/spurious entities, then type errors).
        truth_texts = {t for t, _ in truth}
        predicted_texts = {t for t, _ in predicted}
        if truth_texts == predicted_texts and truth == predicted:
            exact += 1
        elif len(wrong) < 12:
            missing = truth - predicted
            spurious = predicted - truth
            wrong.append((sorted(missing)[:3], sorted(spurious)[:3], prompt[:70]))
    total = len(rows)
    print(f"ner: {total} tasks, answered {answered} "
          f"({100*answered/max(total,1):.0f}% coverage), "
          f"exact-match precision {100*exact/max(answered,1):.1f}%")
    for missing, spurious, snippet in wrong:
        print(f"  WRONG missing={missing} spurious={spurious}: {snippet}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "sentiment"):
        sentiment_report()
    if which in ("both", "ner"):
        ner_report()
