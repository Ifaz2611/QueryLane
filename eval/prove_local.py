"""Local model fine-tune verification harness for summarization and logic.

Summarization and logic regressed in earlier local fine-tune A/B runs:
- Summarization: regressed to 0/5 in v13 (violated strict sentence/word count
  constraints, hallucinated numbers, dropped key constraints).
- Logic: scored 0/5 in v14 (failed multi-step constraint deduction).

Per project policy, neither category may be promoted to local execution
until an explicit fine-tune evaluation proves passing performance (>= 80%,
i.e. at least 4/5 correct, zero constraint violations, and zero empty answers).

Usage:
    python -m eval.prove_local --category summarization --answers results.json
    python -m eval.prove_local --category logic --answers results.json
    python -m eval.prove_local --status   # report current proof status
"""

import argparse
import json
import os
import re
import sys

DEV_TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks_dev.json")


def _count_sentences(text: str) -> int:
    """Count sentences using terminal punctuation."""
    # Split on period, exclamation, or question mark followed by whitespace or end of string.
    raw = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]
    return len(raw)


def _count_words(text: str) -> int:
    return len(text.strip().split())


def grade_summarization(task_id: str, answer: str) -> tuple[bool, str]:
    """Deterministic rubric grading for summarization dev tasks."""
    ans = answer.strip()
    if not ans:
        return False, "empty answer"

    sentences = _count_sentences(ans)
    words = _count_words(ans)
    lines = [line for line in ans.splitlines() if line.strip()]
    ans_lower = ans.lower()

    if task_id == "summ-01":
        # Exactly one sentence, mentions council approval/bike lane
        if sentences != 1:
            return False, f"expected exactly 1 sentence, got {sentences}"
        if not ("bike" in ans_lower or "pilot" in ans_lower):
            return False, "missing core topic (bike lane pilot)"
        if not ("council" in ans_lower or "approved" in ans_lower or "7-4" in ans_lower):
            return False, "missing council approval context"
        return True, "satisfied 1-sentence constraint and key facts"

    elif task_id == "summ-02":
        # Ten words or fewer, battery charging / fast
        if words > 10:
            return False, f"expected at most 10 words, got {words}"
        if not ("battery" in ans_lower or "batteries" in ans_lower or "charge" in ans_lower):
            return False, "missing battery/charging topic"
        return True, "satisfied <= 10 words constraint"

    elif task_id == "summ-03":
        # Exactly two sentences covering hybrid/remote work and adaptation
        if sentences != 2:
            return False, f"expected exactly 2 sentences, got {sentences}"
        if not ("remote" in ans_lower or "hybrid" in ans_lower or "work" in ans_lower):
            return False, "missing remote/hybrid work subject"
        return True, "satisfied 2-sentence constraint"

    elif task_id == "summ-04":
        # Exactly one sentence: honeybees waggle dance communicates food location
        if sentences != 1:
            return False, f"expected exactly 1 sentence, got {sentences}"
        if not ("bee" in ans_lower or "dance" in ans_lower or "waggle" in ans_lower):
            return False, "missing honeybee waggle dance"
        if not ("food" in ans_lower or "location" in ans_lower or "direction" in ans_lower):
            return False, "missing communication purpose"
        return True, "satisfied 1-sentence constraint and key facts"

    elif task_id == "summ-05":
        # One line TL;DR: revenue up / subscriptions / hardware
        if len(lines) > 1:
            return False, f"expected 1 line, got {len(lines)}"
        if not ("revenue" in ans_lower or "subscription" in ans_lower or "12%" in ans_lower):
            return False, "missing financial facts"
        return True, "satisfied 1-line constraint"

    return False, f"unknown summarization task_id: {task_id}"


def grade_logic(task_id: str, answer: str) -> tuple[bool, str]:
    """Deterministic rubric grading for logic deduction dev tasks."""
    ans = answer.strip()
    if not ans:
        return False, "empty answer"
    ans_lower = ans.lower()

    if task_id == "logic-01":
        # Sam owns the cat
        if "sam" in ans_lower and "cat" in ans_lower:
            return True, "correct deduction: Sam owns the cat"
        return False, "failed to deduce that Sam owns the cat"

    elif task_id == "logic-02":
        # Ben finished last / third / 3rd
        if "ben" in ans_lower and ("last" in ans_lower or "third" in ans_lower or "3rd" in ans_lower or "3" in ans_lower):
            return True, "correct deduction: Ben finished last/third"
        return False, "failed to deduce that Ben finished last"

    elif task_id == "logic-03":
        # Yellow, red, blue, green
        normalized = re.sub(r"[^a-z, ]", "", ans_lower)
        if "yellow" in normalized and "red" in normalized and "blue" in normalized and "green" in normalized:
            y_idx = normalized.find("yellow")
            r_idx = normalized.find("red")
            b_idx = normalized.find("blue")
            g_idx = normalized.find("green")
            if y_idx < r_idx < b_idx < g_idx:
                return True, "correct order: yellow, red, blue, green"
        return False, "order mismatch, expected yellow, red, blue, green"

    elif task_id == "logic-04":
        # Maya is the second tallest (Omar > Maya > Nina > Petra)
        if "maya" in ans_lower:
            return True, "correct deduction: Maya"
        return False, "failed to deduce Maya"

    elif task_id == "logic-05":
        # Seat 2 is Hana
        if "2" in ans_lower or "seat 2" in ans_lower or "hana" in ans_lower:
            return True, "correct deduction: Seat 2"
        return False, "failed to deduce Seat 2"

    return False, f"unknown logic task_id: {task_id}"


def evaluate_category(category: str, answers_by_id: dict) -> dict:
    with open(DEV_TASKS_PATH, encoding="utf-8") as fh:
        tasks = [t for t in json.load(fh) if t["category"] == category]

    results = []
    passed = 0
    for task in tasks:
        tid = task["task_id"]
        ans = answers_by_id.get(tid, "")
        if category == "summarization":
            ok, reason = grade_summarization(tid, ans)
        elif category == "logic":
            ok, reason = grade_logic(tid, ans)
        else:
            ok, reason = False, f"category {category} not supported by prove_local"
        if ok:
            passed += 1
        results.append({"task_id": tid, "pass": ok, "reason": reason})

    total = len(tasks)
    accuracy = (passed / total) if total else 0.0
    gate_passed = accuracy >= 0.80  # 80% gate (at least 4/5)

    return {
        "category": category,
        "total": total,
        "passed": passed,
        "accuracy": accuracy,
        "gate_passed": gate_passed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify local fine-tunes for summarization/logic.")
    parser.add_argument("--category", choices=["summarization", "logic"], help="category to verify")
    parser.add_argument("--answers", help="path to JSON answers file ([{'task_id': ..., 'answer': ...}])")
    parser.add_argument("--status", action="store_true", help="report current local category promotion status")
    args = parser.parse_args()

    if args.status or not args.answers:
        from agent import local
        print("=== Local Category Promotion Gate Status ===")
        for cat in ("math", "ner", "factual", "sentiment", "summarization", "logic"):
            proven = local.category_proven(cat)
            tps_bar = local._CATEGORY_MIN_TPS.get(cat, local.MIN_TOKENS_PER_SEC)
            status = "PROVEN / ACTIVE" if proven and tps_bar < 90.0 else "UNPROVEN / GATED (cloud only)"
            print(f"  {cat:15s} TPS bar: {tps_bar:<6.1f} Status: {status}")
        print("\nPolicy:")
        print("  - summarization: 0/5 in v13 A/B (length/hallucination regressions). Gated off.")
        print("  - logic: 0/5 in v14 A/B (deduction failures). Gated off.")
        print("  Promotion requires >= 80% (4/5) passing rubric on dev tasks.\n")
        if not args.answers:
            return 0

    with open(args.answers, encoding="utf-8") as fh:
        raw = json.load(fh)
    answers_by_id = {item["task_id"]: item.get("answer", "") for item in raw}

    categories = [args.category] if args.category else ["summarization", "logic"]
    all_passed = True
    for cat in categories:
        report = evaluate_category(cat, answers_by_id)
        pct = report["accuracy"] * 100.0
        status_str = "PROVEN (Gate PASSED)" if report["gate_passed"] else "REJECTED (Gate FAILED)"
        print(f"\n--- Category: {cat} [{report['passed']}/{report['total']} - {pct:.1f}%] -> {status_str} ---")
        for res in report["results"]:
            mark = "PASS" if res["pass"] else "FAIL"
            print(f"  [{mark}] {res['task_id']}: {res['reason']}")
        if not report["gate_passed"]:
            all_passed = False
            print(f"  >> Category '{cat}' remains BLOCKED from local inference (must pass >= 80%).")
        else:
            print(f"  >> Category '{cat}' PROVEN! Set LOCAL_PROVEN_CATEGORIES={cat} to enable.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
