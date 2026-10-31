"""Synthetic SFT dataset generator for the local specialist model.

Uses the Fireworks API (kimi, thinking off, high temperature) to produce
(prompt, ideal answer) pairs per category, then validates every pair
programmatically before accepting it:
- math: the emitted expression must evaluate to the stated answer
- summarization: the stated length constraint must actually hold
- ner: every line must be 'Entity: Type' with an allowed type
- sentiment: the answer must start with an allowed label

Training pairs embed the EXACT system prompts the agent uses at inference
(imported from agent.local), so there is no train/serve prompt skew.

Usage:
    set -a; source .env; set +a
    python -m finetune.gen_data --per-category 600         # full run
    python -m finetune.gen_data --per-category 30 --pilot  # quick pilot
"""

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from agent import pal
from agent.local import _LOCAL_SYSTEM

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
ITEMS_PER_CALL = 10

_SENTIMENT_DOMAINS = [
    "wireless earbuds", "a hotel stay", "a neighborhood restaurant", "a mobile banking app",
    "a blender", "a food delivery order", "a paperback novel", "a streaming service",
    "a laptop", "an airline flight", "a rental car", "a fitness tracker",
    "a coffee shop visit", "a software subscription", "an office chair", "a board game",
    "a customer support interaction", "a smart thermostat", "a gym membership", "a phone case",
]
_NER_DOMAINS = [
    "business news", "sports results", "a science announcement", "a political summit",
    "a travel itinerary", "a product launch", "a historical treaty or accord",
    "a health organization briefing", "an entertainment awards story", "a finance headline",
    "a university research collaboration", "a court ruling", "a space mission update",
    "a museum exhibition opening", "a company acquisition",
]
_SUMM_TOPICS = [
    "a city council decision", "a research finding", "quarterly company earnings",
    "a severe weather event", "animal behavior research", "a transit project",
    "a school program", "a startup product launch", "a public health study",
    "an energy policy change", "a sports season recap", "a museum discovery",
]
_SUMM_CONSTRAINTS = [
    "Summarize the following in exactly one sentence:",
    "Summarize the following in exactly two sentences:",
    "Summarize this in no more than 10 words:",
    "TL;DR in one line:",
    "Condense the following paragraph into two sentences:",
    "Summarize this in no more than 15 words:",
]
_FACTUAL_TOPICS = [
    "world geography", "basic physics or chemistry concepts", "modern history",
    "computing acronyms and protocols", "classic literature", "units and measurements",
    "human biology", "astronomy", "famous inventions", "world languages",
    "economics terms", "music theory basics",
]
_MATH_SCENARIOS = [
    "store discounts and sales", "travel speed, distance and time", "recipe scaling",
    "weekly savings and spending", "percentage increase or decrease", "splitting a bill",
    "simple averages", "unit price comparisons", "work completed per hour",
    "capacity and batching (needing ceil)", "simple interest", "score totals across games",
]

_NER_TYPES = ("person", "organization", "location", "date", "event", "product")
_SENTIMENT_LABELS = ("positive", "negative", "mixed", "neutral")

_MATH_SYSTEM = _LOCAL_SYSTEM["math"]


def _gen_prompt(category: str, seed: str, k: int) -> str:
    if category == "sentiment":
        return f"""Create {k} short product/service reviews about {seed} for a sentiment-classification dataset.
Vary length (1-3 sentences) and difficulty. Include at least: 2 clearly positive, 2 clearly negative,
2 mixed (praise AND complaint), 1 neutral (purely informational, no praise, no complaint),
and 1 negative review phrased as a plain factual statement of a problem (late, broken, missing).
For each review also give praise_quote (a short verbatim phrase from the review expressing satisfaction,
or "" if none) and complaint_quote (a short verbatim phrase expressing a problem, or "" if none).
Quotes MUST appear verbatim inside the review.
Return STRICT JSON: an array of {k} objects {{"review": str, "label": one of positive/negative/mixed/neutral,
"praise_quote": str, "complaint_quote": str}}. No other text."""
    if category == "ner":
        return f"""Create {k} single sentences in the style of {seed}, each containing 3-6 named entities
(mix of Person, Organization, Location, Date, Event, Product; vary which appear). Use realistic but
invented names. Return STRICT JSON: array of {k} objects {{"sentence": str, "entities": [{{"text": str,
"type": one of Person/Organization/Location/Date/Event/Product}}]}}. Entities must appear verbatim in
the sentence. Treaties, agreements and summits are Event. Countries and cities are Location. No other text."""
    if category == "summarization":
        constraint = random.choice(_SUMM_CONSTRAINTS[:2] + _SUMM_CONSTRAINTS[3:5])
        return f"""Create {k} passages (60-100 words each) about {seed}, each with 2-4 concrete numeric facts.
For each passage, work in this order:
1. "key_numbers": list the 2-4 numeric strings a grader would require (write them exactly as in the passage, e.g. ["46 million", "7", "12%"]).
2. "summary": write the summary satisfying the constraint "{constraint}" and containing EVERY string
from key_numbers verbatim. Check each one before finishing.
Return STRICT JSON: array of {k} objects {{"passage": str, "constraint": "{constraint}",
"key_numbers": [str], "summary": str}}. No other text."""
    if category == "factual":
        return f"""Create {k} factual questions about {seed}, mixing single-part and multi-part questions
(e.g. "What is X and where is it Y?"). Some should state a format ("in two sentences", "name three").
Return STRICT JSON: array of {k} objects {{"question": str, "answer": str}} where answer is accurate,
concise, covers every part, and obeys any stated format. No other text."""
    if category == "math":
        return f"""Create {k} grade-school word problems about {seed} (2-4 steps: percentages, rates,
totals, remainders). Each problem must ask for exactly ONE numeric quantity. For each, give the single
Python arithmetic expression that computes the answer (allowed: numbers, + - * / // % ** parentheses,
ceil, floor, round, abs, min, max, sqrt), the numeric answer, and the unit of the answer ("$", "km/h",
"items", "grams", ... or "" when unitless).
Return STRICT JSON: array of {k} objects {{"problem": str, "expression": str, "answer": number, "unit": str}}. No other text."""
    raise ValueError(category)


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])


def _check_summary(constraint: str, summary: str) -> bool:
    words = len(summary.split())
    sentences = _count_sentences(summary)
    if "exactly one sentence" in constraint:
        return sentences == 1
    if "two sentences" in constraint:
        return sentences == 2
    if "10 words" in constraint:
        return words <= 10
    if "15 words" in constraint:
        return words <= 15
    if "one line" in constraint:
        return "\n" not in summary.strip()
    return False


def _to_rows(category: str, items: list) -> list:
    rows = []
    for it in items:
        try:
            if category == "sentiment":
                label = str(it["label"]).strip().lower()
                if label not in _SENTIMENT_LABELS:
                    continue
                review = str(it["review"])
                praise = str(it.get("praise_quote", "")).strip().strip('"')
                complaint = str(it.get("complaint_quote", "")).strip().strip('"')
                # Quotes must be verbatim, matching the judged specialist style.
                if praise and praise not in review:
                    continue
                if complaint and complaint not in review:
                    continue
                if label == "mixed":
                    if not praise or not complaint:
                        continue
                    answer = (f'Mixed, the review contains both praise ("{praise}") '
                              f'and criticism ("{complaint}").')
                elif label == "positive":
                    if not praise or complaint:
                        continue
                    answer = (f'Positive, the review expresses satisfaction '
                              f'("{praise}") with no complaints.')
                elif label == "negative":
                    if not complaint or praise:
                        continue
                    answer = (f'Negative, the review reports problems '
                              f'("{complaint}") with no praise.')
                else:
                    # A "neutral" row carrying praise or complaint teaches the
                    # model to call two-sided reviews neutral. Reject it.
                    if praise or complaint:
                        continue
                    answer = ("Neutral, the review is a plain factual statement "
                              "with neither praise nor complaint.")
                user = f"Classify the sentiment of this review: {review}"
                system = _LOCAL_SYSTEM["sentiment"]
            elif category == "ner":
                entities = it["entities"]
                if not entities:
                    continue
                if any(not str(e.get("text", "")).strip() for e in entities):
                    continue
                if any(str(e["type"]).strip().lower() not in _NER_TYPES for e in entities):
                    continue
                if any(str(e["text"]) not in it["sentence"] for e in entities):
                    continue
                answer = "\n".join(
                    f"{e['text']}: {str(e['type']).strip().capitalize()}" for e in entities
                )
                user = f"Extract all named entities and their types from: {it['sentence']}"
                system = _LOCAL_SYSTEM["ner"]
            elif category == "summarization":
                if not _check_summary(it["constraint"], it["summary"]):
                    continue
                summary = str(it["summary"]).strip()
                key_numbers = [str(n).strip() for n in it.get("key_numbers", [])]
                # Fact-coverage validation: a summary that loses a key number
                # teaches the fact-dropping behavior that failed round 1.
                if not key_numbers or any(n and n not in summary for n in key_numbers):
                    continue
                user = f"{it['constraint']} {it['passage']}"
                answer = summary
                system = _LOCAL_SYSTEM["summarization"]
            elif category == "factual":
                answer = str(it["answer"]).strip()
                if not answer or len(answer) > 600:
                    continue
                user = str(it["question"]).strip()
                system = _LOCAL_SYSTEM["factual"]
            elif category == "math":
                expr = str(it["expression"]).strip()
                value = pal.safe_eval(expr)
                stated = float(it["answer"])
                if abs(float(value) - stated) > max(1e-6, abs(stated) * 1e-6):
                    continue
                if float(value) < 0:
                    continue  # negative results are usually sign errors
                unit = str(it.get("unit", "")).strip()
                user = str(it["problem"]).strip()
                answer = f"EXPR: {expr} | UNIT: {unit if unit else 'none'}"
                system = _MATH_SYSTEM
            else:
                continue
            rows.append(
                {
                    "category": category,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
        except Exception:
            continue
    return rows


_SEEDS = {
    "sentiment": _SENTIMENT_DOMAINS,
    "ner": _NER_DOMAINS,
    "summarization": _SUMM_TOPICS,
    "factual": _FACTUAL_TOPICS,
    "math": _MATH_SCENARIOS,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--categories", default="sentiment,ner,summarization,factual,math")
    parser.add_argument("--pilot", action="store_true", help="print samples")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in _SEEDS]
    if not categories or unknown:
        parser.error(f"unknown categories: {unknown or '(none given)'}")

    client = OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=120.0,
    )
    allowed = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not allowed:
        parser.error("ALLOWED_MODELS must name at least one model")
    model = next((m for m in allowed if "kimi" in m.lower()), allowed[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    tokens_lock = threading.Lock()
    tokens = {"n": 0}

    def one_call(category: str, seed: str) -> list:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _gen_prompt(category, seed, ITEMS_PER_CALL)}],
            temperature=0.9,
            max_tokens=4000,
            extra_body={"reasoning_effort": "none"},
        )
        usage = getattr(response, "usage", None)
        if usage and usage.total_tokens:
            with tokens_lock:
                tokens["n"] += usage.total_tokens
        if not response.choices:
            return []
        text = (response.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return []
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        return _to_rows(category, items if isinstance(items, list) else [])

    for category in categories:
        seeds = _SEEDS[category]
        calls_needed = max(1, args.per_category // ITEMS_PER_CALL)
        jobs = [(category, seeds[i % len(seeds)]) for i in range(calls_needed)]
        rows, seen = [], set()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for batch in pool.map(lambda j: one_call(*j), jobs):
                for row in batch:
                    key = row["messages"][1]["content"][:120]
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
        path = os.path.join(OUT_DIR, f"{category}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept = len(rows)
        print(f"{category}: kept {kept} validated rows -> {path}", flush=True)
        if args.pilot and rows:
            sample = random.choice(rows)
            print("  sample user:", sample["messages"][1]["content"][:140])
            print("  sample answer:", sample["messages"][2]["content"][:140])

    # Rebuild the combined file from every per-category file on disk. A partial
    # run (`--categories summarization`) must never clobber train_all.jsonl
    # with only the categories it regenerated.
    all_rows = []
    for category in _SEEDS:
        path = os.path.join(OUT_DIR, f"{category}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            all_rows.extend(json.loads(line) for line in fh if line.strip())
    random.Random(42).shuffle(all_rows)
    combined = os.path.join(OUT_DIR, "train_all.jsonl")
    with open(combined, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"TOTAL {len(all_rows)} rows -> {combined}; generator tokens spent: {tokens['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
