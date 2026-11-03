"""Row-level validation for QueryLane's SFT datasets (run before training).

Every row in finetune/data/*.jsonl must satisfy:
- schema: {"category": <one of five>, "messages": [system, user, assistant]}
- the system prompt must byte-match the one agent.local uses at inference,
  otherwise the fine-tune trains on a different contract than the server runs
- category-specific content constraints (mirroring finetune.gen_data's
  acceptance rules, re-derived from the stored prompt/answer text):
  * sentiment: "Classify the sentiment..." user prefix, the fixed answer
    template for the stated label, and quoted praise/complaint spans that
    appear verbatim in the review
  * ner: one 'Entity: Type' per line, allowed types, every entity verbatim
    in the sentence
  * summarization: a known constraint prefix; the summary satisfies the
    constraint (sentence count / word budget); numbers cited in the summary
    must exist in the passage (catches hallucinated facts)
  * factual: non-empty question, answer within the generator's length cap
  * math: a single 'EXPR: <expr> | UNIT: <unit>' line whose expression
    evaluates in the same sandbox (agent.pal) the runtime uses, to a finite,
    non-negative number
Cross-row checks: duplicate (near-identical) prompts.

Usage:
    python -m finetune.validate_data                     # all of finetune/data/*.jsonl
    python -m finetune.validate_data finetune/data/math.jsonl --max-errors 20
Exit code is 1 when any error is found (with --strict, warnings too).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, namedtuple
from typing import Optional

from agent import pal
from agent.local import _LOCAL_SYSTEM

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CATEGORIES = ("sentiment", "ner", "summarization", "factual", "math")
SYSTEM_PROMPTS = _LOCAL_SYSTEM
NER_TYPES = ("Person", "Organization", "Location", "Date", "Event", "Product")
SENTIMENT_LABELS = ("positive", "negative", "mixed", "neutral")

_SENTIMENT_USER_PREFIX = "Classify the sentiment of this review: "
_NER_USER_PREFIX = "Extract all named entities and their types from: "

# Byte-exact answer templates the generator emits per sentiment label.
_POSITIVE_RE = re.compile(
    r'^Positive, the review expresses satisfaction \("(.*?)"\) with no complaints\.$'
)
_NEGATIVE_RE = re.compile(
    r'^Negative, the review reports problems \("(.*?)"\) with no praise\.$'
)
_MIXED_RE = re.compile(
    r'^Mixed, the review contains both praise \("(.*?)"\) and criticism \("(.*?)"\)\.$'
)
_NEUTRAL_TEXT = (
    "Neutral, the review is a plain factual statement with neither praise nor complaint."
)

SUMMARY_CONSTRAINTS = (
    "Summarize the following in exactly one sentence:",
    "Summarize the following in exactly two sentences:",
    "Summarize this in no more than 10 words:",
    "TL;DR in one line:",
    "Condense the following paragraph into two sentences:",
    "Summarize this in no more than 15 words:",
)

_MATH_RE = re.compile(r"^EXPR: (.+?) \| UNIT: (.+)$", re.DOTALL)
_MAX_MATH_EXPR_CHARS = 200
_MAX_FACTUAL_ANSWER_CHARS = 600

# Tokenizes numbers while dropping Q1-Q4 quarter tokens ("Q4" vs "fourth
# quarter" is phrasing, not a hallucinated statistic).
_QUARTER_TOKEN_RE = re.compile(r"\bQ([1-4])\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

Issue = namedtuple("Issue", ("line", "level", "code", "detail"))

_Roles = ("system", "user", "assistant")


# --------------------------------------------------------------------------- #
# Shared parsers — also used by finetune.dedupe_data and finetune.eval_split. #
# --------------------------------------------------------------------------- #
def normalize_prompt(text: str) -> str:
    """Canonical prompt key: casefold, collapse every non-word run to one space.

    This is the dedupe key everywhere (generator, validator, dedupe tool,
    split grouping). It deliberately ignores punctuation/case differences, so
    'Moby-Dick' and "Moby-Dick" (straight vs curly apostrophe) collide, which
    the generator's old raw 120-char prefix key missed.
    """
    return re.sub(r"\W+", " ", text.casefold()).strip()


def strip_category_prefix(category: str, user: str) -> str:
    """Remove the fixed per-category instruction wrapper, keeping only content."""
    if category == "sentiment" and user.startswith(_SENTIMENT_USER_PREFIX):
        return user[len(_SENTIMENT_USER_PREFIX):]
    if category == "ner" and user.startswith(_NER_USER_PREFIX):
        return user[len(_NER_USER_PREFIX):]
    if category == "summarization":
        _constraint, passage = split_summary_user(user)
        return passage if passage is not None else user
    return user


def split_summary_user(user: str) -> tuple:
    """Split '{constraint} {passage}' prompts into (constraint, passage).

    One of the known constraints wins when its prefix matches literally;
    otherwise the text before the first ': ' is treated as the constraint so
    typo'd constraints (e.g. 'TL;DR in one one line:') surface as failures.
    """
    for constraint in SUMMARY_CONSTRAINTS:
        if user.startswith(constraint + " "):
            return constraint, user[len(constraint) + 1:]
    match = re.match(r"^(.{1,200}?):\s(.*)$", user, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None


def summary_satisfies_constraint(constraint: str, summary: str) -> bool:
    """Port of gen_data._check_summary — same sentence/word rules as generation."""
    words = len(summary.split())
    sentences = len([s for s in re.split(r"[.!?]+", summary.strip()) if s.strip()])
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


def sentiment_answer_label(text: str):
    """First-word label of a sentiment answer: 'Positive, ...' -> 'positive'."""
    match = re.match(r"\s*(positive|negative|mixed|neutral)\b", text.casefold())
    return match.group(1) if match else None


def parse_ner_answer(text: str) -> list:
    """Parse 'Entity: Type' lines into (entity, type) pairs."""
    pairs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            pairs.append(("", ""))
            continue
        entity, _, ent_type = line.rpartition(":")
        pairs.append((entity.strip(), ent_type.strip()))
    return pairs


def parse_math_answer(text: str) -> tuple:
    """Parse 'EXPR: <expr> | UNIT: <unit>' into (expr, unit); (None, None) if absent."""
    match = _MATH_RE.match(text.strip())
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def extract_numbers(text: str) -> list:
    """Number tokens in text, ignoring Q1-Q4 quarter tokens."""
    return _NUMBER_RE.findall(_QUARTER_TOKEN_RE.sub(" ", text))


# --------------------------------------------------------------------------- #
# Per-category checks                                                         #
# --------------------------------------------------------------------------- #
def _issue(level: str, code: str, detail: str) -> tuple:
    """Convenience triple used by the per-category checkers."""
    return (level, code, detail)


def _check_sentiment(user: str, answer: str) -> list:
    issues = []
    if not user.startswith(_SENTIMENT_USER_PREFIX):
        return [_issue("error", "sentiment/user-prefix", "user does not start with the review prefix")]
    review = user[len(_SENTIMENT_USER_PREFIX):].strip()
    if not review:
        issues.append(_issue("error", "sentiment/empty-review", "review is empty"))
    label = sentiment_answer_label(answer)
    if label is None:
        issues.append(
            _issue("error", "sentiment/label", f"answer does not start with a known label: {answer[:60]!r}")
        )
        return issues
    if label == "positive":
        match = _POSITIVE_RE.match(answer)
        if not match:
            issues.append(_issue("error", "sentiment/template", "positive answer does not match the template"))
        elif not match.group(1):
            issues.append(_issue("error", "sentiment/empty-quote", "positive answer quotes an empty phrase"))
        elif match.group(1) not in review:
            issues.append(
                _issue("error", "sentiment/verbatim-quote", f"praise quote {match.group(1)!r} not verbatim in review")
            )
    elif label == "negative":
        match = _NEGATIVE_RE.match(answer)
        if not match:
            issues.append(_issue("error", "sentiment/template", "negative answer does not match the template"))
        elif not match.group(1):
            issues.append(_issue("error", "sentiment/empty-quote", "negative answer quotes an empty phrase"))
        elif match.group(1) not in review:
            issues.append(
                _issue("error", "sentiment/verbatim-quote", f"complaint quote {match.group(1)!r} not verbatim in review")
            )
    elif label == "mixed":
        match = _MIXED_RE.match(answer)
        if not match:
            issues.append(_issue("error", "sentiment/template", "mixed answer does not match the template"))
            return issues
        praise, complaint = match.group(1), match.group(2)
        if not praise or not complaint:
            issues.append(_issue("error", "sentiment/empty-quote", "mixed answer must quote praise and complaint"))
            return issues
        if praise not in review:
            issues.append(
                _issue("error", "sentiment/verbatim-quote", f"praise quote {praise!r} not verbatim in review")
            )
        if complaint not in review:
            issues.append(
                _issue("error", "sentiment/verbatim-quote", f"complaint quote {complaint!r} not verbatim in review")
            )
    else:  # neutral
        if answer.strip() != _NEUTRAL_TEXT:
            issues.append(_issue("error", "sentiment/template", "neutral answer does not match the template"))
    return issues


def _check_ner(user: str, answer: str) -> list:
    issues = []
    if not user.startswith(_NER_USER_PREFIX):
        return [_issue("error", "ner/user-prefix", "user does not start with the NER prefix")]
    sentence = user[len(_NER_USER_PREFIX):].strip()
    if not sentence:
        issues.append(_issue("error", "ner/empty-sentence", "sentence is empty"))
    pairs = parse_ner_answer(answer)
    if not pairs:
        issues.append(_issue("error", "ner/empty", "answer has no 'Entity: Type' lines"))
        return issues
    seen = Counter()
    for entity, ent_type in pairs:
        if ent_type not in NER_TYPES:
            issues.append(_issue("error", "ner/type", f"unknown type {ent_type!r} (line {entity!r})"))
            continue
        if not entity:
            issues.append(_issue("error", "ner/empty-text", "entity text is empty"))
            continue
        if entity not in sentence:
            issues.append(_issue("error", "ner/verbatim", f"entity {entity!r} not verbatim in sentence"))
        seen[(entity, ent_type)] += 1
        # agent.local.normalize() force-rewrites document nouns to Event at
        # inference; training targets saying Organization fight that rule.
        if ent_type == "Organization" and re.search(
            r"\b(treaty|agreement|accord|pact|protocol|declaration|charter|convention)\b",
            entity,
            re.IGNORECASE,
        ):
            issues.append(
                _issue(
                    "warning",
                    "ner/document-as-organization",
                    f"{entity!r} would be rewritten to Event at inference",
                )
            )
    for (entity, ent_type), count in seen.items():
        if count > 1:
            issues.append(
                _issue("warning", "ner/duplicate-line", f"{entity!r}: {ent_type} appears {count} times")
            )
    return issues


def _check_summarization(user: str, answer: str) -> list:
    issues = []
    constraint, passage = split_summary_user(user)
    if constraint is None:
        issues.append(_issue("error", "summarization/no-constraint", "user has no '<constraint>: <passage>' shape"))
    elif constraint not in SUMMARY_CONSTRAINTS:
        issues.append(_issue("error", "summarization/unknown-constraint", f"unknown constraint: {constraint!r}"))
    if not answer.strip():
        issues.append(_issue("error", "summarization/summary", "summary is empty"))
        return issues
    if constraint is not None and not summary_satisfies_constraint(constraint, answer.strip()):
        issues.append(
            _issue(
                "error",
                "summarization/length",
                f"summary violates {constraint!r} ({len(answer.split())} words)",
            )
        )
    if passage is not None:
        passage_numbers = set(extract_numbers(passage))
        missing = [n for n in extract_numbers(answer) if n not in passage_numbers]
        if missing:
            issues.append(
                _issue(
                    "error",
                    "summarization/numbers-not-in-passage",
                    f"summary cites numbers absent from the passage: {missing}",
                )
            )
    return issues


def _check_factual(user: str, answer: str) -> list:
    issues = []
    if not user.strip():
        issues.append(_issue("error", "factual/question", "question is empty"))
    if len(answer) > _MAX_FACTUAL_ANSWER_CHARS:
        issues.append(
            _issue(
                "error",
                "factual/too-long",
                f"answer is {len(answer)} chars (generator cap {_MAX_FACTUAL_ANSWER_CHARS})",
            )
        )
    return issues


def _check_math(user: str, answer: str) -> list:
    issues = []
    expr, unit = parse_math_answer(answer)
    if expr is None or unit is None:
        issues.append(_issue("error", "math/format", f"answer is not 'EXPR: ... | UNIT: ...': {answer[:60]!r}"))
        return issues
    if not expr:
        issues.append(_issue("error", "math/empty", "expression is empty"))
        return issues
    if len(expr) > _MAX_MATH_EXPR_CHARS:
        issues.append(_issue("error", "math/too-long", f"expression is {len(expr)} chars"))
    if not unit:
        issues.append(_issue("error", "math/unit", "unit is empty (use 'none')"))
    elif len(unit) > 24:
        issues.append(_issue("error", "math/unit", f"unit {unit!r} is too long to be a unit"))
    try:
        value = pal.safe_eval(expr)
    except Exception as exc:  # pal raises ValueError on anything unsafe
        issues.append(_issue("error", "math/eval", f"expression {expr!r} rejected by the PAL sandbox: {exc}"))
        return issues
    result = float(value)
    if isinstance(value, bool):
        issues.append(_issue("error", "math/eval", f"expression {expr!r} evaluates to a boolean"))
    elif result != result or result in (float("inf"), float("-inf")):
        issues.append(_issue("error", "math/eval", f"expression {expr!r} is not finite"))
    elif result < 0:
        issues.append(_issue("error", "math/negative", f"expression {expr!r} evaluates to {result} (sign error?)"))
    return issues


_CHECKERS = {
    "sentiment": _check_sentiment,
    "ner": _check_ner,
    "summarization": _check_summarization,
    "factual": _check_factual,
    "math": _check_math,
}



# --------------------------------------------------------------------------- #
# Row-level and file-level entry points                                       #
# --------------------------------------------------------------------------- #
def check_row(row, line_no: int = 0, seen_prompts: Optional[dict] = None) -> list:
    """Validate one parsed row; returns a list of Issue tuples."""
    if not isinstance(row, dict):
        return [Issue(line_no, "error", "schema/row", "row is not a JSON object")]
    issues = []
    category = row.get("category")
    if category not in CATEGORIES:
        issues.append(Issue(line_no, "error", "schema/category", f"unknown category {category!r}"))
        return issues
    for key in row:
        if key not in ("category", "messages"):
            issues.append(Issue(line_no, "warning", "schema/extra-key", f"unexpected key {key!r}"))
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        issues.append(Issue(line_no, "error", "schema/messages", "messages must be a list of exactly 3 turns"))
        return issues
    for turn, role in zip(messages, _Roles):
        if not isinstance(turn, dict) or turn.get("role") != role:
            issues.append(Issue(line_no, "error", "schema/roles", f"turn {role!r} missing or out of order"))
            return issues
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            issues.append(Issue(line_no, "error", "schema/content", f"{role!r} content is empty"))
            return issues
    if messages[0]["content"] != SYSTEM_PROMPTS[category]:
        issues.append(
            Issue(
                line_no,
                "error",
                "schema/system-prompt",
                "system prompt differs from agent.local._LOCAL_SYSTEM (train/serve skew)",
            )
        )
    if seen_prompts is not None:
        key = normalize_prompt(messages[1]["content"])
        if key in seen_prompts:
            issues.append(
                Issue(
                    line_no,
                    "warning",
                    "duplicate-prompt",
                    f"near-identical prompt already seen on line {seen_prompts[key]}",
                )
            )
        else:
            seen_prompts[key] = line_no
    user, answer = messages[1]["content"], messages[2]["content"]
    issues.extend(
        Issue(line_no, level, code, detail) for level, code, detail in _CHECKERS[category](user, answer)
    )
    return issues


def validate_rows(rows, first_line: int = 1) -> list:
    """Validate a list of parsed rows; line numbers are 1-based JSONL positions."""
    seen = {}
    issues = []
    for offset, row in enumerate(rows):
        issues.extend(check_row(row, line_no=first_line + offset, seen_prompts=seen))
    return issues


def load_jsonl(path: str) -> tuple:
    """Read a JSONL file; returns (rows, issues) with parse errors as Issues."""
    rows, issues = [], []
    try:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    issues.append(Issue(line_no, "error", "schema/json", f"invalid JSON: {exc}"))
    except UnicodeDecodeError as exc:
        issues.append(Issue(0, "error", "schema/encoding", f"file is not valid UTF-8: {exc}"))
    return rows, issues


def validate_file(path: str) -> tuple:
    """Validate one file; returns (rows, issues, per-category row counts)."""
    rows, issues = load_jsonl(path)
    issues.extend(validate_rows(rows))
    counts = Counter(row.get("category") for row in rows if isinstance(row, dict))
    return rows, issues, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate QueryLane SFT JSONL rows before training.")
    parser.add_argument("files", nargs="*", help="JSONL files to validate (default: every data/*.jsonl)")
    parser.add_argument("--max-errors", type=int, default=15, help="detailed issues to print per file")
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings (duplicate prompts, etc.) as failures"
    )
    args = parser.parse_args(argv)

    files = args.files or sorted(
        os.path.join(DATA_DIR, name) for name in os.listdir(DATA_DIR) if name.endswith(".jsonl")
    )
    if not files:
        print("no JSONL files found to validate", file=sys.stderr)
        return 2

    failed_files = 0
    for path in files:
        rows, issues, category_counts = validate_file(path)
        errors = [issue for issue in issues if issue.level == "error"]
        warnings = [issue for issue in issues if issue.level == "warning"]
        failed = bool(errors) or (args.strict and bool(warnings))
        failed_files += 1 if failed else 0
        status = "FAIL" if failed else "ok"
        print(f"[{status}] {path}: {len(rows)} rows, {len(errors)} errors, {len(warnings)} warnings")
        if category_counts:
            breakdown = ", ".join(f"{cat}={n}" for cat, n in sorted(category_counts.items()))
            print(f"        categories: {breakdown}")
        if not issues:
            continue
        for code, count in sorted(Counter(f"{issue.level}:{issue.code}" for issue in issues).items()):
            print(f"        {count:4d} x {code}")
        for issue in issues[: args.max_errors]:
            print(f"        line {issue.line}: [{issue.level}] {issue.code}: {issue.detail[:140]}")
        if len(issues) > args.max_errors:
            print(f"        ... {len(issues) - args.max_errors} more issues suppressed")
        print()

    verdict = "FAIL" if failed_files else "PASS"
    print(f"{verdict}: {failed_files} of {len(files)} file(s) failed validation")
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())






