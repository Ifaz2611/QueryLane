"""Deterministic checks for the dev evaluation set.

The LLM judge is useful for open-ended answers, but it is noisy on the parts of
the rubric that can be checked mechanically: sentiment labels, summarization
length constraints, exact math results, factual key points, logic answers, and
NER entity/type sets. This module runs those checks first so a re-judge cannot
flip a verdict that the task itself already decided.

Every task may carry a ``check`` object in ``eval/tasks_dev.json``. Tasks that
have no check (currently the code categories) return ``UNDECIDED`` and fall
through to the LLM judge. Known task ids also get their check loaded from
``tasks_dev.json`` when a caller passes only a slim task dict.

A verdict is tri-state:

- ``passed is True`` / ``False`` with ``authoritative=True``: the answer can be
  graded without an LLM.
- ``passed is None``: deterministic checks cannot decide; the caller should use
  the judge. ``authoritative`` is always false in this case.

Usage:
    from eval.deterministic import evaluate
    verdict = evaluate(task, answer)
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

DEV_TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks_dev.json")
CHECKS_PATH = os.path.join(os.path.dirname(__file__), "checks_dev.json")


@dataclass(frozen=True)
class Verdict:
    passed: Optional[bool]
    reason: str
    authoritative: bool
    check_type: str = ""


def _pass(reason: str, check_type: str = "") -> Verdict:
    return Verdict(True, reason, True, check_type)


def _fail(reason: str, check_type: str = "") -> Verdict:
    return Verdict(False, reason, True, check_type)


def _undecided(reason: str, check_type: str = "") -> Verdict:
    return Verdict(None, reason, False, check_type)


# ------------------------------------------------------------------ text utils

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Lowercase and collapse punctuation so phrase checks are formatting-free."""
    return _NORMALIZE_RE.sub(" ", str(text).lower()).strip()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(str(text)))


def count_sentences(text: str) -> int:
    """Count sentences by terminal punctuation, ignoring decimals/abbreviations."""
    cleaned = _ABBREV_RE.sub(lambda match: match.group(0).replace(".", ""), str(text))
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", "", cleaned)
    parts = [part for part in re.split(r"[.!?]+(?=\s|$)", cleaned) if part.strip()]
    return len(parts)


def count_lines(text: str) -> int:
    return sum(1 for line in str(text).splitlines() if line.strip())


def numbers_in(text: str) -> list[float]:
    values = []
    for raw in _NUMBER_RE.findall(str(text)):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _number_close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def _term_regex(term: str):
    normalized = normalize_text(term)
    if not normalized:
        return None
    pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])")


def _has_term(term: str, normalized_text: str) -> bool:
    regex = _term_regex(term)
    return bool(regex and regex.search(normalized_text))


def _unit_present(unit: str, answer_lower: str) -> bool:
    unit = str(unit).strip().lower()
    if not unit:
        return False
    if len(unit) == 1 and unit.isalnum():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(unit)}(?![a-z0-9])", answer_lower))
    return unit in answer_lower


def _scope(answer: str, final_only: bool) -> str:
    """Return the part of the answer that states the conclusion.

    Logic answers often explain the deduction first and put the result on the
    last line (or after ``Answer:``). Checking only that span prevents a
    reasoning paragraph that mentions every name from looking like a correct
    final answer.
    """
    if not final_only:
        return answer
    labels = list(re.finditer(r"\banswer\s*:\s*", answer, re.IGNORECASE))
    if labels:
        return answer[labels[-1].end():]
    lines = [line for line in answer.splitlines() if line.strip()]
    return lines[-1] if len(lines) > 1 else answer


# ------------------------------------------------------------------ constraints

def _check_constraints(check: dict, answer: str) -> Verdict:
    text = _scope(answer, bool(check.get("final_only")))
    normalized = normalize_text(text)

    problems = []

    sentence_count = check.get("sentence_count")
    if sentence_count is not None:
        actual = count_sentences(text)
        if actual != int(sentence_count):
            problems.append(f"expected {sentence_count} sentence(s), got {actual}")

    min_sentences = check.get("min_sentences")
    if min_sentences is not None and count_sentences(text) < int(min_sentences):
        problems.append(f"expected at least {min_sentences} sentence(s)")

    max_sentences = check.get("max_sentences")
    if max_sentences is not None and count_sentences(text) > int(max_sentences):
        problems.append(f"expected at most {max_sentences} sentence(s)")

    min_words = check.get("min_words")
    if min_words is not None and count_words(text) < int(min_words):
        problems.append(f"expected at least {min_words} words")

    max_words = check.get("max_words")
    if max_words is not None and count_words(text) > int(max_words):
        problems.append(f"expected at most {max_words} words, got {count_words(text)}")

    min_lines = check.get("min_lines")
    if min_lines is not None and count_lines(text) < int(min_lines):
        problems.append(f"expected at least {min_lines} line(s)")

    max_lines = check.get("max_lines")
    if max_lines is not None and count_lines(text) > int(max_lines):
        problems.append(f"expected at most {max_lines} line(s), got {count_lines(text)}")

    for term in check.get("required_terms_all", []):
        if not _has_term(term, normalized):
            problems.append(f"missing required term {term!r}")

    for group in check.get("all_groups", []):
        if not any(_has_term(term, normalized) for term in group):
            problems.append(f"missing any of {group!r}")

    for group in check.get("forbidden_groups", []):
        if any(_has_term(term, normalized) for term in group):
            problems.append(f"contains forbidden term from {group!r}")

    tolerance = float(check.get("number_tolerance", 1e-6))
    actual_numbers = numbers_in(text)
    for expected in check.get("required_numbers", []):
        expected = float(expected)
        if not any(_number_close(value, expected, tolerance) for value in actual_numbers):
            problems.append(f"missing required number {expected:g}")

    ordered = check.get("ordered_terms", [])
    if ordered:
        positions = []
        for term in ordered:
            regex = _term_regex(term)
            match = regex.search(normalized) if regex else None
            if not match:
                problems.append(f"missing ordered term {term!r}")
                break
            positions.append(match.start())
        else:
            if positions != sorted(positions):
                problems.append(f"expected order {ordered!r}")

    for pattern in check.get("forbidden_patterns", []):
        if re.search(pattern, text, re.IGNORECASE):
            problems.append(f"matched forbidden pattern {pattern!r}")

    if problems:
        return _fail("; ".join(problems), "constraints")
    return _pass("all deterministic constraints satisfied", "constraints")


# ------------------------------------------------------------------- sentiment

_LABEL_ORDER = ("mixed", "positive", "negative", "neutral")


def _label_positions(answer: str) -> dict:
    lower = answer.lower()
    positions = {}
    for label in _LABEL_ORDER:
        match = re.search(rf"\b{label}\b", lower)
        if match:
            positions[label] = match.start()
    return positions


def _check_sentiment(check: dict, answer: str) -> Verdict:
    allowed = [str(label).lower() for label in check.get("labels", [])]
    if not allowed:
        return _undecided("sentiment check has no labels", "sentiment")

    positions = _label_positions(answer)
    if not positions:
        return _fail(f"no sentiment label found; expected one of {allowed}", "sentiment")

    positive = "positive" in positions
    negative = "negative" in positions

    primary = min(positions, key=positions.get)
    # "Mixed" answers usually mention both polarities too. Prefer an explicit
    # mixed label when the task accepts it, even if "positive" appears first.
    if "mixed" in allowed and "mixed" in positions and (
        positions["mixed"] <= 80 or (positive and negative)
    ):
        primary = "mixed"

    if primary not in allowed:
        if "mixed" in allowed and positive and negative:
            primary = "mixed"
        else:
            return _fail(
                f"label {primary!r} not accepted; expected one of {allowed}",
                "sentiment",
            )

    min_after = int(check.get("min_words_after_label", 0))
    if min_after:
        explicit = re.search(rf"\b{re.escape(primary)}\b", answer, re.IGNORECASE)
        if explicit:
            words_after = count_words(answer[explicit.end():])
        else:
            # Mixed inferred from both polarities: justify from the first
            # polarity label instead of searching for a word that is absent.
            first_position = min(positions.values())
            words_after = count_words(answer[first_position:])
        if words_after < min_after:
            return _fail(
                f"label {primary!r} needs a justification "
                f"({words_after} word(s) after label, need {min_after})",
                "sentiment",
            )

    return _pass(f"sentiment label {primary!r} accepted", "sentiment")


# ------------------------------------------------------------------------ NER

_TYPE_ALIASES = {
    "person": {"person", "per", "people", "individual"},
    "organization": {
        "organization",
        "organisation",
        "org",
        "company",
        "institution",
        "group",
        "agency",
        "business",
    },
    "location": {
        "location",
        "loc",
        "gpe",
        "place",
        "country",
        "city",
        "region",
        "state",
    },
    "date": {"date", "temporal", "time", "day", "month", "year"},
    "event": {"event", "conference", "treaty", "document"},
    "product": {"product", "artifact", "artefact"},
    "misc": {"misc", "other", "unknown"},
}


def _canonical_types(raw: str) -> set:
    tokens = set(re.findall(r"[a-z]+", str(raw).lower()))
    found = set()
    for canonical, aliases in _TYPE_ALIASES.items():
        if tokens & aliases:
            found.add(canonical)
    return found


def _type_matches(accepted: set, parsed: set) -> bool:
    if accepted & parsed:
        return True
    # The dev rubric explicitly allows Misc/Other for products and events.
    if "misc" in parsed and (accepted & {"event", "product"}):
        return True
    if "misc" in accepted:
        return True
    return False


def _clean_entity_name(name: str) -> str:
    name = name.strip().strip("`*_\"' ")
    name = re.sub(r"^(?:the)\s+", "", name, flags=re.IGNORECASE)
    return name.strip()


def _parse_entity_line(line: str):
    """Parse one ``Entity: Type``, ``Entity (Type)``, ``Type: Entity`` or table row."""
    line = line.strip().strip("|").strip()
    line = re.sub(r"^[-*+\u2022]\s*", "", line).strip()
    if not line:
        return None

    if line.count("|") >= 2:
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) >= 2:
            for left, right in ((cells[0], cells[1]), (cells[1], cells[0])):
                right_types = _canonical_types(right)
                if right_types:
                    return _clean_entity_name(left), right_types
            return None

    paren = re.match(r"^(.*?)[\(\[]([^)\]]+)[\)\]]\s*$", line)
    if paren:
        types = _canonical_types(paren.group(2))
        if types:
            return _clean_entity_name(paren.group(1)), types

    parts = re.split(r"\s*(?::|\s-\s|\u2013|\u2014)\s*", line, maxsplit=1)
    if len(parts) == 2:
        left, right = parts
        right_types = _canonical_types(right)
        if right_types:
            return _clean_entity_name(left), right_types
        left_types = _canonical_types(left)
        if left_types:
            return _clean_entity_name(right), left_types

    return None


def _parse_entities(answer: str) -> list[tuple[str, set]]:
    parsed = []
    for line in answer.splitlines():
        entity = _parse_entity_line(line)
        if entity and entity[0]:
            parsed.append(entity)
    return parsed


def _check_entities(check: dict, answer: str) -> Verdict:
    parsed = _parse_entities(answer)
    if not parsed:
        return _fail("no 'Entity: Type' lines found", "entities")

    missing = []
    for expected in check.get("entities", []):
        name = normalize_text(expected.get("name", ""))
        if not name:
            continue
        accepted = set()
        for type_name in expected.get("types", []):
            accepted |= _canonical_types(type_name)
        matched = False
        for parsed_name, parsed_types in parsed:
            parsed_norm = normalize_text(parsed_name)
            name_match = (
                name == parsed_norm
                or name in parsed_norm
                or (len(parsed_norm) > 2 and parsed_norm in name)
            )
            if not name_match:
                continue
            if not accepted or _type_matches(accepted, parsed_types):
                matched = True
                break
        if not matched:
            missing.append(expected.get("name", ""))

    if missing:
        return _fail(f"missing or mistyped entities: {missing}", "entities")
    return _pass("all expected entities present with accepted types", "entities")


# --------------------------------------------------------------------- numbers

def _check_number(check: dict, answer: str) -> Verdict:
    expected = float(check.get("value", 0))
    tolerance = float(check.get("tolerance", 1e-6))
    if not any(
        _number_close(value, expected, tolerance) for value in numbers_in(answer)
    ):
        return _fail(f"missing expected numeric result {expected:g}", "number")

    units = check.get("unit_any")
    if units:
        lower = answer.lower()
        if not any(_unit_present(unit, lower) for unit in units):
            return _fail(f"missing expected unit (one of {units})", "number")

    suffix = f" with unit from {units}" if units else ""
    return _pass(f"contains expected numeric result {expected:g}{suffix}", "number")


# ------------------------------------------------------------------------ code

def _extract_code(answer: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks if block.strip())
    for marker in ("def ", "async def "):
        index = answer.find(marker)
        if index >= 0:
            return answer[index:]
    return None


def _check_code(check: dict, answer: str) -> Verdict:
    """Reject clearly missing/invalid code; leave correctness to the judge."""
    code = _extract_code(answer)
    if code is None:
        return _fail("no code block or function definition", "code")

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Prose around a snippet often breaks parsing. Let the judge decide
        # rather than failing an answer whose code is merely interleaved.
        return _undecided("code does not parse in isolation; judge required", "code")

    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return _fail("no function definition found", "code")

    required = check.get("function")
    if required and required not in functions:
        return _fail(f"missing required function {required!r}", "code")

    return _undecided("code parses; correctness still needs the judge", "code")


# ------------------------------------------------------------------- dispatch

_HANDLERS = {
    "constraints": _check_constraints,
    "sentiment_label": _check_sentiment,
    "entities": _check_entities,
    "number": _check_number,
    "requires_code": _check_code,
}

_CHECKS_BY_ID = None


def _checks_by_id() -> dict:
    """Load task checks from the sidecar file, then any inline checks.

    Keeping the checks in ``eval/checks_dev.json`` leaves the prompt/expect
    fixtures tidy and makes a check-only change reviewable on its own. Inline
    ``check`` objects still win, which keeps custom task batches self-contained.
    """
    global _CHECKS_BY_ID
    if _CHECKS_BY_ID is None:
        checks = {}
        try:
            with open(CHECKS_PATH, encoding="utf-8") as fh:
                checks = {
                    str(task_id): check
                    for task_id, check in json.load(fh).items()
                    if isinstance(check, dict)
                }
        except (OSError, ValueError, TypeError):
            checks = {}
        try:
            with open(DEV_TASKS_PATH, encoding="utf-8") as fh:
                for task in json.load(fh):
                    if isinstance(task.get("check"), dict):
                        checks.setdefault(str(task.get("task_id", "")), task["check"])
        except (OSError, ValueError, TypeError):
            pass
        _CHECKS_BY_ID = checks
    return _CHECKS_BY_ID


def check_for(task: dict) -> Optional[dict]:
    """Return the task's inline check, falling back to the dev-set registry."""
    if isinstance(task.get("check"), dict):
        return task["check"]
    return _checks_by_id().get(str(task.get("task_id", "")))


def evaluate(task: dict, answer: str) -> Verdict:
    """Grade one answer deterministically.

    Returns a tri-state :class:`Verdict`. ``authoritative`` is true only when
    the check fully decided the outcome; otherwise the caller should invoke the
    LLM judge.
    """
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)
    if not answer.strip():
        return _fail("empty answer", "empty")

    check = check_for(task)
    if not check:
        return _undecided("no deterministic check for this task", "none")

    kind = str(check.get("type", ""))
    handler = _HANDLERS.get(kind)
    if handler is None:
        return _undecided(f"unknown deterministic check type {kind!r}", kind)

    try:
        return handler(check, answer)
    except Exception as exc:  # a malformed check must not crash an evaluation
        return _undecided(f"deterministic check error: {exc}", kind)


def grade(task: dict, answer: str) -> tuple[bool, str]:
    """Compatibility wrapper for callers that only need a boolean verdict.

    Undecided tasks are reported as failures here. Use :func:`evaluate` when a
    caller can fall back to the LLM judge.
    """
    verdict = evaluate(task, answer)
    return bool(verdict.passed), verdict.reason
