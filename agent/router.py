"""Zero-token heuristic router.

Classifies each task prompt into one of the eight evaluation categories using
lexical cues only. No model call is made, so routing costs nothing toward the
token score. When unsure it falls through to `factual`, which is handled by the
cloud path (safest for the accuracy gate).

Check order matters: instruction verbs (extract entities, fix this bug, write a
function, summarize) are checked before cues that can occur inside a quoted
passage (sentiment words, numbers, puzzle phrasing).
"""

import os
import re

FACTUAL = "factual"
MATH = "math"
SENTIMENT = "sentiment"
SUMMARIZATION = "summarization"
NER = "ner"
CODE_DEBUG = "code_debug"
LOGIC = "logic"
CODE_GEN = "code_gen"

ALL_CATEGORIES = [
    FACTUAL,
    MATH,
    SENTIMENT,
    SUMMARIZATION,
    NER,
    CODE_DEBUG,
    LOGIC,
    CODE_GEN,
]

# Categories served by the bundled local model when its weights are present.
# Phase 2 promotes categories one at a time, after dev-eval proves them. The
# default is baked into the image via ENV; empty means cloud-only.
LOCAL_CATEGORIES: set = set(
    c.strip()
    for c in os.environ.get("LOCAL_CATEGORIES", "").split(",")
    if c.strip()
)

_NER_RE = re.compile(
    r"named entit|entities and their types|"
    r"(?:extract|identify|list|find)(?:\s+\w+){0,4}\s+entit|"
    r"entit(?:y|ies).{0,30}\b(?:person|organi[sz]ation|location|date)\b|"
    r"(?:person|people)\b.{0,25}\borgani[sz]ation.{0,30}\b(?:place|location|date)|"
    r"(?:identify|list|extract|find) every (?:person|organi[sz]ation|place|location|name)",
    re.IGNORECASE,
)
# Strong code signals only; prose words like "while" or "returns" must not fire.
_CODE_MARKER_RE = re.compile(
    r"def \w+\s*\(|function\s+\w+\s*\(|```|print\s*\(|=>|[{}]|"
    r"\breturn \w+|==|\+=|-=|\w+\(\)|\w+\[\w*\]|\blambda \w"
)
_DEBUG_RE = re.compile(
    r"\bbugg?y?\b|\bfix (?:it|this|the)\b|find and fix|\bdebug\b|"
    r"\bcorrect(?:ed)? (?:it|the |this |implementation|version|code)|"
    r"\berror\b|\brepair\b|\bbroken\b|"
    r"crash(?:es)?\b|fails?\b|wrong (?:output|result)|never stops|infinite loop",
    re.IGNORECASE,
)
_CODE_GEN_RE = re.compile(
    r"\b(?:write|create|build|develop|make|design|implement|code)\b"
    r"(?:\s+\w+){0,3}\s*"
    r"(?:python |javascript |js |java |c\+\+ )?(?:function|program|script|method|class|algorithm)\b",
    re.IGNORECASE,
)
_SUMMARY_VERB_RE = re.compile(
    r"\bsummari[sz]e\b|\btl;?dr\b|\bcondense\b|\bsummary\b|\brecap\b|"
    r"\bbrief overview\b|\bshorten\b|\bdistill\b|\bboil down\b",
    re.IGNORECASE,
)
_LENGTH_CUE_RE = re.compile(
    r"in (?:exactly )?(?:one|two|three|\d+) sentences?|"
    r"in (?:no more than |at most |under |fewer than )?\d+ words",
    re.IGNORECASE,
)
_SENTIMENT_RE = re.compile(
    r"\bsentiment\b|classify (?:this|the following|the) review|"
    r"positive,? negative,? or (?:neutral|mixed)|"
    r"\btone\b.{0,40}\b(?:review|feedback|text|comment)|"
    r"(?:review|feedback|comment).{0,40}\btone\b|"
    r"favou?rable,? unfavou?rable|"
    r"(?:positive|negative) or (?:negative|positive)",
    re.IGNORECASE,
)
_LOGIC_RE = re.compile(
    r"each (?:own|has|have|drink|eat|play|wear|live|sit)|"
    r"\bpuzzle\b|\bdeduce\b|"
    r"(?:does not|doesn't|do not|is not|isn't) (?:own|like|sit|live|play|finish|next|on)|"
    r"who (?:owns|has|likes|finish(?:ed|es)|sits|lives|plays|wears|drinks|is last|is first|came)|"
    r"who is the (?:second|third|tallest|shortest|oldest|youngest|fastest|slowest)|"
    r"who (?:sits|is) (?:in )?the middle|"
    r"\b(?:taller|shorter|older|younger|faster|slower|heavier|lighter) than\b|"
    r"\bbeats?\b.{0,30}\bbeats?\b|finish(?:es)? (?:ahead of|before|after)|"
    r"immediately (?:to the )?(?:left|right)|directly (?:to the )?(?:left|right)|"
    r"\bfar (?:left|right)\b|"
    r"\bnext to\b|from left to right|"
    r"\bseated?\b.{0,30}\brow\b|sit(?:s)? (?:in|next to|beside)|"
    r"all (?:of the |the )?(?:conditions|constraints|clues)",
    re.IGNORECASE,
)
_MATH_CUE_RE = re.compile(
    r"\bhow (?:many|much|far|long|old)\b|\bpercent(?:age)?\b|%|"
    r"\bcalculate\b|\bcompute\b|\bremain\b|\bleft over\b|\bin total\b|"
    r"\baverage\b|\bper (?:hour|week|month|day|year|person)\b|"
    r"\bdiscount\b|\bprofit\b|\bcost(?:s)?\b|\bprice\b|\bspeed\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _looks_numeric(prompt: str) -> bool:
    """True when the prompt leans on arithmetic (a math cue plus several
    numbers), which separates word problems from pure deduction puzzles."""
    return bool(_MATH_CUE_RE.search(prompt)) and len(_NUMBER_RE.findall(prompt)) >= 2


def classify(prompt: str) -> str:
    has_code = bool(_CODE_MARKER_RE.search(prompt))

    if _NER_RE.search(prompt):
        return NER
    if has_code and _DEBUG_RE.search(prompt):
        return CODE_DEBUG
    if _CODE_GEN_RE.search(prompt):
        return CODE_GEN
    if _SUMMARY_VERB_RE.search(prompt):
        return SUMMARIZATION
    if _SENTIMENT_RE.search(prompt):
        return SENTIMENT
    if _LOGIC_RE.search(prompt) and not _looks_numeric(prompt):
        return LOGIC
    if _MATH_CUE_RE.search(prompt) and len(_NUMBER_RE.findall(prompt)) >= 2:
        return MATH
    if _LOGIC_RE.search(prompt):
        return LOGIC
    # A bare length cue only implies summarization when there is an actual
    # passage attached; short prompts are constrained factual questions.
    if _LENGTH_CUE_RE.search(prompt) and not has_code and len(prompt) > 200:
        return SUMMARIZATION
    return FACTUAL


def route(prompt: str):
    """Return (category, target) where target is 'local' or 'cloud'."""
    category = classify(prompt)
    target = "local" if category in LOCAL_CATEGORIES else "cloud"
    return category, target
