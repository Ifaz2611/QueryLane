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
    r"(?:extract|identify|list|find|pull out|tag|mark)(?:\s+\w+){0,4}\s+entit|"
    r"entit(?:y|ies).{0,30}\b(?:person|organi[sz]ation|location|date)\b|"
    r"(?:person|people)\b.{0,25}\borgani[sz]ation.{0,30}\b(?:place|location|date)|"
    r"(?:identify|list|extract|find) every "
    r"(?:person|organi[sz]ation|place|location|name)|"
    r"(?:find|identify|list|extract|pull out|tag|mark)\s+(?:all\s+)?(?:the\s+)?"
    r"(?:people|persons?|names?|places?|locations?|organi[sz]ations?|dates?|entities)"
    r"\b.{0,40}\b(?:mentioned|named|in (?:this|the) "
    r"(?:sentence|text|passage|paragraph|excerpt)|below)|"
    r"\bproper nouns?\b.{0,40}\b(?:person|place|organi[sz]ation|type|label)",
    re.IGNORECASE,
)
# Strong code signals only; prose words like "while" or "returns" must not fire.
_CODE_MARKER_RE = re.compile(
    r"def \w+\s*\(|function\s+\w+\s*\(|```|print\s*\(|console\.log\s*\(|"
    r"class\s+\w+|while\s+True\b|for\s+\w+\s+in\s+\w+\s*:|"
    r"=>|[{}]|\breturn \w+|==|\+=|-=|\w+\(\)|\w+\[\w*\]|\blambda \w"
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
    r"\bbrief overview\b|\bshorten\b|\bdistill\b|\bboil down\b|"
    r"\breduce\b.{0,30}\b(?:summary|points?|sentences?|words?|paragraph)\b",
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


# Below this score the router keeps its best-guess category for logging but
# sends the task straight to cloud. A specialist/local answer on an ambiguous
# prompt risks the accuracy gate for a tiny token saving.
MIN_CONFIDENCE = float(os.environ.get("ROUTER_MIN_CONFIDENCE", "0.70"))
CONFIDENCE_THRESHOLD = MIN_CONFIDENCE

# Earlier checks win ties. Scores encode both signal strength and that order.
_CATEGORY_PRIORITY = [
    NER,
    CODE_DEBUG,
    CODE_GEN,
    SUMMARIZATION,
    SENTIMENT,
    LOGIC,
    MATH,
    FACTUAL,
]

_STRONG = {
    NER: 0.98,
    CODE_DEBUG: 0.97,
    CODE_GEN: 0.94,
    SUMMARIZATION: 0.92,
    SENTIMENT: 0.90,
    LOGIC: 0.84,
    MATH: 0.86,
}
_LENGTH_CUE_SCORE = 0.72


def _signals(prompt: str) -> dict:
    """Positive evidence per category; absence means no cue matched."""
    scores = {}
    has_code = bool(_CODE_MARKER_RE.search(prompt))

    def bump(category: str, score: float):
        scores[category] = max(scores.get(category, 0.0), score)

    if _NER_RE.search(prompt):
        bump(NER, _STRONG[NER])
    if has_code and _DEBUG_RE.search(prompt):
        bump(CODE_DEBUG, _STRONG[CODE_DEBUG])
    if _CODE_GEN_RE.search(prompt):
        bump(CODE_GEN, _STRONG[CODE_GEN])
    if _SUMMARY_VERB_RE.search(prompt):
        bump(SUMMARIZATION, _STRONG[SUMMARIZATION])
    if _SENTIMENT_RE.search(prompt):
        bump(SENTIMENT, _STRONG[SENTIMENT])
    if _LOGIC_RE.search(prompt) and not _looks_numeric(prompt):
        bump(LOGIC, _STRONG[LOGIC])
    if _MATH_CUE_RE.search(prompt) and len(_NUMBER_RE.findall(prompt)) >= 2:
        bump(MATH, _STRONG[MATH])
    # A bare length cue only implies summarization when there is an actual
    # passage attached; short prompts are constrained factual questions.
    if _LENGTH_CUE_RE.search(prompt) and not has_code and len(prompt) > 200:
        bump(SUMMARIZATION, _LENGTH_CUE_SCORE)
    return scores


def _best_category(scores: dict) -> str:
    if not scores:
        return FACTUAL
    return max(
        scores,
        key=lambda category: (
            scores[category],
            -_CATEGORY_PRIORITY.index(category),
        ),
    )


def _confidence(category: str, scores: dict) -> float:
    if category == FACTUAL or category not in scores:
        return 0.30
    score = scores[category]
    others = [value for other, value in scores.items() if other != category]
    if not others:
        return round(min(1.0, score), 3)
    second = max(others)
    margin = max(0.0, score - second)
    ratio = second / score if score else 0.0
    confidence = score * (0.70 + 0.30 * min(margin / 0.25, 1.0))
    if ratio >= 0.85:  # near-tie: another category explains the prompt too
        confidence *= 0.90
    return round(max(0.0, min(1.0, confidence)), 3)


def classify_with_confidence(prompt: str):
    """Return (category, confidence) without changing the selected category."""
    scores = _signals(prompt)
    category = _best_category(scores)
    return category, _confidence(category, scores)


def classify(prompt: str) -> str:
    return classify_with_confidence(prompt)[0]


def route_with_confidence(prompt: str):
    """Return (category, target, confidence).

    `target` is always cloud when confidence is below MIN_CONFIDENCE, even if
    the category is configured for local handling.
    """
    category, confidence = classify_with_confidence(prompt)
    target = "local" if category in LOCAL_CATEGORIES else "cloud"
    if confidence < MIN_CONFIDENCE:
        target = "cloud"
    return category, target, confidence


def route(prompt: str, with_confidence: bool = False):
    """Return (category, target); with_confidence=True adds the score."""
    result = route_with_confidence(prompt)
    return result if with_confidence else result[:2]
