"""Per-category prompt specs for cloud calls.

Design rules:
- System prompts are terse: every system token is billed on every call.
- max_tokens is generous in phase 1 (accuracy first); phase 3 tightens it.
- Categories whose judge expects a specific shape (label + justification,
  exact sentence counts, corrected code) get that shape spelled out in as
  few words as possible.
"""

from dataclasses import dataclass

from agent import router


@dataclass(frozen=True)
class Spec:
    system: str
    max_tokens: int
    # Fireworks reasoning models honor reasoning_effort; "none" disables
    # thinking entirely, which is the single biggest token saver. Categories
    # that fail dev-eval without thinking get it re-enabled selectively.
    effort: str = "none"


SPECS = {
    # factual is also the router's fallback bucket, so its cap must be high
    # enough that a misrouted code/logic task is never truncated mid-answer.
    # A cap is not a target: short factual answers still cost only what they
    # generate.
    router.FACTUAL: Spec(
        system="Answer every part of the question in the fewest words. Plain text.",
        max_tokens=500,
    ),
    router.MATH: Spec(
        system=(
            "Reply with one line: 'EXPR: <python arithmetic computing the "
            "answer> | UNIT: <unit or none>'. Functions allowed: ceil, floor, "
            "round, abs, min, max, sqrt."
        ),
        max_tokens=80,
    ),
    router.SENTIMENT: Spec(
        system="Sentiment label (positive, negative, mixed, or neutral) plus a one-sentence justification. Plain text.",
        max_tokens=100,
    ),
    router.SUMMARIZATION: Spec(
        system=(
            "Obey the stated length and format exactly (a sentence count means "
            "exactly that many sentences), packing in every key fact that fits. "
            "Plain text."
        ),
        max_tokens=200,
    ),
    router.NER: Spec(
        system="List each named entity as 'Entity: Type' (Person, Organization, Location, Date, or other) per line. Countries are Location.",
        max_tokens=200,
    ),
    router.CODE_DEBUG: Spec(
        system="One short sentence naming the bug, then only the corrected code. No comments or explanation.",
        max_tokens=400,
    ),
    router.LOGIC: Spec(
        system="Only essential deductions, then final line 'Answer: <value>'.",
        max_tokens=350,
    ),
    router.CODE_GEN: Spec(
        system="Only the code. No comments, docstrings, examples, or explanation.",
        max_tokens=400,
    ),
}

# Fallback used when a math EXPR answer fails to parse/evaluate: one retry in
# plain step-by-step form. Rare, so its extra tokens are a good trade.
MATH_FALLBACK = Spec(
    system="Minimal working, then final line 'Answer: <value>'.",
    max_tokens=300,
)

# Model preference per category, matched by substring against ALLOWED_MODELS.
# Measured on Fireworks: kimi-k2p7-code has ~54 prompt-token overhead versus
# minimax-m3's ~163 for identical messages (minimax injects a hidden preamble),
# so kimi leads everywhere. Unmatched allowed models are appended as fallbacks
# in their published order.
DEFAULT_PREFERENCE = ["kimi", "minimax"]
MODEL_PREFERENCE = {}
