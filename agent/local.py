"""Local model wrapper (phase 2).

Answers cost zero Fireworks tokens. The wrapper is inert until two things
exist: the llama-cpp-python dependency and a GGUF file at LOCAL_MODEL_PATH.
That lets the same code ship in the phase 1 image without the weights.

Grading environment is 4 GB RAM / 2 vCPU with no GPU, so keep the model at
about 2 GB quantized and the context small.
"""

import os
import re
import threading
import time

from agent import pal

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/model.gguf")

# Small models mislabel document-like entities as Organization; named legal
# instruments are events/documents, never organizations. General rule, applied
# to whatever entities the model finds.
_DOCUMENT_ENTITY_RE = re.compile(
    r"^(?P<entity>[^:]*\b(?:treaty|agreement|accord|pact|protocol|declaration|"
    r"charter|convention)\b[^:]*):\s*organization\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Countries labeled Organization (a recurring model error in political/treaty
# contexts, e.g. 'Germany: Organization'). Only bare country names are
# rewritten; 'Bank of England' etc. never matches because the whole entity
# must be the country name.
_COUNTRIES = (
    "afghanistan|argentina|australia|austria|belgium|brazil|canada|chile|china|"
    "colombia|cuba|denmark|egypt|england|ethiopia|finland|france|germany|ghana|"
    "greece|hungary|india|indonesia|iran|iraq|ireland|israel|italy|japan|kenya|"
    "mexico|morocco|netherlands|nigeria|norway|pakistan|peru|philippines|poland|"
    "portugal|romania|russia|saudi arabia|scotland|singapore|south africa|"
    "south korea|spain|sweden|switzerland|thailand|turkey|ukraine|"
    "united kingdom|united states|uruguay|venezuela|vietnam|wales"
)
_COUNTRY_ENTITY_RE = re.compile(
    rf"^(?P<entity>(?:the )?(?:{_COUNTRIES})):\s*organization\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_lock = threading.Lock()
_llm = None
_import_ok = None

# Wall-clock budget for ALL local inference combined. Grading hardware speed
# is unknown; when the budget runs out, remaining tasks fall back to cloud
# instead of risking the 10-minute container limit.
TIME_BUDGET = float(os.environ.get("LOCAL_TIME_BUDGET", "300"))
# Minimum generation speed measured at warmup; below this the VM is too slow
# for ANY local inference. Set at the speed where even a PAL math answer
# (~15 tokens) stops fitting sensibly. Category bars below gate the rest.
# Measured: the grader runs the 1.5B at ~1.2 tok/s (our 0.25-CPU throttle).
MIN_TOKENS_PER_SEC = float(os.environ.get("LOCAL_MIN_TPS", "0.4"))

# Category-level fallback budgets: prevent any single slow category from
# consuming the global local time budget and locking out other local categories.
DEFAULT_CATEGORY_TIME_BUDGET = float(
    os.environ.get("LOCAL_CATEGORY_TIME_BUDGET", "120.0")
)

_CATEGORY_TIME_BUDGETS = {
    "math": float(os.environ.get("LOCAL_CATEGORY_BUDGET_MATH", "100.0")),
    "ner": float(os.environ.get("LOCAL_CATEGORY_BUDGET_NER", "100.0")),
    "sentiment": float(os.environ.get("LOCAL_CATEGORY_BUDGET_SENTIMENT", "60.0")),
    "factual": float(os.environ.get("LOCAL_CATEGORY_BUDGET_FACTUAL", "90.0")),
    "summarization": float(os.environ.get("LOCAL_CATEGORY_BUDGET_SUMMARIZATION", "60.0")),
    "logic": float(os.environ.get("LOCAL_CATEGORY_BUDGET_LOGIC", "60.0")),
    "code_debug": float(os.environ.get("LOCAL_CATEGORY_BUDGET_CODE_DEBUG", "60.0")),
    "code_gen": float(os.environ.get("LOCAL_CATEGORY_BUDGET_CODE_GEN", "60.0")),
}

# Summarization and logic regressed in earlier A/B runs (v13 summarization: 0/5
# due to hallucination/length violations; v14 logic: 0/5 deduction failures).
# Both remain disabled (bar 99.0 TPS) unless an explicit fine-tune evaluation
# proves them above the 80% accuracy gate.
PROVEN_CATEGORIES = set(
    c.strip().lower()
    for c in os.environ.get("LOCAL_PROVEN_CATEGORIES", "").split(",")
    if c.strip()
)

_budget_lock = threading.Lock()
_time_spent = 0.0
_category_time_spent: dict = {}
_speed_ok = None  # None = not yet calibrated
_tps = 0.0
_calibrated = threading.Event()

def category_budget(category: str) -> float:
    """Return the time budget for a specific category in seconds."""
    env_key = f"LOCAL_CATEGORY_BUDGET_{category.upper()}"
    if env_key in os.environ:
        try:
            return float(os.environ[env_key])
        except ValueError:
            pass
    return _CATEGORY_TIME_BUDGETS.get(category, DEFAULT_CATEGORY_TIME_BUDGET)


def category_time_spent(category: str) -> float:
    """Return wall-clock seconds spent answering tasks in this category."""
    with _budget_lock:
        return _category_time_spent.get(category, 0.0)


def category_budget_remaining(category: str) -> float:
    """Remaining time budget for a specific category."""
    budget = category_budget(category)
    spent = category_time_spent(category)
    return max(0.0, budget - spent)


def reset_budgets() -> None:
    """Reset global and per-category time-spent counters."""
    global _time_spent
    with _budget_lock:
        _time_spent = 0.0
        _category_time_spent.clear()


def category_proven(category: str) -> bool:
    """True if this category is proven by fine-tune A/B evaluation.

    Local summarization and logic regressed in earlier A/B runs (summarization:
    0/5 due to hallucination/length violations; v14 logic: 0/5 deduction failures).
    Both are blocked by default until an explicit fine-tune evaluation proves them.
    """
    if category in ("summarization", "logic"):
        if category in PROVEN_CATEGORIES:
            return True
        env_flag = os.environ.get(f"PROVE_LOCAL_{category.upper()}", "").lower()
        return env_flag in ("1", "true", "yes")
    return True

# Per-category speed bars: categories degrade individually instead of local
# switching off wholesale. Bars follow output length: math-PAL emits ~15
# tokens (viable at crawl speed), NER ~40-60, factual ~60-120 (needs a faster
# VM or it eats the whole budget / truncates).
_CATEGORY_MIN_TPS = {
    "math": 0.4,
    "ner": 0.9,
    "factual": 1.3,
    "sentiment": 0.9,
    "summarization": 99.0,  # never local (measured regression)
    "logic": 99.0,          # never local (measured 0/5)
}

# Few-shot examples are free locally; use them to steady the small model.
_LOCAL_SYSTEM = {
    "sentiment": (
        "Classify the sentiment of the review as positive, negative, mixed, or "
        "neutral, then justify in one short sentence. Use mixed when there is "
        "both praise and complaint. Reported problems (delays, damage, defects) "
        "are negative even when stated factually. Use neutral only when the "
        "review expresses neither satisfaction nor any problem."
    ),
    "ner": (
        "Extract every named entity from the text and label each with its type: "
        "Person, Organization, Location, Date, Event, or Product. One "
        "'Entity: Type' per line. Label carefully: conferences and treaties are "
        "Event, countries and cities are Location, companies and institutions "
        "are Organization."
    ),
    "summarization": (
        "Summarize the text accurately. Obey the stated length and format "
        "exactly. Keep every concrete detail that fits: numbers, quantities, "
        "names, places, outcomes, and stated purposes."
    ),
    "factual": "Answer the question accurately and concisely.",
    "math": (
        "Output exactly one line: 'EXPR: <arithmetic expression> | UNIT: <unit>' "
        "where the expression computes the numeric answer and unit is the "
        "answer's unit ($, km/h, items, grams, ...) or none. Allowed in the "
        "expression: numbers, + - * / // % ** parentheses, ceil, floor, round, "
        "abs, min, max, sqrt. No prose."
    ),
    "logic": (
        "Work through the constraints step by step, keeping each step short. "
        "End with a final line 'Answer: <value>'."
    ),
}

# Optional example turns inserted before the real prompt, per category.
_LOCAL_FEWSHOT = {
    "ner": [
        {
            "role": "user",
            "content": (
                "Extract all named entities and their types from: Elon Musk "
                "spoke at CES in Las Vegas on 8 January about Tesla and the "
                "Paris Agreement."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Elon Musk: Person\nCES: Event\nLas Vegas: Location\n"
                "8 January: Date\nTesla: Organization\nParis Agreement: Event"
            ),
        },
    ],
}

_MAX_TOKENS = {
    "sentiment": 60,
    "ner": 150,
    "summarization": 160,
    "factual": 200,
    "math": 60,  # program-aided: one short EXPR line
    "logic": 400,
}


def available() -> bool:
    global _import_ok
    if not os.path.exists(MODEL_PATH):
        return False
    if _import_ok is None:
        try:
            import llama_cpp  # noqa: F401

            _import_ok = True
        except ImportError:
            _import_ok = False
    if not _import_ok:
        return False
    with _budget_lock:
        spent = _time_spent
    if _speed_ok is False or spent >= TIME_BUDGET:
        return False
    return True


def _get_llm():
    global _llm
    with _lock:
        if _llm is None:
            from llama_cpp import Llama

            _llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=int(os.environ.get("LOCAL_THREADS", "2")),
                verbose=False,
            )
    return _llm


def _calibrate() -> None:
    """Load the model and measure generation speed with a tiny completion.
    Sets _speed_ok; a too-slow VM disables local inference entirely."""
    global _speed_ok, _time_spent, _tps
    started = time.time()
    try:
        llm = _get_llm()
        with _lock:
            # First generation absorbs cold mmap page-in; do not measure it.
            # Kept tiny: on a crawling VM calibration itself burns the budget.
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "Count: 1 2 3"}],
                temperature=0,
                max_tokens=8,
            )
            gen_start = time.time()
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": "List: a b c d"}],
                temperature=0,
                max_tokens=16,
            )
            elapsed = time.time() - gen_start
        # Use the real completion count. Assuming max_tokens would overestimate
        # throughput whenever the model stops early, which can wrongly enable
        # local inference on a slow grader.
        usage = (result.get("usage") or {}) if isinstance(result, dict) else {}
        generated = usage.get("completion_tokens") or 16
        _tps = float(generated) / max(elapsed, 0.01)
        _speed_ok = _tps >= MIN_TOKENS_PER_SEC
        print(
            f"local calibration: {_tps:.1f} tok/s "
            f"({'enabled' if _speed_ok else 'DISABLED, too slow'}), "
            f"load+warmup {time.time() - started:.1f}s",
            flush=True,
        )
    except Exception as exc:
        _speed_ok = False
        print(f"local calibration failed, local disabled: {exc}", flush=True)
    finally:
        elapsed = time.time() - started
        with _budget_lock:
            _time_spent += elapsed
        _calibrated.set()


_preload_started = False
_preload_lock = threading.Lock()


def preload() -> None:
    """Start loading + calibrating in the background so the first local task
    does not pay the cold-start cost serially. Idempotent."""
    global _preload_started
    with _preload_lock:
        if _preload_started or not os.path.exists(MODEL_PATH):
            return
        _preload_started = True
    threading.Thread(target=_calibrate, daemon=True).start()


def ready() -> bool:
    """True once calibration has finished and approved local inference."""
    with _budget_lock:
        spent = _time_spent
    return _speed_ok is True and spent < TIME_BUDGET


def category_enabled(category: str) -> bool:
    """Whether the measured speed clears this category's bar, its category-level
    fallback budget has not been exhausted, and (for unproven categories like
    summarization and logic) it has been proven by an A/B evaluation."""
    if not ready():
        return False
    if not category_proven(category):
        return False
    if _tps < _CATEGORY_MIN_TPS.get(category, MIN_TOKENS_PER_SEC):
        return False
    with _budget_lock:
        if _category_time_spent.get(category, 0.0) >= category_budget(category):
            return False
    return True


def wait_ready(timeout: float) -> bool:
    """Block until calibration finishes (or timeout), then report readiness.
    Cloud-routed work never calls this; only local-routed tasks wait."""
    preload()  # no-op if already started; covers callers that skip preload
    _calibrated.wait(timeout)
    return ready()


_ANSWER_LINE_RE = re.compile(r"answer:\s*\S*\d", re.IGNORECASE)
_NER_TYPES = ("person", "organization", "location", "date", "event", "product")

# Completeness guard for local NER. The model's one failure mode is dropping
# entities, so every capitalized span and date in the passage must be covered
# by the answer or the task escalates to cloud. Over-escalation costs ~110
# tokens; under-escalation costs an accuracy-gate miss.
_STOP_STARTERS = {
    "the", "a", "an", "in", "on", "at", "by", "for", "of", "to", "and", "but",
    "during", "after", "before", "when", "while", "extract", "identify",
    "list", "find", "he", "she", "it", "they", "we", "i", "his", "her", "its",
    "their", "this", "that", "these", "those", "there", "here", "who", "what",
    "dr", "mr", "mrs", "ms", "prof",
    # Role titles preceding names: the answer names the person, not the role.
    "ceo", "cto", "cfo", "coo", "founder", "cofounder", "analyst",
    "president", "chancellor", "secretary", "minister", "director",
    "professor", "judge", "senator", "mayor", "captain", "coach",
    "chairman", "chairwoman", "spokesperson", "spokesman", "spokeswoman",
    "governor", "general", "researcher", "scientist", "economist",
}
_WORD_RE = re.compile(r"[A-Za-z][\w.'&-]*")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _norm(text: str) -> str:
    """Normalize for containment checks: lowercase, alnum+spaces only."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _capitalized_spans(passage: str):
    """Group consecutive capitalized words (allowing internal of/the/and
    connectors) into candidate entity spans. Sentence punctuation between
    words breaks the span so entities never merge across sentences."""
    words = list(_WORD_RE.finditer(passage))
    spans, current = [], []
    for i, match in enumerate(words):
        word = match.group(0)
        boundary = False
        if current:
            gap = passage[current[-1][1].end(): match.start()]
            boundary = any(ch in gap for ch in ".!?;:,()\"")
        if boundary and current:
            spans.append(current)
            current = []
        if word[0].isupper():
            current.append((i, match))
        elif current and word.lower() in ("of", "the", "and", "de") and (
            i + 1 < len(words) and words[i + 1].group(0)[0].isupper()
        ):
            current.append((i, match))
        else:
            if current:
                spans.append(current)
                current = []
    if current:
        spans.append(current)

    texts = []
    for span in spans:
        # Trim leading stopwords/instruction verbs (incl. sentence starters).
        while span and _norm(span[0][1].group(0)) in _STOP_STARTERS:
            span = span[1:]
        while span and span[-1][1].group(0).lower() in ("of", "the", "and", "de"):
            span = span[:-1]
        if not span:
            continue
        text = passage[span[0][1].start(): span[-1][1].end()]
        normalized = _norm(text)
        if normalized in _STOP_STARTERS or len(normalized) < 2:
            continue
        texts.append(text)
    return texts


def ner_complete(prompt: str, answer: str) -> bool:
    """True when every heuristic entity candidate in the passage is covered
    by the answer text."""
    from agent import specialists

    passage = specialists.extract_passage(prompt)
    answer_norm = _norm(answer)
    for span in _capitalized_spans(passage):
        if _norm(span) not in answer_norm:
            return False
    for year in _YEAR_RE.findall(passage):
        if year not in answer:
            return False
    return True


def answer_ok(category: str, answer: str, prompt: str = "") -> bool:
    """Cheap plausibility check on a local answer. False means escalate to
    cloud: paying ~100 tokens beats risking a miss against the accuracy gate.
    """
    if not answer.strip():
        return False
    if category == "sentiment":
        # Observed real-set failure mode: factual-toned complaints mislabeled
        # neutral. True neutrals are rare; let the cloud confirm them.
        return "neutral" not in answer.lower()
    if category == "ner":
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if not lines:
            return False
        for line in lines:
            if ":" not in line:
                return False
            type_part = line.rsplit(":", 1)[1].strip().lower()
            if not any(t in type_part for t in _NER_TYPES):
                return False
        return not prompt or ner_complete(prompt, answer)
    if category == "math":
        return bool(_ANSWER_LINE_RE.search(answer))
    return True


def answer(category: str, prompt: str) -> str:
    global _time_spent
    if not ready() or not category_proven(category):
        return ""  # budget exhausted, unproven, or too slow; caller falls back to cloud
    with _budget_lock:
        if _category_time_spent.get(category, 0.0) >= category_budget(category):
            return ""  # category-level fallback budget exhausted
    started = time.time()
    try:
        llm = _get_llm()
        system = _LOCAL_SYSTEM.get(category, _LOCAL_SYSTEM["factual"])
        messages = [{"role": "system", "content": system}]
        messages.extend(_LOCAL_FEWSHOT.get(category, []))
        messages.append({"role": "user", "content": prompt})
        # Hard per-answer time backstop: cap generated tokens to ~75s of
        # measured speed so one generation can never blow the budget.
        cap = _MAX_TOKENS.get(category, 200)
        if _tps:
            cap = min(cap, max(24, int(_tps * 75)))
        with _lock:  # llama.cpp context is not thread-safe
            result = llm.create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=cap,
            )
    finally:
        elapsed = time.time() - started
        with _budget_lock:
            _time_spent += elapsed
            _category_time_spent[category] = (
                _category_time_spent.get(category, 0.0) + elapsed
            )
    text = (result["choices"][0]["message"]["content"] or "").strip()
    if category == "math":
        # Evaluate the emitted expression; None (bad expr) escalates to cloud.
        return pal.answer_from_expression(text) or ""
    return normalize(category, text)


def normalize(category: str, text: str) -> str:
    """Type-correct known model mislabels. Applies to any answer source."""
    if category == "ner":
        text = _DOCUMENT_ENTITY_RE.sub(lambda m: m.group("entity") + ": Event", text)
        text = _COUNTRY_ENTITY_RE.sub(lambda m: m.group("entity") + ": Location", text)
    return text
