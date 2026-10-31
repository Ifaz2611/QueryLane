"""Non-generative specialists: ONNX classifiers for NER and sentiment.

Unlike the generative local model, these run in ~1s per task on any CPU, so
they work even on the slow grading VM where token-by-token generation is too
slow. Zero Fireworks tokens. Every uncertain case returns "" and escalates to
the cloud path, protecting the accuracy gate.

Models (int8-quantized ONNX, bundled in the image):
- models/ner:  bert-base-NER (B/I tags for PER, ORG, LOC, MISC) + date regex
- models/sst2: distilbert SST-2 (positive/negative) + clause aggregation
"""

import json
import os
import re
import threading

try:
    import numpy as np
except ImportError:  # cloud-only image: specialists stay disabled
    np = None

NER_DIR = os.environ.get("NER_MODEL_DIR", "/app/models/ner")
SST2_DIR = os.environ.get("SST2_MODEL_DIR", "/app/models/sst2")

_lock = threading.Lock()
_sessions = {}


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _load(kind: str):
    """Lazy-load (session, tokenizer, id2label) for 'ner' or 'sst2'."""
    with _lock:
        if kind in _sessions:
            return _sessions[kind]
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = NER_DIR if kind == "ner" else SST2_DIR
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            os.path.join(directory, "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        _sessions[kind] = (session, tokenizer, id2label)
        return _sessions[kind]


def available(kind: str) -> bool:
    if np is None:
        return False
    directory = NER_DIR if kind == "ner" else SST2_DIR
    if not os.path.exists(os.path.join(directory, "model.onnx")):
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        return True
    except ImportError:
        return False


def _run(kind: str, text: str):
    session, tokenizer, id2label = _load(kind)
    encoding = tokenizer.encode(text)
    ids = encoding.ids[:512]
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
    }
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
    logits = session.run(None, feed)[0][0]
    return encoding, ids, logits, id2label


# ---------------------------------------------------------------- sentiment

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|,\s*(?:but|though|although|however|yet)\s+", re.IGNORECASE)
# SST2 has no neutral class and is confidently wrong on factual statements
# and hedged language. Verdicts require explicit cue words agreeing with the
# classifier, and hedges escalate outright. Measured on 597 ground-truth
# reviews: these gates trade coverage for precision.
# Every cue starts at a word boundary: without it, "uncomfortable" matches
# "comfortable", "imperfect" matches "perfect", and "glove" matches "love".
# "crush" is deliberately narrowed to "crushed": "crushes ice" is praise.
_POS_CUE_RE = re.compile(
    r"\blove|\bgreat|\bexcellent|\bamazing|\bfantastic|\bwonderful|\bperfect|"
    r"\bawesome|\benjoy|\bimpress|\brecommend|\bsatisf|\bdelight|\bbreeze|"
    r"\bsuperb|\bcomfortable|\bspotless|\bfriendly|\bcrisp|\bflawless|"
    r"\bdream\b|\bhighly",
    re.IGNORECASE,
)
_NEG_CUE_RE = re.compile(
    r"\bterrible|\bawful|\bhorrible|\bworst|\bhate|\bdisappoint|\bpoor\b|"
    r"\bbroken|\bcrush(?:ed)?\b|\blate\b|\bdelay|\bslow|\bfail|\bcrash|"
    r"\bscratch|\bdefect|\brefund|\breturn|\bwaste|\buseless|\bannoy|"
    r"\bfrustrat|\bleak|\bcheap|\bdirty|\brude|\bmuddy|\bunbearable|"
    r"\bstopped working|\bdoesn.t work|\bnever (?:arrived|worked|again)",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\bnot bad\b|\bnothing (?:special|terrible|more|less)\b|\baverage\b|"
    r"\bdecent\b|\bokay\b|\bok\b|\bfine\b|\bdoes (?:what it says|the job)\b|"
    r"\bas expected\b|\bit works\b|\badequate\b|\bmediocre\b|\bso-so\b|"
    r"\bmiddle of the road\b|\bcan.t complain\b",
    re.IGNORECASE,
)
# A contrast conjunction almost always signals a two-sided review; one-sided
# verdicts are unsafe there, only a confirmed Mixed (or escalation) is.
_CONTRAST_RE = re.compile(
    r"\b(?:but|though|although|however|yet|except|aside from|other than)\b",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(
    r"^.*?(?:sentiment of this review|sentiment of the review|the following review|this review)\s*[:.]?\s*",
    re.IGNORECASE | re.DOTALL,
)


def extract_review(prompt: str) -> str:
    """Drop the instruction wrapper, keep the review text."""
    stripped = _PREFIX_RE.sub("", prompt, count=1).strip()
    return stripped if stripped else prompt


def _classify_clause(text: str):
    _, _, logits, id2label = _run("sst2", text)
    probs = _softmax(logits.reshape(-1))
    index = int(np.argmax(probs))
    return id2label.get(index, "POSITIVE"), float(probs[index])


def sentiment_answer(prompt: str) -> str:
    """Label + templated justification, or '' to escalate."""
    review = extract_review(prompt)
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(review) if len(c.strip()) >= 4]
    if not clauses:
        clauses = [review]
    if _HEDGE_RE.search(review):
        return ""  # hedged language: SST2 is unreliable here, escalate

    votes = []
    for clause in clauses[:8]:
        label, confidence = _classify_clause(clause)
        votes.append((label, confidence, clause))

    # A side only counts when the classifier AND an explicit cue word agree
    # on the same clause; classifier-only signals escalate.
    strong_pos = [
        v for v in votes
        if v[0] == "POSITIVE" and v[1] >= 0.9 and _POS_CUE_RE.search(v[2])
    ]
    strong_neg = [
        v for v in votes
        if v[0] == "NEGATIVE" and v[1] >= 0.9 and _NEG_CUE_RE.search(v[2])
    ]

    whole_label, whole_conf = _classify_clause(review)
    has_pos_cue = bool(_POS_CUE_RE.search(review))
    has_neg_cue = bool(_NEG_CUE_RE.search(review))

    if strong_pos and strong_neg:
        return (
            "Mixed, the review contains both praise "
            f"({_quotes(strong_pos)}) and criticism ({_quotes(strong_neg)})."
        )
    if _CONTRAST_RE.search(review):
        return ""  # two-sided phrasing without confirmed Mixed: escalate
    if (
        whole_conf >= 0.97
        and whole_label == "POSITIVE"
        and has_pos_cue
        and not has_neg_cue
        and not strong_neg
    ):
        return (
            "Positive, the review expresses satisfaction "
            f"({_quotes(strong_pos) or _quote(review)}) with no complaints."
        )
    if (
        whole_conf >= 0.97
        and whole_label == "NEGATIVE"
        and has_neg_cue
        and not has_pos_cue
        and not strong_pos
    ):
        return (
            "Negative, the review reports problems "
            f"({_quotes(strong_neg) or _quote(review)}) with no praise."
        )
    return ""  # ambiguous: escalate to cloud


def _quote(text: str, limit: int = 90) -> str:
    text = text.strip().rstrip(".")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f'"{text}"'


def _quotes(votes, limit: int = 90) -> str:
    return "; ".join(_quote(v[2], limit) for v in votes[:3])


# ---------------------------------------------------------------------- ner

_NER_PREFIX_RE = re.compile(
    r"^.*?(?:entities and their types from|entities and their types in this sentence|"
    r"entities from|entities in|types from|entit(?:y|ies)[^:]*)[:.]\s*",
    re.IGNORECASE | re.DOTALL,
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_MONTH_NAMES = frozenset(_MONTHS.split("|"))
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\b\s+\d{{1,2}}(?!\d)(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\b\s+\d{{4}}\b"
    rf"|(?:last|next|this|early|late)\s+(?:{_MONTHS})\b"
    rf"|(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    rf"|\b(?:19|20)\d{{2}}\b"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})",
    re.IGNORECASE,
)
_DOC_EVENT_RE = re.compile(
    r"treaty|agreement|accord|pact|protocol|declaration|charter|convention|summit|"
    r"conference|olympics|festival|expo|championship|cup\b",
    re.IGNORECASE,
)
_TYPE_NAMES = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": None}


def extract_passage(prompt: str) -> str:
    stripped = _NER_PREFIX_RE.sub("", prompt, count=1).strip()
    return stripped if stripped else prompt


def ner_answer(prompt: str) -> str:
    """'Entity: Type' lines, or '' to escalate."""
    passage = extract_passage(prompt)
    encoding, ids, logits, id2label = _run("ner", passage)
    probs = _softmax(logits, axis=-1)

    entities = []  # (text, tag, min_confidence, start, end)
    current_tokens, current_tag, current_conf = [], None, 1.0
    offsets = encoding.offsets[: len(ids)]

    def flush():
        nonlocal current_tokens, current_tag, current_conf
        if current_tokens and current_tag:
            start = current_tokens[0][0]
            end = current_tokens[-1][1]
            text = passage[start:end].strip()
            if text:
                entities.append((text, current_tag, current_conf, start, end))
        current_tokens, current_tag, current_conf = [], None, 1.0

    for position, (start, end) in enumerate(offsets):
        if end == 0:  # special token
            continue
        index = int(np.argmax(probs[position]))
        label = id2label.get(index, "O")
        confidence = float(probs[position][index])
        if label == "O":
            flush()
            continue
        prefix, tag = label.split("-", 1)
        if prefix == "B" or tag != current_tag:
            flush()
            current_tag = tag
        current_tokens.append((start, end))
        current_conf = min(current_conf, confidence)
    flush()

    if not entities:
        return ""
    if any(conf < 0.60 for _, _, conf, _, _ in entities):
        return ""  # shaky span: let the cloud handle the whole task

    # Merge spans separated only by whitespace ("Allied" + "Powers"); the
    # model sometimes restarts a span mid-name. First tag wins. Use the real
    # offsets, not passage.find(): repeated entity text would otherwise merge
    # against the wrong occurrence.
    merged = [entities[0]]
    for text, tag, conf, start, end in entities[1:]:
        prev_text, prev_tag, prev_conf, prev_start, prev_end = merged[-1]
        if start >= prev_end and passage[prev_end:start].strip() == "":
            merged[-1] = (
                f"{prev_text} {text}".strip(),
                prev_tag,
                min(prev_conf, conf),
                prev_start,
                end,
            )
        else:
            merged.append((text, tag, conf, start, end))
    entities = merged

    lines, seen = [], set()
    for text, tag, _, _, _ in entities:
        type_name = _TYPE_NAMES.get(tag)
        if type_name is None:  # MISC
            type_name = "Event" if _DOC_EVENT_RE.search(text) else "Other"
        key = text.lower()
        if key not in seen:
            seen.add(key)
            lines.append(f"{text}: {type_name}")

    covered = {line.split(":")[0].strip().lower() for line in lines}
    for match in _DATE_RE.finditer(passage):
        date_text = match.group(0).strip()
        # "may" as a modal verb is not a date; a bare month is trusted only
        # when capitalized and not immediately followed by a lowercase word
        # ("May be wrong" is not a date).
        if date_text.lower() in _MONTH_NAMES:
            tail = passage[match.end():].lstrip()
            if date_text[:1].islower() or tail[:1].islower():
                continue
        inside = any(date_text.lower() in c or c in date_text.lower() for c in covered)
        if not inside:
            covered.add(date_text.lower())
            lines.append(f"{date_text}: Date")

    return "\n".join(lines)
