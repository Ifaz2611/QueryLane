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
from collections import defaultdict
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # cloud-only image: specialists stay disabled
    np = None

NER_DIR = os.environ.get("NER_MODEL_DIR", "/app/models/ner")
SST2_DIR = os.environ.get("SST2_MODEL_DIR", "/app/models/sst2")

# NER is enabled by default now that the post-processor reconciles the model's
# four-class schema with the task's six types and refuses incomplete answers.
# Set NER_SPECIALIST=off to disable it explicitly.
_NER_OFF_VALUES = {"off", "0", "false", "no", "disabled"}

# Precision gates. The model only wins when every token in a span is confident
# and the answer covers every likely entity in the passage; anything else
# escalates to cloud rather than risking an exact-match miss.
NER_MIN_CONFIDENCE = float(os.environ.get("NER_MIN_CONFIDENCE", "0.85"))
NER_MIN_MARGIN = float(os.environ.get("NER_MIN_MARGIN", "0.05"))
NER_COMPLETENESS_GUARD = os.environ.get(
    "NER_COMPLETENESS_GUARD", "on"
).strip().lower() not in _NER_OFF_VALUES

_lock = threading.Lock()
_sessions = {}


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def get_optimal_intra_op_threads() -> int:
    """Return tuned intra-op thread count for ONNX Runtime.

    Respects ONNX_INTRA_OP_THREADS if explicitly configured.
    Otherwise adapts to the host: on 1-2 vCPU containers, uses all available CPUs.
    On larger machines (4+ CPUs), bounds at 4 to prevent CPU contention with the
    worker thread pool and reduce context-switching overhead.
    """
    env_threads = os.environ.get("ONNX_INTRA_OP_THREADS")
    if env_threads:
        try:
            return max(1, int(env_threads))
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 2
    return min(4, max(1, cpu_count))


def _load(kind: str):
    """Lazy-load (session, tokenizer, id2label) for 'ner' or 'sst2'."""
    with _lock:
        if kind in _sessions:
            return _sessions[kind]
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = NER_DIR if kind == "ner" else SST2_DIR
        options = ort.SessionOptions()
        options.intra_op_num_threads = get_optimal_intra_op_threads()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Serialized graph cache across processes:
        opt_cache_dir = os.environ.get("ONNX_CACHE_DIR", directory)
        opt_model_name = f"model.{kind}.optimized.onnx"
        opt_model_path = os.path.join(opt_cache_dir, opt_model_name)
        orig_model_path = os.path.join(directory, "model.onnx")

        # Load optimized model if exists, otherwise load original and optimize
        model_to_load = opt_model_path if os.path.exists(opt_model_path) else orig_model_path
        if model_to_load == orig_model_path and os.environ.get("ONNX_DISABLE_CACHE") != "1":
            try:
                if os.access(opt_cache_dir, os.W_OK):
                    options.optimized_model_filepath = opt_model_path
            except Exception:
                pass

        session = ort.InferenceSession(
            model_to_load,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        _sessions[kind] = (session, tokenizer, id2label)
        return _sessions[kind]


def preload(kind: str = "all") -> None:
    """Pre-warm ONNX sessions and tokenizers in the background or at startup."""
    targets = ["sst2", "ner"] if kind == "all" else [kind]
    for k in targets:
        if available(k):
            try:
                _load(k)
            except Exception:
                pass


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


def ner_enabled() -> bool:
    """Whether the NER specialist is enabled and has its model available."""
    flag = os.environ.get("NER_SPECIALIST", "on").strip().lower()
    return flag not in _NER_OFF_VALUES and available("ner")


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

_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+"
    r"|[,;]\s*(?:but|though|although|however|yet|whereas)\s+"
    r"|\s+\b(?:but|though|although|however|yet|whereas)\b\s+",
    re.IGNORECASE,
)
# SST2 has no neutral class and is confidently wrong on factual statements
# and hedged language. Verdicts require explicit cue words agreeing with the
# classifier, and hedges escalate outright. Measured on 597 ground-truth
# reviews: these gates trade coverage for precision.
# Every cue starts at a word boundary: without it, "uncomfortable" matches
# "comfortable", "imperfect" matches "perfect", and "glove" matches "love".
# "crush" is deliberately narrowed to "crushed": "crushes ice" is praise.
_POS_CUE_RE = re.compile(
    r"\blove[ds]?\b|\bgreat\b|\bexcellent\b|\bamazing\b|\bfantastic\b|"
    r"\bwonderful\b|\bperfect\b|\bawesome\b|\benjoy\w*\b|\bimpress\w*\b|"
    r"\brecommend\w*\b|\bsatisf\w*\b|\bdelight\w*\b|\bbreeze\b|\bsuperb\b|"
    r"\bcomfortable\b|\bspotless\b|\bfriendly\b|\bcrisp\b|\bflawless\b|"
    r"\bdream\b|\bhighly\b|\bgood\b|\bhappy\b|\bworth\b|\bpleased\b|"
    r"\bglad\b|\bthrilled\b|\boutstanding\b|\bincredible\b|\bsmooth\b|"
    r"\beasy\b|\breliable\b|\bsturdy\b|"
    r"\b(?:fast|quick)\s+(?:delivery|shipping|service|setup|response|"
    r"turnaround|support)\b|\bbeautiful\b|"
    r"\bgorgeous\b|\bpremium\b|\bfavourite\b|\bfavorite\b|\bexceed\w*\b|"
    r"\bworks? well\b",
    re.IGNORECASE,
)
_NEG_CUE_RE = re.compile(
    r"\bterrible\b|\bawful\b|\bhorrible\b|\bworst\b|\bhate[ds]?\b|"
    r"\bdisappoint\w*\b|\bpoor\b|\bbroke\w*\b|\bbroken\b|\bcrush(?:ed)?\b|"
    r"\blate\b|\bdelay\w*\b|\bslow\w*\b|\bfail\w*\b|\bcrash\w*\b|"
    r"\bscratch\w*\b|"
    r"\bdefect\w*\b|\brefund\w*\b|\breturn\w*\b|\bwaste\w*\b|\buseless\b|"
    r"\bannoy\w*\b|\bfrustrat\w*\b|\bleak\w*\b|\bcheap\w*\b|\bdirty\b|"
    r"\brude\b|\bmuddy\b|\bunbearable\b|\bbad\b|\bcomplaint\w*\b|"
    r"\bproblem\w*\b|\bissue\w*\b|\bdamag(?:e|ed|es)\b|\bfaulty\b|"
    r"\bmissing\b|\bwrong\b|\bincorrect\b|\bnoisy\b|\bloud\b|"
    r"\buncomfortable\b|\bflimsy\b|\bweak\b|\bleak\w*\b|"
    r"\bdownside\b|\bcons\b|\bpricey\b|\bexpensive\b|"
    r"\bdead on arrival\b|\bdoa\b|\brip[- ]?off\b|\boverpriced\b|"
    r"\bregret\w*\b|\bcancel\w*\b|\blost\b|\bstolen\b|\bmold\w*\b|"
    r"\bstain\w*\b|\btear\w*\b|\btorn\b|\bburn\w*\b|\boverheat\w*\b|"
    r"\bfreez\w*\b|\bdisconnect\w*\b|\bdropout\w*\b|\bmuffled\b|"
    r"\bstatic\b|\bglitch\w*\b|\blag\w*\b|\bbug\w*\b|\bdies?\b|\bdied\b|"
    r"\bcould be better\b|\bstopped working\b|\bdoesn.t work\b|"
    r"\bnever (?:arrived|worked|again)\b",
    re.IGNORECASE,
)
# Negation is handled by `_is_negated`, not by listing negated idioms here:
# "not bad", "never disappoints", and "no complaints" become positive
# evidence, while "not great" and "not worth it" become negative evidence.
_HEDGE_RE = re.compile(
    r"\bnothing (?:special|more|less)\b|\baverage\b|\bdecent\b|\bokay\b|"
    r"\bok\b|\bfine\b|\bdoes (?:what it says|the job)\b|\bas expected\b|"
    r"\bit works\b|\badequate\b|\bmediocre\b|\bso-so\b|"
    r"\bmiddle of the road\b|\bnot sure\b|\bmixed feelings\b|"
    r"\bon the fence\b|\bhard to say\b|\bneither\b|\bit depends\b|"
    r"\bno strong feelings\b|\bwish\b|\bnot for me\b|\bdownside\b",
    re.IGNORECASE,
)
# A contrast conjunction often signals a two-sided review. It only forces
# escalation when evidence actually appears on both sides (an unconfirmed mix)
# or on neither side (nothing to stand on). One-sided evidence is safe: e.g.
# `The box says "premium quality" but it feels cheap.` is a complaint whose
# praise lives in a non-endorsed marketing quote.
_CONTRAST_RE = re.compile(
    r"\b(?:but|though|although|however|yet|whereas|except|aside from|"
    r"other than|even though|that said|having said that)\b",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(
    r"^.*?(?:sentiment of this review|sentiment of the review|"
    r"sentiment of the following review|the following review|this review|"
    r"the review says|review text|customer review)\s*[:.\-–—\n]\s*",
    re.IGNORECASE | re.DOTALL,
)

# Quoted material is not the reviewer's own praise by default: a marketing
# claim in quotes must not turn a complaint into a positive verdict. Quotes do
# count when they are the review content or when the reviewer endorses them.
_QUOTE_RE = re.compile(
    r'"[^"\n]{1,240}"|“[^”\n]{1,240}”|‘[^’\n]{1,240}’|'
    r"(?<![\w])'[^'\n]{2,240}'(?![\w])"
)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:review|feedback|comment|text|quote|customer|user|reviewer)\b"
    r".{0,40}\b(?:says?|said|wrote|reads?|states?|called (?:it|the \w+)|"
    r"describes? (?:it|the \w+) as)\s*:?\s*$",
    re.IGNORECASE,
)
_ENDORSE_RE = re.compile(
    r"\b(?:i agree|agree(?:d)?|couldn'?t agree more|indeed|exactly|true|"
    r"accurate|right|lives? up to|as (?:promised|advertised|described)|"
    r"can confirm|same here|spot on|not wrong|well said)\b",
    re.IGNORECASE,
)
# A quote introduced by packaging or marketing is third-party copy, not the
# reviewer's own words. It only becomes evidence when explicitly endorsed.
_THIRD_PARTY_RE = re.compile(
    r"\b(?:box|packaging|package|label|listing|advert\w*|marketing|"
    r"brochure|manual|website|site|description|seller|manufacturer|company|"
    r"brand)\b.{0,25}\b(?:says?|said|claims?|claimed|promises?|promised|"
    r"advertis\w*|markets?|marketed|calls?|describes?)\b",
    re.IGNORECASE,
)
_NEGATION_WORD_RE = re.compile(
    r"^(?:not|no|never|none|nothing|nobody|neither|nor|nowhere|hardly|"
    r"barely|scarcely|without|zero|isn'?t|aren'?t|wasn'?t|weren'?t|"
    r"doesn'?t|don'?t|didn'?t|won'?t|wouldn'?t|can'?t|cannot|couldn'?t|"
    r"shouldn'?t|hasn'?t|haven'?t|hadn'?t|ain'?t|isnt|arent|wasnt|werent|"
    r"doesnt|dont|didnt|wont|wouldnt|cant|couldnt|shouldnt|hasnt|havent|hadnt)$",
    re.IGNORECASE,
)


def extract_review(prompt: str) -> str:
    """Drop instruction wrappers, keep the review text.

    Strip repeatedly: prompts sometimes nest a wrapper ("Classify the sentiment
    of this review: The review says: ...").
    """
    text = prompt.strip()
    for _ in range(3):
        stripped = _PREFIX_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text or prompt


def _classify_clause(text: str):
    _, _, logits, id2label = _run("sst2", text)
    probs = _softmax(logits.reshape(-1))
    index = int(np.argmax(probs))
    return id2label.get(index, "POSITIVE"), float(probs[index])


@dataclass(frozen=True)
class _Evidence:
    """Polarity terms found in a clause after negation and quote handling."""

    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()
    negated_positive: tuple[str, ...] = ()  # "not great" -> negative
    negated_negative: tuple[str, ...] = ()  # "not bad" -> positive
    quoted: tuple[str, ...] = ()

    @property
    def has_positive(self) -> bool:
        return bool(self.positive)

    @property
    def has_negative(self) -> bool:
        return bool(self.negative)


def _is_negated(text: str, start: int) -> bool:
    """Whether a cue at `start` sits inside a short negation scope.

    The window deliberately stops at clause punctuation and only looks back a
    few words: "no doubt this is great" must not negate "great", while
    "not very great" and "never had any problems" must.
    """
    prefix = text[max(0, start - 48):start]
    clause = re.split(r"[.!?;:,()\[\]]", prefix)[-1]
    if re.search(r"\bnot only\s+(?:\w+\s+){0,2}$", clause, re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:no doubt|no question|without a doubt|not surprisingly)\b",
        clause,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:far from|anything but|free of|free from)\s+(?:\w+\s+){0,2}$",
        clause,
        re.IGNORECASE,
    ):
        return True
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", clause)
    return any(_NEGATION_WORD_RE.match(word) for word in words[-3:])


def _mask_quotes(text: str) -> str:
    return _QUOTE_RE.sub(lambda match: " " * len(match.group(0)), text)


def _quote_counts_as_evidence(text: str, match, context: str = "") -> bool:
    """Quoted words are evidence only when they are the review content or the
    reviewer explicitly endorses them. `context` is the full review so a short
    clause cannot make a marketing quote look like the reviewer's own words."""
    context = context or text
    before = text[max(0, match.start() - 90):match.start()]
    after = text[match.end():match.end() + 60]
    endorsed = bool(_ENDORSE_RE.search(before) or _ENDORSE_RE.search(after))
    if _THIRD_PARTY_RE.search(before):
        return endorsed
    quoted_chars = sum(len(item.group(0)) for item in _QUOTE_RE.finditer(context))
    if quoted_chars >= 0.45 * max(len(context.strip()), 1):
        return True
    return bool(_ATTRIBUTION_RE.search(before) or endorsed)


def _scan_terms(text: str):
    """Return (positive, negative, negated_positive, negated_negative)."""
    positive, negative, negated_pos, negated_neg = [], [], [], []
    for regex, polarity in ((_POS_CUE_RE, "pos"), (_NEG_CUE_RE, "neg")):
        for match in regex.finditer(text):
            term = match.group(0).lower()
            negated = _is_negated(text, match.start())
            final = polarity
            if negated:
                final = "neg" if polarity == "pos" else "pos"
                if polarity == "pos":
                    negated_pos.append(term)
                else:
                    negated_neg.append(term)
            (positive if final == "pos" else negative).append(term)
    return positive, negative, negated_pos, negated_neg


def _evidence(text: str, quote_context: str = "") -> _Evidence:
    """Extract polarity evidence with negation scopes and quote awareness."""
    if not text.strip():
        return _Evidence()
    positive, negative, negated_pos, negated_neg = _scan_terms(_mask_quotes(text))
    quoted = []
    for match in _QUOTE_RE.finditer(text):
        if not _quote_counts_as_evidence(text, match, quote_context):
            continue
        body = match.group(0)[1:-1]
        q_pos, q_neg, q_neg_pos, q_neg_neg = _scan_terms(body)
        positive.extend(q_pos)
        negative.extend(q_neg)
        negated_pos.extend(q_neg_pos)
        negated_neg.extend(q_neg_neg)
        quoted.extend(term for term in q_pos + q_neg if term)
    return _Evidence(
        tuple(positive),
        tuple(negative),
        tuple(negated_pos),
        tuple(negated_neg),
        tuple(quoted),
    )


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
        votes.append((label, confidence, clause, _evidence(clause, review)))

    # A side only counts when the classifier AND explicit evidence agree on
    # the same clause; classifier-only signals escalate.
    strong_pos = [
        v for v in votes
        if v[0] == "POSITIVE" and v[1] >= 0.9 and v[3].has_positive
        and not v[3].has_negative
    ]
    strong_neg = [
        v for v in votes
        if v[0] == "NEGATIVE" and v[1] >= 0.9 and v[3].has_negative
        and not v[3].has_positive
    ]
    pos_votes = [v for v in votes if v[3].has_positive and not v[3].has_negative]
    neg_votes = [v for v in votes if v[3].has_negative and not v[3].has_positive]

    whole_label, whole_conf = _classify_clause(review)
    whole_ev = _evidence(review, review)

    if pos_votes and neg_votes and (strong_pos or strong_neg):
        return (
            "Mixed, the review contains both praise "
            f"({_quotes(strong_pos or pos_votes)}) and criticism "
            f"({_quotes(strong_neg or neg_votes)})."
        )
    if _CONTRAST_RE.search(review):
        two_sided = whole_ev.has_positive and whole_ev.has_negative
        no_evidence = not whole_ev.has_positive and not whole_ev.has_negative
        if two_sided or no_evidence:
            return ""  # unconfirmed mix or unsupported contrast: escalate

    negation_involved = bool(
        whole_ev.negated_positive or whole_ev.negated_negative
    )
    if whole_ev.has_positive and not whole_ev.has_negative:
        threshold = 0.90 if negation_involved else 0.97
        if whole_ev.quoted and not negation_involved:
            threshold = min(threshold, 0.92)
        if whole_label == "POSITIVE" and whole_conf >= threshold:
            return (
                "Positive, the review expresses satisfaction "
                f"({_quotes(strong_pos or pos_votes) or _terms(whole_ev.positive)}) "
                "with no complaints."
            )
    if whole_ev.has_negative and not whole_ev.has_positive:
        threshold = 0.90 if negation_involved else 0.97
        if whole_ev.quoted and not negation_involved:
            threshold = min(threshold, 0.92)
        if whole_label == "NEGATIVE" and whole_conf >= threshold:
            return (
                "Negative, the review reports problems "
                f"({_quotes(strong_neg or neg_votes) or _terms(whole_ev.negative)}) "
                "with no praise."
            )
    return ""  # ambiguous: escalate to cloud


def _terms(terms) -> str:
    return "; ".join(_quote(term) for term in terms[:3])


def _quotes(votes, limit: int = 90) -> str:
    return "; ".join(_quote(v[2], limit) for v in votes[:3])


def _quote(text: str, limit: int = 90) -> str:
    text = text.strip().rstrip(".")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    if text[:1] in "\"“‘" and text[-1:] in "\"”’":
        return text
    return f'"{text}"'


# ---------------------------------------------------------------------- ner

_NER_PREFIX_RE = re.compile(
    r"^.*?(?:entities and their types from|entities and their types in this sentence|"
    r"entities and their types in|entities and their types|entities from|"
    r"entities in|types from|entit(?:y|ies)[^:]*)[:.]\s*",
    re.IGNORECASE | re.DOTALL,
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_MONTH_NAMES = frozenset(_MONTHS.split("|"))
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?(?:\s*(?:-|–|to)\s*\d{{1,2}}"
    rf"(?:st|nd|rd|th)?)?\s+(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\b\s+\d{{1,2}}(?!\d)(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\b\s+\d{{4}}\b"
    rf"|(?:last|next|this|early|late)\s+(?:{_MONTHS})\b"
    rf"|(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    rf"|\bQ[1-4]\s+\d{{4}}\b"
    rf"|\b(?:19|20)\d{{2}}s?\b"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})",
    re.IGNORECASE,
)

# The task schema has six types; the ONNX model only knows PER/ORG/LOC/MISC.
# These cue lists reconcile the two and never emit the old "Other" fallback,
# which cannot match any gold answer.
_EVENT_CUE_RE = re.compile(
    r"\b(?:summit|conference|forum|expo|exposition|symposium|congress|assembly|"
    r"convention|festival|gala|awards?|cup|league|championship|prix|"
    r"invitational|series|games|olympics?|olympiad|tour|rally|marathon|"
    r"cyclothon|derby|open|showcase|fair|biennale|premiere|dialogue|meet|"
    r"meeting|ceremony|initiative|project|programme|program|expedition|"
    r"mission|survey|review|negotiations?|deal|merger|breach|treaty|"
    r"accords?|pact|agreement|protocol|charter|declaration|compact|"
    r"concordat|edict|centennial|anniversary|black friday|launch event)\b",
    re.IGNORECASE,
)
_PRODUCT_CUE_RE = re.compile(
    r"\b(?:ev|ai|app|api|software|platform|system|device|tablet|phone|"
    r"laptop|headset|earbuds|headphones|camera|sensor|battery|chip|"
    r"processor|vaccine|drug|kits?|scanner|probe|rover|lander|satellite|"
    r"rocket|shuttle|engine|turbine|generator|drill|pen|watch|band|ring|"
    r"lamp|projector|router|drone|robot|exoskeleton|shoes?|boots?|jacket|"
    r"suit|necklace|game|novel|book|film|album|song|collection|edition|"
    r"model|version|pro|max|plus|ultra|mini|lite|vr|ar|xbox|playstation)\b"
    r"|[A-Za-z]+-\d+|\b[A-Za-z]+\d+\b",
    re.IGNORECASE,
)
_LOCATION_CUE_RE = re.compile(
    r"\b(?:republic|kingdom|federation|union|confederation|duchy|sultanate|"
    r"state|province|county|region|city|town|village|island|isle|coast|bay|"
    r"gulf|sea|ocean|lake|river|strait|valley|mountain|crater|park|reserve|"
    r"forest|desert|peninsula|cape|harbou?r|port|airport|station|observatory|"
    r"museum|gallery|theat(?:er|re)|arena|stadium|complex|cent(?:er|re)|hall|"
    r"palace|castle|temple|tower|building|bridge|street|road|square|garden|"
    r"gateway|base|colony|settlement|capital|district|territory)\b",
    re.IGNORECASE,
)
_ORG_CUE_RE = re.compile(
    r"\b(?:inc|corp|corporation|ltd|llc|plc|gmbh|holdings|group|labs|"
    r"laboratories|bank|university|institute|foundation|council|committee|"
    r"agency|association|club|fc|united|motors|energy|pharmaceuticals|"
    r"telecom|solutions|industries|logistics|capital|partners|systems|"
    r"technologies|networks|research|coalition|league|academy|society|"
    r"organi[sz]ation|ministry|department|bureau|office|commission|"
    r"authority|services?)\b",
    re.IGNORECASE,
)
_PRODUCT_CONTEXT_RE = re.compile(
    r"\b(?:launch(?:ed|es|ing)?|unveil(?:ed|s|ing)?|releas(?:ed|es|ing)|"
    r"introduc(?:ed|es|ing)|debut(?:ed|s|ing)?|retails?|priced|ships?|"
    r"sold|sells|available|product|device)\b",
    re.IGNORECASE,
)
_EVENT_CONTEXT_RE = re.compile(
    r"\b(?:signed|host(?:ed|s|ing)?|held|attend(?:ed|s|ing)?|"
    r"conven(?:ed|es|ing)?|sponsor(?:ed|s)?|won|competed|"
    r"participat(?:ed|es|ing)?)\b",
    re.IGNORECASE,
)
_ORG_CONTEXT_RE = re.compile(
    r"\b(?:company|firm|startup|bank|institute|university|agency|group|"
    r"organi[sz]ation|club|team|ceo|spokesperson|headquartered|based)\b",
    re.IGNORECASE,
)
_PERSON_CONTEXT_RE = re.compile(
    r"\b(?:said|says|joined|met|led|spoke|wrote|announced|appointed|hired)\b",
    re.IGNORECASE,
)
_TYPE_NAMES = {"PER": "Person", "ORG": "Organization", "LOC": "Location"}
_TYPE_ORDER = ("Person", "Organization", "Location", "Date", "Event", "Product")
_TYPE_ALIASES = {
    "person": "Person",
    "per": "Person",
    "organization": "Organization",
    "organisation": "Organization",
    "org": "Organization",
    "location": "Location",
    "loc": "Location",
    "date": "Date",
    "event": "Event",
    "product": "Product",
}
_TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:dr|mr|mrs|ms|prof|ceo|cto|cfo|coo|founder|cofounder|"
    r"president|chancellor|secretary|minister|director|professor|judge|"
    r"senator|mayor|captain|coach|chairman|chairwoman|spokesperson|"
    r"spokesman|spokeswoman|governor|general|researcher|scientist|"
    r"economist)\.?\s+)+",
    re.IGNORECASE,
)
_SENTENCE_STARTERS = frozenset(
    "however meanwhile yesterday today tomorrow later earlier finally first "
    "second third next then according despite although though while when if as "
    "since because before after following during already still just also only "
    "many most some several both each every another other such new recent last "
    "this that these those it he she they we i you his her their our my your its".split()
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def extract_passage(prompt: str) -> str:
    stripped = _NER_PREFIX_RE.sub("", prompt, count=1).strip()
    return stripped if stripped else prompt


def _iter_dates(passage: str):
    for match in _DATE_RE.finditer(passage):
        date_text = match.group(0).strip()
        # "may" as a modal verb is not a date; a bare month is trusted only
        # when capitalized and not immediately followed by a lowercase word
        # ("May be wrong" is not a date).
        if date_text.lower() in _MONTH_NAMES:
            tail = passage[match.end():].lstrip()
            if date_text[:1].islower() or tail[:1].islower():
                continue
        yield match


def _clean_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n.,;:!?()[]{}\"'“”‘’")
    text = re.sub(r"['’]s$", "", text)
    text = _TITLE_PREFIX_RE.sub("", text).strip()
    return text


def _candidate_spans(passage: str):
    """Capitalized spans in the passage, minus sentence starters and dates."""
    from agent import local

    candidates = []
    for span in local._capitalized_spans(passage):
        text = _clean_entity_text(span)
        normalized = _norm(text)
        if not normalized or len(normalized) < 2:
            continue
        if normalized in _SENTENCE_STARTERS or normalized in _MONTH_NAMES:
            continue
        if _DATE_RE.fullmatch(text.strip()):
            continue
        candidates.append(text)
    return candidates


def _context(passage: str, start: int, end: int, radius: int = 70) -> str:
    return passage[max(0, start - radius):min(len(passage), end + radius)]


def _text_type_scores(text: str) -> dict:
    """Lexical type cues on the entity text itself."""
    from agent import local

    scores = defaultdict(float)
    if re.fullmatch(rf"(?:the )?(?:{local._COUNTRIES})", text, re.IGNORECASE):
        scores["Location"] += 4.0
    if _EVENT_CUE_RE.search(text):
        scores["Event"] += 3.0
    if _PRODUCT_CUE_RE.search(text):
        scores["Product"] += 3.0
    if _LOCATION_CUE_RE.search(text):
        scores["Location"] += 3.0
    if _ORG_CUE_RE.search(text):
        scores["Organization"] += 3.0
    return scores


def _context_type_scores(passage: str, start: int, end: int) -> dict:
    """Weaker cues from the sentence around a span."""
    scores = defaultdict(float)
    window = _context(passage, start, end)
    if _PRODUCT_CONTEXT_RE.search(window):
        scores["Product"] += 3.0
    if _EVENT_CONTEXT_RE.search(window):
        scores["Event"] += 2.0
    if _ORG_CONTEXT_RE.search(window):
        scores["Organization"] += 1.0
    if _PERSON_CONTEXT_RE.search(window):
        scores["Person"] += 1.0
    return scores


def _infer_entity_type(
    text: str, passage: str, start: int, end: int, model_tag: str
):
    """Map a model span to the six-type task schema.

    The model's PER/ORG/LOC label is trusted for non-MISC spans; only a country
    name (or an explicit date) is strong enough to override it. Context words
    are deliberately ignored there: "unveiled" or "retail" near a person or
    city must not turn that span into a Product.

    MISC spans carry no type, so cues on the span win first. Nearby launch or
    event context is used only when the span itself has no cue. Ambiguous cases
    return ``None`` and the whole answer escalates.
    """
    if _DATE_RE.fullmatch(text.strip()):
        return "Date"
    model_type = _TYPE_NAMES.get(model_tag)
    text_scores = _text_type_scores(text)

    if model_tag == "MISC":
        scores = text_scores or _context_type_scores(passage, start, end)
        if not scores:
            return None
        best_score = max(scores.values())
        best = [name for name, score in scores.items() if score == best_score]
        if best_score < 2.0 or len(best) != 1:
            return None
        return best[0]

    if model_type is None:
        return None
    if text_scores.get("Location", 0.0) >= 4.0:
        return "Location"
    return model_type


def _decode_entities(passage: str, encoding, ids, logits, id2label):
    """Decode model logits into typed entities, or ``None`` to escalate."""
    probs = _softmax(logits, axis=-1)
    offsets = encoding.offsets[: len(ids)]
    entities = []  # [tag, start, end, min_confidence]
    current_tokens, current_tag, current_conf, current_margin = [], None, 1.0, 1.0

    def flush():
        nonlocal current_tokens, current_tag, current_conf, current_margin
        if current_tokens and current_tag:
            start = current_tokens[0][0]
            end = current_tokens[-1][1]
            if end > start:
                entities.append([current_tag, start, end, current_conf, current_margin])
        current_tokens, current_tag, current_conf, current_margin = [], None, 1.0, 1.0

    for position, (start, end) in enumerate(offsets):
        if end == 0:  # special token
            continue
        row = probs[position]
        index = int(np.argmax(row))
        ordered = np.sort(row)
        confidence = float(ordered[-1])
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0
        label = id2label.get(index, "O")
        if label == "O" or "-" not in label:
            flush()
            continue
        prefix, tag = label.split("-", 1)
        if prefix == "B" or tag != current_tag:
            flush()
            current_tag = tag
        current_tokens.append((start, end))
        current_conf = min(current_conf, confidence)
        current_margin = min(current_margin, margin)
    flush()

    # A single shaky token can change an exact-match answer; escalate instead.
    if any(
        conf < NER_MIN_CONFIDENCE or margin < NER_MIN_MARGIN
        for _, _, _, conf, margin in entities
    ):
        return None

    # Merge spans the model restarted mid-name ("Allied" + "Powers"), but only
    # when the tag agrees and the gap is whitespace or a name connector.
    connectors = {"of", "the", "and", "de", "da", "van", "von", "&", "for"}
    merged = [entities[0]] if entities else []
    for tag, start, end, conf, margin in entities[1:]:
        prev_tag, prev_start, prev_end, prev_conf, prev_margin = merged[-1]
        gap = passage[prev_end:start]
        gap_clean = gap.strip()
        if (
            tag == prev_tag
            and start >= prev_end
            and (not gap_clean or gap_clean.lower() in connectors)
        ):
            merged[-1] = [
                prev_tag,
                prev_start,
                end,
                min(prev_conf, conf),
                min(prev_margin, margin),
            ]
        else:
            merged.append([tag, start, end, conf, margin])

    date_matches = list(_iter_dates(passage))
    typed = []  # (text, type, start, end)
    for tag, start, end, _, _ in merged:
        overlapping = [
            match for match in date_matches
            if match.start() < end and match.end() > start
        ]
        if overlapping:
            match = max(overlapping, key=lambda item: len(item.group(0)))
            text = _clean_entity_text(match.group(0))
            if text:
                typed.append((text, "Date", match.start(), match.end()))
            continue
        text = _clean_entity_text(passage[start:end])
        if not text:
            continue
        type_name = _infer_entity_type(text, passage, start, end, tag)
        if type_name is None:
            return None  # MISC without a reliable cue: escalate
        typed.append((text, type_name, start, end))

    # Add dates the model missed/did not tag, but never duplicate a date that
    # is already inside an entity such as "CES 2025".
    for match in date_matches:
        if any(match.start() >= start and match.end() <= end for _, _, start, end in typed):
            continue
        text = _clean_entity_text(match.group(0))
        if text:
            typed.append((text, "Date", match.start(), match.end()))

    # Prefer longer spans when one entity is nested inside another, then drop
    # duplicate text.
    typed.sort(key=lambda item: (item[2], -(item[3] - item[2])))
    kept = []
    for text, type_name, start, end in typed:
        key = _norm(text)
        if not key:
            continue
        if any(
            key in _norm(other_text) and start >= other_start and end <= other_end
            for other_text, _, other_start, other_end in typed
            if (other_start, other_end) != (start, end)
        ):
            continue
        if any(key == _norm(other_text) for other_text, _, _, _ in kept):
            continue
        kept.append((text, type_name, start, end))

    if not kept:
        return []
    if NER_COMPLETENESS_GUARD:
        answer_norm = " " + " ".join(_norm(text) for text, _, _, _ in kept) + " "
        for candidate in _candidate_spans(passage):
            if _norm(candidate) not in answer_norm:
                return None
        for match in date_matches:
            if _norm(match.group(0)) not in answer_norm:
                return None
    return kept


def ner_answer(prompt: str) -> str:
    """'Entity: Type' lines, or '' to escalate."""
    passage = extract_passage(prompt)
    encoding, ids, logits, id2label = _run("ner", passage)
    entities = _decode_entities(passage, encoding, ids, logits, id2label)
    if not entities:
        return ""
    return "\n".join(f"{text}: {type_name}" for text, type_name, _, _ in entities)
