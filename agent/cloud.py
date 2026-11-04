"""Fireworks client.

Every call goes through FIREWORKS_BASE_URL (required for token accounting) and
logs usage.total_tokens. One answer per task: no content-based retries, only
transient-error retries and model fallback, because every retry costs tokens.
Transient retries are spaced with exponential backoff plus jitter, and the whole
answer attempt runs under a hard per-task time budget so one slow call can never
consume the global deadline in agent/main.py.
"""

import os
import random
import re
import threading
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from agent import obs, pal, prompts

# The system prompt already instructs these categories fully, so the user
# message can be just the payload. Strips only exact-shape instruction
# wrappers; anything unusual passes through untouched.
_STRIP_RES = {
    "sentiment": re.compile(
        r"^\s*classify the sentiment of (?:this|the following) review\s*[:.]\s*",
        re.IGNORECASE,
    ),
    "ner": re.compile(
        r"^\s*(?:extract|identify|list) (?:all )?(?:the )?named entities "
        r"(?:and their types )?(?:from|in)(?: this sentence)?\s*[:.]\s*",
        re.IGNORECASE,
    ),
}


def strip_instruction(category: str, prompt: str) -> str:
    pattern = _STRIP_RES.get(category)
    if not pattern:
        return prompt
    stripped = pattern.sub("", prompt, count=1)
    return stripped if stripped.strip() else prompt

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRANSIENT_RETRIES = 2


def _strip_thinking(text: str) -> str:
    """Remove reasoning blocks some models embed in content."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def usage_totals(usage) -> dict:
    """Normalize a provider ``usage`` object (or dict) into token counters.

    Proxies omit fields, return ``None``, or send strings; every counter is
    coerced to a non-negative int so accounting can never turn a good answer
    into a failed task.
    """
    totals = {"prompt": 0, "completion": 0, "total": 0}

    def read(name):
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return max(0, int(value))

    totals["prompt"] = read("prompt_tokens")
    totals["completion"] = read("completion_tokens")
    totals["total"] = read("total_tokens")
    if not totals["total"]:
        # Some compatible providers only report the two halves.
        totals["total"] = totals["prompt"] + totals["completion"]
    return totals


def response_text(response) -> str:
    """Return the first choice's stripped content, or "" when there is none.

    Handles ``choices == []``, ``choices is None``, a missing ``message``, and
    a non-string ``content`` without raising: a malformed response is treated
    like any other empty answer and escalates.
    """
    choices = getattr(response, "choices", None) or []
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return ""
    return _strip_thinking(content)


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` when unset or invalid.

    A typo in a tuning variable must never crash a run mid-retry.
    """
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _backoff_delay(attempt: int, base: float, cap: float, jitter: float = 0.5) -> float:
    """Seconds to wait before retry ``attempt`` (1-based): base*2^n plus jitter.

    Equal-jitter backoff: half the exponential delay is fixed so retries keep
    spreading out, the other half is random so a batch of workers that all
    failed at once does not retry in lockstep.
    """
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    return max(0.0, delay * (1.0 - jitter) + random.uniform(0.0, delay * jitter))


class CloudClient:
    def __init__(self, api_key: str, base_url: str, allowed_models: list):
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0
        )
        self._allowed = allowed_models
        self._lock = threading.Lock()
        self._total_tokens = 0
        self._calls = 0
        self._category_tokens: dict = {}
        self._category_calls: dict = {}
        # Observability counters (exported by agent/obs.metrics on request).
        self._retry_count = 0
        self._empty_responses = 0
        self._budget_exceeded = 0
        # Hard per-task cloud budget: one slow call must not eat the global
        # deadline that agent/main.py is racing against.
        self.task_budget = max(1.0, float(os.environ.get("CLOUD_TASK_BUDGET", "75")))

    @classmethod
    def from_env(cls) -> "CloudClient":
        allowed = []
        for model in os.environ["ALLOWED_MODELS"].split(","):
            model = model.strip()
            if model and model not in allowed:
                allowed.append(model)
        if not allowed:
            raise ValueError("ALLOWED_MODELS must name at least one model")
        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
            allowed_models=allowed,
        )

    def total_tokens(self) -> int:
        return self._total_tokens

    def call_count(self) -> int:
        return self._calls

    def tokens_for_category(self, category: str) -> int:
        with self._lock:
            return self._category_tokens.get(category, 0)

    def calls_for_category(self, category: str) -> int:
        with self._lock:
            return self._category_calls.get(category, 0)

    def tokens_by_category(self) -> dict:
        with self._lock:
            return dict(self._category_tokens)

    def calls_by_category(self) -> dict:
        with self._lock:
            return dict(self._category_calls)

    def retry_count(self) -> int:
        """Number of transient retries issued (observability only)."""
        with self._lock:
            return self._retry_count

    def empty_response_count(self) -> int:
        """Calls that came back with no usable content."""
        with self._lock:
            return self._empty_responses

    def budget_exceeded_count(self) -> int:
        """Tasks that hit the per-task cloud budget before finishing."""
        with self._lock:
            return self._budget_exceeded

    def _models_for(self, category: str) -> list:
        """Order ALLOWED_MODELS by per-category preference tags; keep every
        allowed model as a fallback so a missing deployment never strands us."""
        tags = prompts.MODEL_PREFERENCE.get(category, prompts.DEFAULT_PREFERENCE)
        ordered = []
        for tag in tags:
            for model in self._allowed:
                if tag in model.lower() and model not in ordered:
                    ordered.append(model)
        for model in self._allowed:
            if model not in ordered:
                ordered.append(model)
        return ordered

    def answer(self, category: str, prompt: str) -> str:
        prompt = strip_instruction(category, prompt)
        tokens_before = self._total_tokens
        deadline = time.time() + self.task_budget
        try:
            if category == "math":
                # Cloud math path kept behaviorally identical to the twice-scored
                # v13: PAL first, step-by-step only on parse failure. The multipart
                # detector gates only the LOCAL path (agent/main.py), where routing
                # a two-quantity question away from single-EXPR is pure upside.
                text = self._answer_with_spec(
                    prompts.SPECS[category], category, prompt, deadline
                )
                parsed = pal.answer_from_expression(text) if text else None
                if parsed is not None:
                    return parsed
                if time.time() >= deadline:
                    # No budget left for the step-by-step fallback call.
                    self._note_budget_exceeded(category)
                    return ""
                return self._answer_with_spec(
                    prompts.MATH_FALLBACK, category, prompt, deadline
                )
            return self._answer_with_spec(
                prompts.SPECS[category], category, prompt, deadline
            )
        finally:
            used = max(0, self._total_tokens - tokens_before)
            with self._lock:
                self._category_tokens[category] = (
                    self._category_tokens.get(category, 0) + used
                )
                self._category_calls[category] = (
                    self._category_calls.get(category, 0) + 1
                )

    def _answer_with_spec(self, spec, category: str, prompt: str, deadline: float = None):
        """Run one spec across models/attempts inside the per-task budget."""
        if deadline is None:
            deadline = time.time() + self.task_budget
        last_error = None
        for model in self._models_for(category):
            effort = spec.effort
            # Second budget exists only for thinking models whose reasoning can
            # eat the whole first budget and leave content empty.
            for max_tokens in (spec.max_tokens, spec.max_tokens * 4):
                if time.time() >= deadline:
                    self._note_budget_exceeded(category)
                    return ""  # partial budget is spent; caller escalates
                text, error = self._call_with_retries(
                    model, spec.system, prompt, max_tokens, effort, deadline
                )
                if text:
                    return text
                if error is not None:
                    # A grading proxy that rejects reasoning_effort would 400
                    # every call; drop the field once and retry before giving
                    # up on the model.
                    if effort and getattr(error, "status_code", None) == 400:
                        text, error = self._call_with_retries(
                            model, spec.system, prompt, max_tokens, None, deadline
                        )
                        if text:
                            return text
                        effort = None
                    if error is not None:
                        last_error = error
                        break  # model-level problem: move to the next model
        if last_error is None:
            # Every model returned an empty (but valid) response: no model-level
            # error to report, so escalate quietly instead of raising.
            return ""
        raise RuntimeError(f"all models failed for {category}: {last_error}")

    def _call_with_retries(self, model, system, user, max_tokens, effort, deadline=None):
        """Returns (text, error). error is set only for model-level failures
        (bad id, undeployed, 4xx) that should skip to the next model.

        Connection errors, timeouts, 429s and 5xx responses are retried after an
        exponential backoff with jitter; every wait is clipped to the remaining
        per-task budget so a slow provider cannot blow the global deadline.
        """
        if deadline is None:
            deadline = time.time() + self.task_budget
        for attempt in range(1 + _TRANSIENT_RETRIES):
            try:
                return self._call(model, system, user, max_tokens, effort), None
            except RateLimitError as exc:
                self._sleep_before_retry(attempt, "rate_limit", model, deadline, exc)
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    self._sleep_before_retry(
                        attempt, f"http_{exc.status_code}", model, deadline, exc
                    )
                else:
                    return "", exc
            except (APIConnectionError, APITimeoutError) as exc:
                self._sleep_before_retry(attempt, "connection", model, deadline, exc)
            if time.time() >= deadline:
                break  # spending another attempt past the budget is not allowed
        return "", None

    def _sleep_before_retry(self, attempt, reason, model, deadline, exc) -> None:
        """Back off before the next attempt, or give up when the budget is gone."""
        if attempt >= _TRANSIENT_RETRIES:
            return  # final attempt: a wait cannot be followed by another try
        remaining = deadline - time.time()
        if remaining <= 0:
            return  # the caller's budget check stops the remaining retries
        delay = min(
            _backoff_delay(
                attempt + 1,
                _env_float("CLOUD_BACKOFF_BASE", 0.5),
                _env_float("CLOUD_BACKOFF_CAP", 8.0),
            ),
            remaining,
        )
        with self._lock:
            self._retry_count += 1
        obs.log_event(
            "cloud_retry",
            model=model,
            reason=reason,
            attempt=attempt + 1,
            delay=round(delay, 3),
            error=str(exc) or type(exc).__name__,
            level="warning",
        )
        time.sleep(delay)

    def _note_budget_exceeded(self, category: str) -> None:
        """Record that the per-task cloud budget ran out before an answer."""
        with self._lock:
            self._budget_exceeded += 1
        obs.log_event(
            "cloud_budget_exceeded",
            category=category,
            budget_seconds=self.task_budget,
            level="warning",
        )

    def _call(self, model, system, user, max_tokens, effort="none"):
        # reasoning_effort goes through extra_body to bypass client-side
        # validation; "none" fully disables thinking on kimi/minimax and is
        # the single biggest completion-token saver.
        extra = {"reasoning_effort": effort} if effort else {}
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body=extra,
        )
        usage = getattr(response, "usage", None)
        # Some OpenAI-compatible proxies omit fields or return a partial usage
        # object; never let accounting turn a good answer into a failed task.
        totals = usage_totals(usage)
        with self._lock:
            self._calls += 1
            self._total_tokens += totals["total"]
        text = response_text(response)
        if not text:
            with self._lock:
                self._empty_responses += 1
        if os.environ.get("AGENT_DEBUG"):
            print(
                f"    call model={model} prompt={totals['prompt']} "
                f"completion={totals['completion']} total={totals['total']}",
                flush=True,
            )
        obs.log_event(
            "cloud_call",
            model=model,
            prompt_tokens=totals["prompt"],
            completion_tokens=totals["completion"],
            total_tokens=totals["total"],
            empty=not text,
        )
        return text
