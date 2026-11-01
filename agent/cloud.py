"""Fireworks client.

Every call goes through FIREWORKS_BASE_URL (required for token accounting) and
logs usage.total_tokens. One answer per task: no content-based retries, only
transient-error retries and model fallback, because every retry costs tokens.
"""

import os
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

from agent import pal, prompts

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


class CloudClient:
    def __init__(self, api_key: str, base_url: str, allowed_models: list):
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0
        )
        self._allowed = allowed_models
        self._lock = threading.Lock()
        self._total_tokens = 0
        self._calls = 0

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
        if category == "math":
            # Cloud math path kept behaviorally identical to the twice-scored
            # v13: PAL first, step-by-step only on parse failure. The multipart
            # detector gates only the LOCAL path (agent/main.py), where routing
            # a two-quantity question away from single-EXPR is pure upside.
            text = self._answer_with_spec(prompts.SPECS[category], category, prompt)
            parsed = pal.answer_from_expression(text)
            if parsed is not None:
                return parsed
            return self._answer_with_spec(prompts.MATH_FALLBACK, category, prompt)
        return self._answer_with_spec(prompts.SPECS[category], category, prompt)

    def _answer_with_spec(self, spec, category: str, prompt: str) -> str:
        last_error = None
        for model in self._models_for(category):
            effort = spec.effort
            # Second budget exists only for thinking models whose reasoning can
            # eat the whole first budget and leave content empty.
            for max_tokens in (spec.max_tokens, spec.max_tokens * 4):
                text, error = self._call_with_retries(
                    model, spec.system, prompt, max_tokens, effort
                )
                if text:
                    return text
                if error is not None:
                    # A grading proxy that rejects reasoning_effort would 400
                    # every call; drop the field once and retry before giving
                    # up on the model.
                    if effort and getattr(error, "status_code", None) == 400:
                        text, error = self._call_with_retries(
                            model, spec.system, prompt, max_tokens, None
                        )
                        if text:
                            return text
                        effort = None
                    if error is not None:
                        last_error = error
                        break  # model-level problem: move to the next model
        raise RuntimeError(f"all models failed for {category}: {last_error}")

    def _call_with_retries(self, model, system, user, max_tokens, effort):
        """Returns (text, error). error is set only for model-level failures
        (bad id, undeployed, 4xx) that should skip to the next model."""
        for attempt in range(1 + _TRANSIENT_RETRIES):
            try:
                return self._call(model, system, user, max_tokens, effort), None
            except (APIConnectionError, APITimeoutError):
                continue
            except RateLimitError:
                time.sleep(2 * (attempt + 1))
                continue
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    continue
                return "", exc
        return "", None

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
        total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = (
            getattr(usage, "completion_tokens", 0) if usage is not None else 0
        )
        with self._lock:
            self._calls += 1
            if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
                if total_tokens > 0:
                    self._total_tokens += total_tokens
        if os.environ.get("AGENT_DEBUG") and usage is not None:
            print(
                f"    call model={model} prompt={prompt_tokens} "
                f"completion={completion_tokens} total={total_tokens}",
                flush=True,
            )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) or ""
        return _strip_thinking(content)
