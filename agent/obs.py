"""Structured run logging and per-run metrics.

Two opt-in outputs keep a graded run quiet by default while making regressions
visible when they are switched on:

- ``AGENT_LOG_JSON=1`` writes one JSON object per event to stdout (task id,
  category, lane, elapsed seconds, token usage, retry reasons). ``AGENT_DEBUG``
  stays available for ad hoc human-readable token lines.
- ``METRICS_PATH`` writes a per-run metrics snapshot (tokens by category, lane
  counts, escalations, empty answers, cloud retries) so a cost or accuracy
  regression shows up without reading thousands of log lines.

Both are plain ``json`` on stdout/a file: nothing here is billed, nothing is
imported from the optional extras, and no logging framework is required.
"""

import json
import os
import threading
import time
from collections import Counter

# Unset or any of these means "off"; everything else is on. Mirrors the
# specialist flags in agent/specialists.py so one convention covers the repo.
_FALSE_VALUES = {"", "0", "false", "no", "off", "disabled"}

_LOG_LOCK = threading.Lock()


def flag(name: str) -> bool:
    """Read a boolean environment flag, defaulting to False when unset."""
    return os.environ.get(name, "").strip().lower() not in _FALSE_VALUES


def logging_enabled() -> bool:
    """True when structured JSON logging is switched on for this process."""
    return flag("AGENT_LOG_JSON")


def metrics_path() -> str:
    """Configured metrics output path ('' when metrics export is disabled).

    A dry run never writes this file: its lane counts are predictions, and
    mixing them into a measured run's snapshot would corrupt regressions.
    """
    return os.environ.get("METRICS_PATH", "").strip()


def log_event(event: str, task_id: str = "", level: str = "info", **fields) -> None:
    """Emit one JSON log line, or nothing when JSON logging is disabled.

    Args:
        event: Short event name, e.g. ``task``, ``cloud_call``, ``run_summary``.
        task_id: Task the event belongs to (omitted when empty).
        level: Severity hint for log shippers (``info``/``warning``/``error``).
        **fields: Extra JSON-serializable fields; unknown objects fall back to
            ``str`` so a log line can never raise on an odd provider object.
    """
    if not logging_enabled():
        return
    record = {"ts": round(time.time(), 3), "level": level, "event": event}
    if task_id:
        record["task_id"] = task_id
    record.update(fields)
    line = json.dumps(record, ensure_ascii=False, default=str)
    # One write per line keeps concurrent workers from interleaving output.
    with _LOG_LOCK:
        print(line, flush=True)


class RunMetrics:
    """Thread-safe counters for a single run.

    Lanes are recorded as the lane that actually produced the answer, so
    ``lanes`` sums to the task count while ``categories`` exposes the router's
    classification and ``tokens_by_category`` mirrors the billed spend.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self.reset()

    def reset(self) -> None:
        """Clear every counter (called once per run, or by tests)."""
        with self._lock:
            self._started = time.time()
            self.tasks = 0
            self.lanes: Counter = Counter()
            self.categories: Counter = Counter()
            self.escalations: Counter = Counter()
            self.empty_answers = 0
            self.low_confidence = 0
            self.tokens_total = 0
            self.tokens_by_category: Counter = Counter()
            self.duplicate_task_ids = 0
            self.skipped_duplicates = 0
            self.missing_prompts = 0
            self.elapsed_seconds = 0.0

    def record_task(
        self,
        category: str,
        lane: str,
        tokens: int = 0,
        escalated: bool = False,
        empty: bool = False,
        low_confidence: bool = False,
    ) -> None:
        """Count one finished task by lane, category, tokens, and outcome."""
        tokens = tokens if isinstance(tokens, int) and tokens > 0 else 0
        with self._lock:
            self.tasks += 1
            self.lanes[lane] += 1
            self.categories[category] += 1
            self.tokens_total += tokens
            self.tokens_by_category[category] += tokens
            if escalated:
                self.escalations[lane] += 1
            if empty:
                self.empty_answers += 1
            if low_confidence:
                self.low_confidence += 1

    def record_duplicates(self, duplicate_ids: int, skipped: int) -> None:
        """Record how many task_ids repeated and how many rows were collapsed."""
        with self._lock:
            self.duplicate_task_ids += max(0, int(duplicate_ids))
            self.skipped_duplicates += max(0, int(skipped))

    def record_missing_prompts(self, count: int) -> None:
        """Record rows whose prompt was missing or not a non-empty string."""
        with self._lock:
            self.missing_prompts += max(0, int(count))

    def snapshot(self, cloud=None, elapsed: float = None) -> dict:
        """Return a JSON-ready metrics dict.

        Args:
            cloud: Optional cloud client; when it exposes ``total_tokens()``,
                ``call_count()``, ``tokens_by_category()``, ``retry_count()``,
                ``empty_response_count()``, or ``budget_exceeded_count()``,
                those authoritative numbers are merged in so the export agrees
                with the billed total.
            elapsed: Wall-clock seconds for the run (defaults to the time since
                ``reset()``).
        """
        with self._lock:
            tasks = self.tasks
            snapshot = {
                "tasks": tasks,
                "lanes": dict(self.lanes),
                "categories": dict(self.categories),
                "escalations": dict(self.escalations),
                "empty_answers": self.empty_answers,
                "low_confidence_routes": self.low_confidence,
                "duplicates": {
                    "repeated_task_ids": self.duplicate_task_ids,
                    "skipped_rows": self.skipped_duplicates,
                },
                "missing_prompts": self.missing_prompts,
                "tokens": {
                    "total": self.tokens_total,
                    # Zero-token lanes are already visible in "categories";
                    # listing them here as 0 just adds noise.
                    "by_category": {
                        category: tokens
                        for category, tokens in self.tokens_by_category.items()
                        if tokens
                    },
                    "per_task": round(self.tokens_total / tasks, 2) if tasks else 0.0,
                },
                "cloud": self._cloud_snapshot(cloud),
                "elapsed_seconds": round(
                    self.elapsed_seconds if elapsed is None else elapsed, 3
                ),
            }
        return snapshot

    @staticmethod
    def _cloud_snapshot(cloud) -> dict:
        """Best-effort copy of a cloud client's own counters."""

        def call(name):
            method = getattr(cloud, name, None)
            if not callable(method):
                return None
            try:
                return method()
            except Exception:  # accounting must never break the run summary
                return None

        def count(name):
            value = call(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
            return 0

        by_category = call("tokens_by_category")
        return {
            "calls": count("call_count"),
            "tokens": count("total_tokens"),
            "retries": count("retry_count"),
            "empty_responses": count("empty_response_count"),
            "budget_exceeded": count("budget_exceeded_count"),
            "tokens_by_category": by_category if isinstance(by_category, dict) else {},
        }

    def save(self, path: str, cloud=None, elapsed: float = None) -> dict:
        """Write the snapshot to ``path`` (creating directories) and return it."""
        snapshot = self.snapshot(cloud=cloud, elapsed=elapsed)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=1)
        return snapshot


# One process-wide instance: the entrypoint is a single run per container, and
# a shared counter keeps eval/judge's direct answer_task() calls accounted too.
metrics = RunMetrics()
