"""QueryLane agent entrypoint.

Contract:
- read /input/tasks.json:  [{"task_id": "...", "prompt": "..."}, ...]
- write /output/results.json: [{"task_id": "...", "answer": "..."}, ...]
- exit 0 on success, non-zero on failure, all inside the 10-minute limit.

Every task_id always gets a string answer, even on per-task failure, so the
schema stays valid no matter what. A global deadline guarantees results are
written well before the harness timeout.

Optional diagnosis:
- ``--dry-run`` (or ``AGENT_DRY_RUN=1``) routes every task, writes the lane each
  one would take, and never touches the cloud.
- ``AGENT_LOG_JSON=1`` emits one JSON line per task; ``METRICS_PATH`` writes a
  per-run metrics snapshot. Neither costs tokens.
"""

import json
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from agent import local as local_model
from agent import obs, pal, specialists
from agent.cloud import CloudClient
from agent.router import MIN_CONFIDENCE, route

INPUT_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")
MAX_WORKERS = max(1, int(os.environ.get("AGENT_WORKERS", "4")))
# Answer budget in seconds. The harness kills us at 600; stop early and write
# whatever is finished.
DEADLINE_SECONDS = float(os.environ.get("AGENT_DEADLINE", "480"))
# A local generation must not still be running when the deadline hits (it
# would be recorded as an empty answer); flip to cloud when this close.
LOCAL_CUTOFF_SECONDS = float(os.environ.get("LOCAL_CUTOFF", "150"))

_START = time.time()

# Only routing switches are parsed; anything else a harness passes is ignored.
_ARG_PARSER = ArgumentParser(add_help=False, allow_abbrev=False)
_ARG_PARSER.add_argument(
    "--dry-run",
    action="store_true",
    help="route every task and report lanes without any cloud or local call",
)
_ARG_PARSER.add_argument(
    "--explain",
    type=str,
    nargs="?",
    const="",
    default=None,
    help="explain the lane decision for a prompt and exit",
)


def _remaining() -> float:
    return DEADLINE_SECONDS - (time.time() - _START)


def _route(prompt: str):
    """Route with confidence, tolerating old two-tuple test/embedder patches."""
    result = route(prompt, with_confidence=True)
    if len(result) == 2:  # backwards-compatible patch or embedder
        return result[0], result[1], 1.0
    return result


# Highest savings-per-second first: math-PAL emits ~15 tokens (cheapest by
# far), then NER, then factual. On a slow grader the budget dies before the
# list ends, so order decides which savings actually land. Cloud-routed tasks
# go first overall since they only occupy a worker for seconds.
_LOCAL_COST_ORDER = {
    "math": 0,
    "ner": 1,
    "sentiment": 2,
    "factual": 3,
    "summarization": 4,
    "logic": 5,
}


def _get_cloud_tokens(cloud) -> int:
    try:
        val = cloud.total_tokens()
        return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0
    except Exception:
        return 0


def _prompt_of(task: dict) -> str:
    """Return the task's prompt as a string, treating null/non-strings as empty."""
    prompt = task.get("prompt")
    return prompt if isinstance(prompt, str) else ""


def predict_lane(category: str, target: str, confidence: float, prompt: str) -> str:
    """Best guess at the lane that will actually produce the answer.

    No model is run, so this mirrors the gates in ``answer_task``: a
    low-confidence route skips the specialists, local needs the GGUF weights and
    a proven/enabled category, and everything else lands on cloud. Used by
    ``--dry-run`` (and the per-task log line) before any tokens are spent.
    ``"specialist"`` appears here only when a specialist model file exists; the
    specialists themselves may still decline and escalate at run time.
    """
    if confidence < MIN_CONFIDENCE:
        return "cloud"  # same short-circuit as the router, cloud is the fallback
    if category == "sentiment" and specialists.available("sst2"):
        return "specialist"
    if category == "ner" and specialists.ner_enabled():
        return "specialist"
    if target != "local":
        return "cloud"
    if category == "math" and pal.is_multipart(prompt):
        return "cloud"  # single-EXPR local path cannot answer two quantities
    if local_model.available() and local_model.category_enabled(category):
        return "local"
    return "cloud"


def _escalation_lane(lane: str) -> str:
    """Lane name recorded when a cheaper lane declined or its answer looked bad.

    Escalations always end in cloud, so this keeps the historical
    ``cloud-fallback`` / ``escalated`` labels for them while local and
    specialist wins keep their own names.
    """
    return "escalated" if lane == "local" else "cloud-fallback"


def _record_task(entry: dict, elapsed: float) -> None:
    """Log one structured task line and feed the per-run metrics counters."""
    empty = not entry["answer"]
    obs.metrics.record_task(
        entry["category"],
        entry["routed"],
        tokens=entry["tokens"],
        escalated=entry["routed"] in ("cloud-fallback", "escalated"),
        empty=empty,
        low_confidence=entry["confidence"] < MIN_CONFIDENCE,
    )
    obs.log_event(
        "task",
        task_id=entry["task_id"],
        category=entry["category"],
        lane=entry["routed"],
        confidence=entry["confidence"],
        predicted_lane=entry["predicted"],
        elapsed_seconds=round(elapsed, 3),
        # Tokens billed for this task alone. Task-level counters are read
        # concurrently by every worker, so a shared-delta reading can overlap
        # another worker's call; the exact total lives in "run_summary".
        tokens=entry["tokens"],
        empty_answer=empty,
    )


def dry_run_entry(task: dict) -> dict:
    """Route one task without any model or cloud call."""
    started = time.time()
    task_id = str(task.get("task_id", ""))
    prompt = _prompt_of(task)
    if not prompt.strip():
        obs.metrics.record_missing_prompts(1)
    category, target, confidence = _route(prompt)
    lane = predict_lane(category, target, confidence, prompt)
    # Predicted lanes are counted so a dry-run summary shows the lane mix; no
    # answer was attempted, so nothing is counted as an empty answer.
    obs.metrics.record_task(
        category, lane, tokens=0, low_confidence=confidence < MIN_CONFIDENCE
    )
    entry = {
        "task_id": task_id,
        "answer": "",
        "category": category,
        "routed": lane,
        "predicted": lane,
        "confidence": confidence,
        "tokens": 0,
    }
    obs.log_event(
        "task",
        task_id=task_id,
        category=category,
        lane=lane,
        confidence=confidence,
        predicted_lane=lane,
        elapsed_seconds=round(time.time() - started, 3),
        tokens=0,
        dry_run=True,
    )
    return entry


def answer_task(cloud: CloudClient, task: dict) -> dict:
    task_id = str(task.get("task_id", ""))
    prompt = _prompt_of(task)
    category, target, confidence = _route(prompt)
    low_confidence = confidence < MIN_CONFIDENCE
    started = time.time()
    tokens_before = _get_cloud_tokens(cloud)
    if not prompt.strip():
        obs.metrics.record_missing_prompts(1)

    if category == "math" and target == "local" and pal.is_multipart(prompt):
        target = "cloud"  # single-EXPR local path can't answer two quantities

    entry = {
        "task_id": task_id,
        "answer": "",
        "category": category,
        # "cloud" is the lane of last resort; _answer_task_once overwrites this
        # as soon as a cheaper lane actually produces the answer.
        "routed": "cloud",
        "predicted": predict_lane(category, target, confidence, prompt),
        "confidence": confidence,
        "tokens": 0,
    }
    try:
        entry["answer"] = _answer_task_once(
            cloud, task_id, prompt, category, target, low_confidence, entry
        )
    finally:
        # Metrics and the JSON log line are recorded even if a lane raises, so
        # a crashed task is still visible as an empty answer.
        entry["tokens"] = max(0, _get_cloud_tokens(cloud) - tokens_before)
        _record_task(entry, time.time() - started)
    return entry


def _answer_task_once(cloud, task_id, prompt, category, target, low_confidence, entry) -> str:
    """Run the lane cascade for one task and return its final answer string.

    ``entry`` is mutated so the reported lane reflects the last lane tried.
    """
    answer = ""
    # Non-generative specialists first: ~1s per task on any CPU, zero tokens,
    # and they escalate ("") on anything uncertain. Low-confidence routing
    # skips them so an ambiguous prompt can never be answered by a specialist.
    if category == "sentiment" and not low_confidence and specialists.available("sst2"):
        try:
            answer = specialists.sentiment_answer(prompt)
            if answer:
                entry["routed"] = "specialist"
        except Exception as exc:
            print(f"[{task_id}] sentiment specialist failed: {exc}", flush=True)
            answer = ""
    elif category == "ner" and not low_confidence and specialists.ner_enabled():
        # Self-gating: complete capitalized-span coverage plus a high token
        # confidence bar, otherwise "" falls through to cloud.
        try:
            answer = specialists.ner_answer(prompt)
            if answer:
                entry["routed"] = "specialist"
        except Exception as exc:
            print(f"[{task_id}] ner specialist failed: {exc}", flush=True)
            answer = ""

    if not answer and target == "local":
        if (
            _remaining() > LOCAL_CUTOFF_SECONDS
            and local_model.available()
            and local_model.wait_ready(150)
            and local_model.category_enabled(category)
            and _remaining() > LOCAL_CUTOFF_SECONDS  # re-check after the wait
        ):
            entry["routed"] = "local"
            try:
                answer = local_model.answer(category, prompt)
            except Exception as exc:  # local failure falls through to cloud
                print(f"[{task_id}] local failed: {exc}", flush=True)
                answer = ""
        else:
            # local unavailable, too slow, or over budget
            entry["routed"] = _escalation_lane("cloud")
        if answer and not local_model.answer_ok(category, answer, prompt):
            print(f"[{task_id}] local answer suspicious, escalating", flush=True)
            answer = ""
            entry["routed"] = _escalation_lane("local")

    if not answer.strip():
        try:
            answer = local_model.normalize(category, cloud.answer(category, prompt))
        except Exception as exc:
            print(f"[{task_id}] cloud failed: {exc}", flush=True)
            answer = ""
    return answer.strip()


def main(argv=None) -> int:
    global _START
    _START = time.time()
    start = _START
    if _wants_help(argv):
        _print_usage()
        return 0
    raw_args = sys.argv[1:] if argv is None else argv
    try:
        parsed, _ = _ARG_PARSER.parse_known_args(raw_args)
        if getattr(parsed, "explain", None) is not None:
            from agent.explain import explain_decision, format_explanation

            info = explain_decision(parsed.explain)
            print(format_explanation(info))
            return 0
    except SystemExit:
        pass
    dry_run, _argv = _parse_args(argv)
    # Read paths at call time so tests and embedders can override them without
    # reloading the module (the container still gets the /input defaults).
    input_path = os.environ.get("TASKS_PATH", INPUT_PATH)
    output_path = os.environ.get("RESULTS_PATH", OUTPUT_PATH)
    run_metrics = obs.metrics
    run_metrics.reset()
    try:
        with open(input_path, encoding="utf-8") as fh:
            tasks = json.load(fh)
    except Exception as exc:
        print(f"failed to read {input_path}: {exc}", file=sys.stderr, flush=True)
        return 1
    if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
        print(
            f"{input_path} must be a JSON array of task objects",
            file=sys.stderr,
            flush=True,
        )
        return 1

    unique_tasks, duplicates = _dedupe_tasks(tasks)
    if duplicates:
        print(
            f"input repeats {duplicates} task_id(s); answering each once",
            file=sys.stderr,
            flush=True,
        )
        obs.log_event("duplicate_task_ids", count=duplicates, level="warning")

    cloud = None
    if dry_run:
        print("dry run: reporting lanes only, no cloud or local calls", flush=True)
        obs.log_event("dry_run", tasks=len(unique_tasks))
    else:
        try:
            cloud = CloudClient.from_env()
        except (KeyError, ValueError) as exc:
            print(f"bad cloud configuration: {exc}", file=sys.stderr, flush=True)
            return 1
        local_model.preload()  # load + speed-calibrate in the background

    # Every id is present even if a worker never finishes.
    results = {str(task.get("task_id", "")): "" for task in unique_tasks}

    if dry_run:
        for task in unique_tasks:
            entry = dry_run_entry(task)
            results[entry["task_id"]] = entry["answer"]
    else:
        _answer_batch(cloud, unique_tasks, results, start)

    # Rows are rebuilt from the original list so repeated ids keep their place
    # and receive the answer the first occurrence produced.
    output = [
        {
            "task_id": str(task.get("task_id", "")),
            "answer": results[str(task.get("task_id", ""))],
        }
        for task in tasks
    ]
    run_metrics.record_duplicates(duplicates, duplicates)
    # abspath keeps dirname non-empty when RESULTS_PATH is a bare filename
    # (e.g. "results.json"), which would make makedirs("") raise.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=1)

    elapsed = time.time() - start
    if dry_run or cloud is None:
        totals = f"0 fireworks tokens (dry run), {elapsed:.1f}s"
        calls = 0
    else:
        calls = cloud.call_count()
        totals = (
            f"{calls} fireworks calls, {cloud.total_tokens()} fireworks tokens, "
            f"{elapsed:.1f}s"
        )
    print(f"done: {len(output)} tasks, {totals}", flush=True)
    _finish_run(cloud, elapsed, dry_run)
    return 0


def _answer_batch(cloud, tasks: list, results: dict, start: float) -> None:
    """Answer every task in the thread pool and collect finished answers."""

    # Batching same-category tasks was measured as a net token LOSS at real
    # group sizes (2-3 per category): the format instruction plus echoed item
    # markers cost more than the saved per-call overhead. One call per task.
    def submit_order(task):
        prompt = _prompt_of(task)
        category, target, _ = _route(prompt)
        # Multipart math is flipped to cloud inside answer_task; schedule it
        # with the cloud tasks instead of letting it sit behind local work.
        if target != "local" or (category == "math" and pal.is_multipart(prompt)):
            return (0, 0)  # cloud first: seconds per task, frees workers fast
        return (1, _LOCAL_COST_ORDER.get(category, 9))

    ordered = sorted(tasks, key=submit_order)

    # Do not use a `with` block: its __exit__ calls shutdown(wait=True), which
    # would wait for slow workers past the deadline. We write whatever finished
    # and hard-exit in __main__.
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {
        pool.submit(answer_task, cloud, task): str(task.get("task_id", ""))
        for task in ordered
    }
    try:
        for future, task_id in futures.items():
            remaining = DEADLINE_SECONDS - (time.time() - start)
            try:
                entry = future.result(timeout=max(remaining, 1.0))
                results[entry["task_id"]] = entry["answer"]
            except Exception as exc:
                print(f"[{task_id}] gave up: {exc}", flush=True)
                obs.log_event(
                    "task_abandoned",
                    task_id=task_id,
                    error=str(exc) or type(exc).__name__,
                    level="error",
                )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _parse_args(argv=None):
    """Return ``(dry_run, argv)`` for the optional agent switches.

    Only ``--dry-run`` is consumed; unknown arguments are ignored so the same
    entrypoint keeps working when a harness or test runner passes its own
    flags.
    """
    if argv is None:
        argv = sys.argv[1:]
    try:
        args, _unknown = _ARG_PARSER.parse_known_args(argv)
        dry_run = args.dry_run
    except SystemExit:  # argparse exits on --help; treat as "no flags"
        dry_run = False
    if not dry_run and obs.flag("AGENT_DRY_RUN"):
        dry_run = True
    return dry_run, argv


def _wants_help(argv) -> bool:
    """True when the caller asked for usage instead of a run."""
    if argv is None:
        argv = sys.argv[1:]
    return any(arg in ("-h", "--help") for arg in argv)


def _print_usage(stream=None) -> None:
    """Print the entrypoint's usage, switches, and main env vars."""
    stream = stream or sys.stdout
    print(
        "usage: querylane [-h] [--dry-run] [--explain [PROMPT]]\n"
        "\n"
        "Route each task in TASKS_PATH to the cheapest lane that can answer it and\n"
        "write answers to RESULTS_PATH.\n"
        "\n"
        "options:\n"
        "  -h, --help            show this message and exit\n"
        "  --dry-run             route every task and report lanes without any cloud or\n"
        "                        local call (also AGENT_DRY_RUN=1)\n"
        "  --explain [PROMPT]    explain the routing and lane decision for a prompt and exit\n"
        "\n"
        "observability:\n"
        "  AGENT_LOG_JSON=1     one JSON event per task/cloud call\n"
        "  METRICS_PATH=<file>  per-run metrics snapshot\n"
        "  AGENT_DEBUG=1        human-readable per-call token lines\n"
        "\n"
        "see README.md for every environment variable",
        file=stream,
        flush=True,
    )


def _dedupe_tasks(tasks: list):
    """Collapse rows that repeat a task_id, keeping the first occurrence.

    Returns ``(unique_tasks, duplicate_count)``. Duplicates are dropped so each
    id is billed and answered once; the output rows for dropped duplicates are
    rebuilt from the first task's answer because results are keyed by task_id.
    """
    unique = []
    seen = set()
    duplicates = 0
    for task in tasks:
        key = str(task.get("task_id", ""))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(task)
    return unique, duplicates


def _finish_run(cloud, elapsed: float, dry_run: bool) -> None:
    """Write the metrics snapshot (if requested) and the JSON run summary."""
    path = obs.metrics_path()
    snapshot = obs.metrics.snapshot(cloud=cloud, elapsed=elapsed)
    if not dry_run and path:
        try:
            obs.metrics.save(path, cloud=cloud, elapsed=elapsed)
            print(f"metrics written to {path}", flush=True)
        except Exception as exc:  # a metrics failure must not fail the run
            print(f"failed to write metrics to {path}: {exc}", file=sys.stderr, flush=True)
    obs.log_event("run_summary", dry_run=dry_run, **snapshot)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard exit: lingering worker threads must not hold the container past the
    # harness timeout after results are already on disk.
    os._exit(code)
