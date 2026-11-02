"""QueryLane agent entrypoint.

Contract:
- read /input/tasks.json:  [{"task_id": "...", "prompt": "..."}, ...]
- write /output/results.json: [{"task_id": "...", "answer": "..."}, ...]
- exit 0 on success, non-zero on failure, all inside the 10-minute limit.

Every task_id always gets a string answer, even on per-task failure, so the
schema stays valid no matter what. A global deadline guarantees results are
written well before the harness timeout.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from agent import local as local_model
from agent import pal, specialists
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


def answer_task(cloud: CloudClient, task: dict) -> dict:
    task_id = str(task.get("task_id", ""))
    prompt = str(task.get("prompt", ""))
    category, target, confidence = _route(prompt)
    low_confidence = confidence < MIN_CONFIDENCE
    started = time.time()

    if category == "math" and target == "local" and pal.is_multipart(prompt):
        target = "cloud"  # single-EXPR local path can't answer two quantities

    answer = ""
    # Non-generative specialists first: ~1s per task on any CPU, zero tokens,
    # and they escalate ("") on anything uncertain. Low-confidence routing
    # skips them so an ambiguous prompt can never be answered by a specialist.
    if category == "sentiment" and not low_confidence and specialists.available("sst2"):
        try:
            answer = specialists.sentiment_answer(prompt)
            if answer:
                target = "specialist"
        except Exception as exc:
            print(f"[{task_id}] sentiment specialist failed: {exc}", flush=True)
            answer = ""
    elif category == "ner" and not low_confidence and specialists.ner_enabled():
        # Self-gating: complete capitalized-span coverage plus a high token
        # confidence bar, otherwise "" falls through to cloud.
        try:
            answer = specialists.ner_answer(prompt)
            if answer:
                target = "specialist"
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
            try:
                answer = local_model.answer(category, prompt)
            except Exception as exc:  # local failure falls through to cloud
                print(f"[{task_id}] local failed: {exc}", flush=True)
                answer = ""
        else:
            target = "cloud-fallback"  # local unavailable, too slow, or over budget
        if answer and not local_model.answer_ok(category, answer, prompt):
            print(f"[{task_id}] local answer suspicious, escalating", flush=True)
            answer = ""
            target = "escalated"

    if not answer.strip():
        try:
            answer = local_model.normalize(category, cloud.answer(category, prompt))
        except Exception as exc:
            print(f"[{task_id}] cloud failed: {exc}", flush=True)
            answer = ""

    print(
        f"[{task_id}] category={category} route={target} "
        f"conf={confidence:.2f} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return {
        "task_id": task_id,
        "answer": answer.strip(),
        "category": category,
        "routed": target,
    }


def main() -> int:
    global _START
    _START = time.time()
    start = _START
    # Read paths at call time so tests and embedders can override them without
    # reloading the module (the container still gets the /input defaults).
    input_path = os.environ.get("TASKS_PATH", INPUT_PATH)
    output_path = os.environ.get("RESULTS_PATH", OUTPUT_PATH)
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

    try:
        cloud = CloudClient.from_env()
    except (KeyError, ValueError) as exc:
        print(f"bad cloud configuration: {exc}", file=sys.stderr, flush=True)
        return 1

    local_model.preload()  # load + speed-calibrate in the background

    results = {
        str(task.get("task_id", "")): "" for task in tasks
    }  # every id present even if a worker never finishes

    # Batching same-category tasks was measured as a net token LOSS at real
    # group sizes (2-3 per category): the format instruction plus echoed item
    # markers cost more than the saved per-call overhead. One call per task.
    def submit_order(task):
        prompt = str(task.get("prompt", ""))
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
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    output = [
        {
            "task_id": str(task.get("task_id", "")),
            "answer": results[str(task.get("task_id", ""))],
        }
        for task in tasks
    ]
    # abspath keeps dirname non-empty when RESULTS_PATH is a bare filename
    # (e.g. "results.json"), which would make makedirs("") raise.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=1)

    print(
        f"done: {len(output)} tasks, {cloud.call_count()} fireworks calls, "
        f"{cloud.total_tokens()} fireworks tokens, "
        f"{time.time() - start:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard exit: lingering worker threads must not hold the container past the
    # harness timeout after results are already on disk.
    os._exit(code)
