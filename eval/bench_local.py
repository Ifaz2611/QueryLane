"""Head-to-head local model benchmark under grading constraints.

For each GGUF passed as an argument: load, run the same warm-up, then measure
steady-state generation speed and answer quality-relevant latencies (a
sentiment-sized 40-token generation and a PAL-sized 20-token generation).

Run inside the container with --cpus=2 --memory=4g to mirror the grader.
"""

import sys
import time


def bench(path: str):
    from llama_cpp import Llama

    print(f"=== {path}")
    load_start = time.time()
    try:
        llm = Llama(model_path=path, n_ctx=2048, n_threads=2, verbose=False)
    except Exception as exc:
        print(f"  LOAD FAILED: {exc}")
        return
    print(f"  load: {time.time() - load_start:.1f}s")

    def gen(prompt: str, max_tokens: int):
        start = time.time()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        usage = out.get("usage", {})
        completion = usage.get("completion_tokens", max_tokens)
        text = out["choices"][0]["message"]["content"] or ""
        return completion / max(elapsed, 0.01), elapsed, text

    gen("Count: 1 2 3", 16)  # cold warm-up, unmeasured
    tps, elapsed, _ = gen("List ten animals, one word each.", 40)
    print(f"  steady-state: {tps:.1f} tok/s ({elapsed:.1f}s for 40 tok)")
    tps2, elapsed2, text = gen(
        "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. "
        "How many remain? Output exactly one line: 'EXPR: <expression> | UNIT: <unit>'.",
        30,
    )
    print(f"  PAL-sized gen: {elapsed2:.1f}s -> {text.strip()[:80]!r}")
    sentiment = (
        "Classify the sentiment (one label plus one short sentence): "
        "The battery is great but the screen scratches too easily."
    )
    tps3, elapsed3, text3 = gen(sentiment, 50)
    print(f"  sentiment-sized gen: {elapsed3:.1f}s -> {text3.strip()[:80]!r}")


if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        bench(model_path)
