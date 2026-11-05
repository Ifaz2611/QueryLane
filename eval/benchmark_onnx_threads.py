"""Benchmark ONNX Runtime intra_op_num_threads and cross-process session caching.

Evaluates inference latency, throughput, and CPU scaling across thread counts (1, 2, 4, 8)
on the current host to find the optimal ONNX threading configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort

from agent.specialists import (
    NER_DIR,
    SST2_DIR,
    get_optimal_intra_op_threads,
)


def _create_synthetic_model(output_path: str, hidden_dim: int = 512, ffn_dim: int = 2048) -> None:
    """Create a synthetic transformer-like ONNX model to benchmark CPU tensor ops."""
    import onnx
    from onnx import TensorProto, helper

    nodes = []
    initializers = []
    current_input = "input_ids"

    rng = np.random.default_rng(42)

    # 4 layers of MatMul + Add + Relu + MatMul (simulating transformer FFN blocks)
    for layer in range(4):
        w1_name = f"W1_l{layer}"
        b1_name = f"b1_l{layer}"
        w2_name = f"W2_l{layer}"
        b2_name = f"b2_l{layer}"

        h1_name = f"h1_l{layer}"
        act1_name = f"act1_l{layer}"
        out_name = f"out_l{layer}" if layer < 3 else "logits"

        w1_weights = rng.normal(0, 0.02, (hidden_dim, ffn_dim)).astype(np.float32).flatten().tolist()
        b1_weights = np.zeros(ffn_dim, dtype=np.float32).tolist()
        w2_weights = rng.normal(0, 0.02, (ffn_dim, hidden_dim)).astype(np.float32).flatten().tolist()
        b2_weights = np.zeros(hidden_dim, dtype=np.float32).tolist()

        initializers.append(helper.make_tensor(w1_name, TensorProto.FLOAT, [hidden_dim, ffn_dim], w1_weights))
        initializers.append(helper.make_tensor(b1_name, TensorProto.FLOAT, [1, ffn_dim], b1_weights))
        initializers.append(helper.make_tensor(w2_name, TensorProto.FLOAT, [ffn_dim, hidden_dim], w2_weights))
        initializers.append(helper.make_tensor(b2_name, TensorProto.FLOAT, [1, hidden_dim], b2_weights))

        nodes.append(helper.make_node("MatMul", [current_input, w1_name], [f"mm1_l{layer}"]))
        nodes.append(helper.make_node("Add", [f"mm1_l{layer}", b1_name], [h1_name]))
        nodes.append(helper.make_node("Relu", [h1_name], [act1_name]))
        nodes.append(helper.make_node("MatMul", [act1_name, w2_name], [f"mm2_l{layer}"]))
        nodes.append(helper.make_node("Add", [f"mm2_l{layer}", b2_name], [out_name]))

        current_input = out_name

    graph = helper.make_graph(
        nodes,
        "synthetic_transformer_benchmark",
        [helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, hidden_dim])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, hidden_dim])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="querylane-benchmark",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.save(model, output_path)


def benchmark_model_threads(
    model_path: str,
    thread_counts: List[int],
    iterations: int = 40,
) -> Dict[str, Any]:
    """Benchmark a model across thread counts."""
    results = {}

    tmp_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    feed = {}
    for inp in tmp_sess.get_inputs():
        shape = [dim if (isinstance(dim, int) and dim > 0) else (64 if "mask" in inp.name or "ids" in inp.name else 1) for dim in inp.shape]
        if inp.type == "tensor(int64)":
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    baseline_latency = None

    for threads in thread_counts:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Measure session init / cold start
        init_start = time.perf_counter()
        session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        init_time_ms = (time.perf_counter() - init_start) * 1000.0

        # Warmup
        for _ in range(8):
            session.run(None, feed)

        # Timed runs
        timings = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            session.run(None, feed)
            timings.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(timings))
        p50_ms = float(np.percentile(timings, 50))
        p95_ms = float(np.percentile(timings, 95))
        p99_ms = float(np.percentile(timings, 99))
        throughput = 1000.0 / mean_ms if mean_ms > 0 else 0

        if baseline_latency is None:
            baseline_latency = mean_ms
        speedup = (baseline_latency / mean_ms) if mean_ms > 0 else 1.0

        results[str(threads)] = {
            "threads": threads,
            "init_time_ms": round(init_time_ms, 2),
            "mean_latency_ms": round(mean_ms, 3),
            "p50_latency_ms": round(p50_ms, 3),
            "p95_latency_ms": round(p95_ms, 3),
            "p99_latency_ms": round(p99_ms, 3),
            "throughput_qps": round(throughput, 1),
            "speedup_vs_1thread": round(speedup, 2),
        }

    return results


def benchmark_session_caching(model_path: str, cache_path: str) -> Dict[str, Any]:
    """Measure session load speedup when using pre-optimized serialized graph."""
    opts_uncached = ort.SessionOptions()
    opts_uncached.intra_op_num_threads = 2
    opts_uncached.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    ort.InferenceSession(model_path, sess_options=opts_uncached, providers=["CPUExecutionProvider"])
    uncached_init_ms = (time.perf_counter() - t0) * 1000.0

    # Save optimized model
    opts_save = ort.SessionOptions()
    opts_save.intra_op_num_threads = 2
    opts_save.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts_save.optimized_model_filepath = cache_path
    ort.InferenceSession(model_path, sess_options=opts_save, providers=["CPUExecutionProvider"])

    # Load from pre-optimized cache
    opts_cached = ort.SessionOptions()
    opts_cached.intra_op_num_threads = 2
    opts_cached.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    t1 = time.perf_counter()
    ort.InferenceSession(cache_path, sess_options=opts_cached, providers=["CPUExecutionProvider"])
    cached_init_ms = (time.perf_counter() - t1) * 1000.0

    speedup = (uncached_init_ms / cached_init_ms) if cached_init_ms > 0 else 1.0
    return {
        "uncached_init_ms": round(uncached_init_ms, 2),
        "cached_init_ms": round(cached_init_ms, 2),
        "cache_speedup": round(speedup, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime intra_op_num_threads and caching.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations per thread count.")
    parser.add_argument("--output", default="eval/onnx_thread_benchmark.json", help="Path to save benchmark JSON report.")
    args = parser.parse_args(argv)

    cpu_count = os.cpu_count() or 2
    recommended_threads = get_optimal_intra_op_threads()
    thread_candidates = sorted(list({1, 2, 4, min(8, cpu_count), cpu_count}))

    print(f"Host CPU count: {cpu_count}")
    print(f"Current default intra_op_num_threads: {recommended_threads}")
    print(f"Benchmarking thread counts: {thread_candidates}...")

    # Check for real specialist models
    ner_path = os.path.join(NER_DIR, "model.onnx")
    sst2_path = os.path.join(SST2_DIR, "model.onnx")

    temp_dir = None
    if os.path.exists(sst2_path):
        target_model = sst2_path
        model_name = "DistilBERT SST-2 (Real Weights)"
    elif os.path.exists(ner_path):
        target_model = ner_path
        model_name = "BERT-base NER (Real Weights)"
    else:
        temp_dir = tempfile.TemporaryDirectory()
        target_model = os.path.join(temp_dir.name, "synthetic_model.onnx")
        print("Real model weights not found locally; generating synthetic transformer workload for benchmark...")
        _create_synthetic_model(target_model)
        model_name = "Synthetic Transformer Workload (4 layers, 512x2048)"

    print(f"\nBenchmarking {model_name}...")
    thread_benchmarks = benchmark_model_threads(target_model, thread_candidates, iterations=args.iterations)

    # Cross-process caching benchmark
    cache_file = os.path.join(temp_dir.name if temp_dir else tempfile.gettempdir(), "test_opt.onnx")
    cache_bench = benchmark_session_caching(target_model, cache_file)

    # Print summary table
    print("\n" + "=" * 78)
    print(f"  ONNX RUNTIME THREAD BENCHMARK RESULTS ({model_name})")
    print("=" * 78)
    print(f"{'Threads':<8} {'Init (ms)':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'QPS':<10} {'Speedup':<10}")
    print("-" * 78)

    best_threads = recommended_threads
    best_latency = float("inf")

    for t_str, data in thread_benchmarks.items():
        t = data["threads"]
        mean_lat = data["mean_latency_ms"]
        if mean_lat < best_latency:
            best_latency = mean_lat
            best_threads = t
        marker = " *" if t == recommended_threads else ""
        print(
            f"{t:<8}{data['init_time_ms']:<12.1f}{data['mean_latency_ms']:<12.2f}"
            f"{data['p95_latency_ms']:<12.2f}{data['throughput_qps']:<10.1f}"
            f"{data['speedup_vs_1thread']:<8.2f}x{marker}"
        )

    print("-" * 78)
    print(f"Cross-process session caching: {cache_bench['uncached_init_ms']} ms -> {cache_bench['cached_init_ms']} ms ({cache_bench['cache_speedup']}x speedup)")
    print(f"Optimal intra_op_num_threads on this host: {best_threads} (QueryLane configured default: {recommended_threads})")
    print("=" * 78)

    report = {
        "host": {
            "cpu_count": cpu_count,
            "configured_default_threads": recommended_threads,
            "best_threads_measured": best_threads,
        },
        "model": model_name,
        "threads_benchmark": thread_benchmarks,
        "caching_benchmark": cache_bench,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.output}")

    if temp_dir:
        temp_dir.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
