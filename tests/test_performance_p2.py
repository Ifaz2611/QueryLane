"""Unit tests for P2 performance features: ONNX thread tuning, context checks, and batch measurement."""

import os
from unittest.mock import patch

from agent import local as local_model
from agent import specialists
from agent.main import predict_lane
from eval.measure_batching import simulate_category_batch


def test_optimal_intra_op_threads():
    # Test explicit override
    with patch.dict(os.environ, {"ONNX_INTRA_OP_THREADS": "3"}):
        assert specialists.get_optimal_intra_op_threads() == 3

    # Test invalid override falls back
    with patch.dict(os.environ, {"ONNX_INTRA_OP_THREADS": "invalid"}):
        with patch("os.cpu_count", return_value=8):
            # Bounded to 4 max by default
            assert specialists.get_optimal_intra_op_threads() == 4

    with patch.dict(os.environ, {}, clear=True):
        with patch("os.cpu_count", return_value=2):
            assert specialists.get_optimal_intra_op_threads() == 2


def test_context_length_estimates():
    short_text = "What is 2 + 2?"
    long_text = "word " * 2000  # ~2000 words, ~10000 chars

    assert local_model.estimate_tokens(short_text) < 15
    assert local_model.estimate_tokens(long_text) > 1500

    assert not local_model.is_prompt_too_long(short_text, "factual")
    assert local_model.is_prompt_too_long(long_text, "factual")


def test_predict_lane_rejects_oversized_prompt():
    huge_prompt = "Calculate the following sequence: " + ("123 + 456 - 789 * " * 600)
    # Despite math and local availability mock, context overflow forces cloud
    with patch.object(local_model, "available", return_value=True), patch.object(
        local_model, "category_enabled", return_value=True
    ):
        lane = predict_lane("math", "local", 0.95, huge_prompt)
        assert lane == "cloud"


def test_local_answer_rejects_oversized_prompt():
    huge_prompt = "A " * 6000
    with patch.object(local_model, "ready", return_value=True), patch.object(
        local_model, "category_proven", return_value=True
    ):
        ans = local_model.answer("factual", huge_prompt)
        assert ans == ""  # Immediately returns empty to escalate without wasting local budget


def test_measure_batching_simulation():
    sample_tasks = [
        {"task_id": "1", "prompt": "What is the capital of France?"},
        {"task_id": "2", "prompt": "What is the capital of Japan?"},
    ]
    res = simulate_category_batch("factual", sample_tasks)
    assert res["group_size"] == 2
    # At N=2, batched tokens exceed unbatched tokens due to delimiters & batch instruction
    assert res["net_token_delta"] > 0
    assert not res["is_beneficial"]
