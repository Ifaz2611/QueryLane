"""Tests for the Fireworks cloud client (agent/cloud.py)."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import cloud, prompts


def _response(content="Answer", total_tokens=7, choices=True, usage=True):
    message = SimpleNamespace(content=content)
    usage_obj = None
    if usage:
        usage_obj = SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=3,
            completion_tokens=4,
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)] if choices else [],
        usage=usage_obj,
    )


@contextlib.contextmanager
def _patched_openai_errors():
    """Replace the SDK exception classes with tiny stand-ins.

    The retry loop catches module-level names, so patching those names lets the
    tests exercise every branch without constructing real SDK exceptions.
    """

    class ConnectionError_(Exception):
        pass

    class TimeoutError_(Exception):
        pass

    class RateLimitError_(Exception):
        pass

    class APIStatusError_(Exception):
        def __init__(self, status_code):
            super().__init__(str(status_code))
            self.status_code = status_code

    with mock.patch.object(cloud, "APIConnectionError", ConnectionError_), mock.patch.object(
        cloud, "APITimeoutError", TimeoutError_
    ), mock.patch.object(cloud, "RateLimitError", RateLimitError_), mock.patch.object(
        cloud, "APIStatusError", APIStatusError_
    ):
        yield ConnectionError_, TimeoutError_, RateLimitError_, APIStatusError_


class CloudClientConfigTests(unittest.TestCase):
    def test_from_env_deduplicates_and_strips_models(self):
        env = {
            "FIREWORKS_API_KEY": "test-key",
            "FIREWORKS_BASE_URL": "https://example.invalid/v1",
            "ALLOWED_MODELS": " model-a , model-b,model-a ",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            client = cloud.CloudClient.from_env()
        self.assertEqual(client._allowed, ["model-a", "model-b"])

    def test_from_env_rejects_empty_model_list(self):
        env = {
            "FIREWORKS_API_KEY": "test-key",
            "FIREWORKS_BASE_URL": "https://example.invalid/v1",
            "ALLOWED_MODELS": " , ",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                cloud.CloudClient.from_env()

    def test_from_env_requires_all_variables(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                cloud.CloudClient.from_env()

    def test_models_for_puts_preferred_models_first(self):
        client = cloud.CloudClient(
            "test-key",
            "https://example.invalid/v1",
            [
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/kimi-k2p7-code",
                "accounts/fireworks/models/other",
            ],
        )
        self.assertEqual(
            client._models_for("factual"),
            [
                "accounts/fireworks/models/kimi-k2p7-code",
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/other",
            ],
        )

    def test_models_for_honors_category_preference(self):
        client = cloud.CloudClient(
            "test-key",
            "https://example.invalid/v1",
            ["model-minimax", "model-kimi"],
        )
        with mock.patch.dict(prompts.MODEL_PREFERENCE, {"math": ["minimax"]}):
            self.assertEqual(client._models_for("math"), ["model-minimax", "model-kimi"])


class CloudResponseTests(unittest.TestCase):
    def setUp(self):
        self.client = cloud.CloudClient(
            "test-key", "https://example.invalid/v1", ["model-a"]
        )

    def test_strip_thinking_removes_complete_and_partial_blocks(self):
        self.assertEqual(
            cloud._strip_thinking("<think>internal</think> external "),
            "external",
        )
        self.assertEqual(
            cloud._strip_thinking("internal</think> external "),
            "external",
        )
        self.assertEqual(cloud._strip_thinking("  plain answer  "), "plain answer")

    def test_call_counts_usage_and_strips_thinking(self):
        self.client._client = mock.Mock()
        self.client._client.chat.completions.create.return_value = _response(
            "  <think>x</think>hello ", total_tokens=11
        )
        self.assertEqual(self.client._call("model-a", "system", "user", 10), "hello")
        self.assertEqual(self.client.total_tokens(), 11)
        self.assertEqual(self.client.call_count(), 1)

    def test_call_tolerates_empty_choices(self):
        self.client._client = mock.Mock()
        self.client._client.chat.completions.create.return_value = _response(
            choices=False, total_tokens=4
        )
        self.assertEqual(self.client._call("model-a", "system", "user", 10), "")
        self.assertEqual(self.client.total_tokens(), 4)

    def test_call_tolerates_malformed_usage(self):
        self.client._client = mock.Mock()
        response = _response(usage=False)
        response.usage = SimpleNamespace()  # missing every counter
        self.client._client.chat.completions.create.return_value = response
        self.assertEqual(self.client._call("model-a", "system", "user", 10), "Answer")
        self.assertEqual(self.client.total_tokens(), 0)

    def test_call_sends_reasoning_effort_and_temperature_zero(self):
        self.client._client = mock.Mock()
        self.client._client.chat.completions.create.return_value = _response()
        self.client._call("model-a", "system", "user", 10, "none")
        kwargs = self.client._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["max_tokens"], 10)
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "none"})


class CloudRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = cloud.CloudClient(
            "test-key", "https://example.invalid/v1", ["model-a"]
        )

    def test_transient_errors_are_retried(self):
        with _patched_openai_errors() as (conn, timeout, _, _):
            self.client._call = mock.Mock(
                side_effect=[conn("down"), timeout("slow"), "ok"]
            )
            text, error = self.client._call_with_retries(
                "model-a", "system", "user", 10, "none"
            )
        self.assertEqual(text, "ok")
        self.assertIsNone(error)
        self.assertEqual(self.client._call.call_count, 3)

    def test_rate_limit_sleeps_before_retrying(self):
        with _patched_openai_errors() as (_, _, rate_limit, _):
            self.client._call = mock.Mock(side_effect=[rate_limit("429"), "ok"])
            with mock.patch.object(cloud.time, "sleep") as sleep:
                text, error = self.client._call_with_retries(
                    "model-a", "system", "user", 10, "none"
                )
        self.assertEqual(text, "ok")
        self.assertIsNone(error)
        sleep.assert_called_once_with(2)

    def test_client_errors_are_model_level_and_not_retried(self):
        with _patched_openai_errors() as (_, _, _, status):
            error = status(400)
            self.client._call = mock.Mock(side_effect=error)
            text, returned = self.client._call_with_retries(
                "model-a", "system", "user", 10, "none"
            )
        self.assertEqual(text, "")
        self.assertIs(returned, error)
        self.assertEqual(self.client._call.call_count, 1)

    def test_server_errors_exhaust_retries_and_return_no_error(self):
        with _patched_openai_errors() as (_, _, _, status):
            self.client._call = mock.Mock(side_effect=status(500))
            text, error = self.client._call_with_retries(
                "model-a", "system", "user", 10, "none"
            )
        self.assertEqual(text, "")
        self.assertIsNone(error)
        self.assertEqual(self.client._call.call_count, 3)

    def test_connection_errors_exhaust_retries(self):
        with _patched_openai_errors() as (conn, _, _, _):
            self.client._call = mock.Mock(side_effect=conn("down"))
            text, error = self.client._call_with_retries(
                "model-a", "system", "user", 10, "none"
            )
        self.assertEqual(text, "")
        self.assertIsNone(error)
        self.assertEqual(self.client._call.call_count, 3)


class CloudAnswerTests(unittest.TestCase):
    def setUp(self):
        self.client = cloud.CloudClient(
            "test-key", "https://example.invalid/v1", ["model-a", "model-b"]
        )

    def test_answer_uses_category_spec(self):
        with mock.patch.object(
            self.client, "_answer_with_spec", return_value="cloud answer"
        ) as call:
            self.assertEqual(self.client.answer("factual", "What is a parsec?"), "cloud answer")
        self.assertIs(call.call_args.args[0], prompts.SPECS["factual"])

    def test_answer_strips_known_instruction_wrapper(self):
        with mock.patch.object(
            self.client, "_answer_with_spec", return_value="Positive."
        ) as call:
            self.client.answer(
                "sentiment",
                "Classify the sentiment of this review: The battery is great.",
            )
        self.assertEqual(call.call_args.args[2], "The battery is great.")

    def test_math_uses_pal_when_expression_parses(self):
        with mock.patch.object(
            self.client,
            "_answer_with_spec",
            return_value="EXPR: 240 * (1 - 0.15) - 60 | UNIT: items",
        ) as call:
            self.assertEqual(
                self.client.answer("math", "How many items remain?"),
                "Answer: 144 items",
            )
        self.assertEqual(call.call_count, 1)

    def test_math_falls_back_to_step_by_step_on_bad_expression(self):
        with mock.patch.object(
            self.client,
            "_answer_with_spec",
            side_effect=["not an expression", "Answer: 42"],
        ) as call:
            self.assertEqual(self.client.answer("math", "What is 6 * 7?"), "Answer: 42")
        self.assertEqual(call.call_count, 2)
        self.assertIs(call.call_args_list[1].args[0], prompts.MATH_FALLBACK)

    def test_answer_with_spec_retries_without_reasoning_effort_on_400(self):
        spec = SimpleNamespace(system="system", max_tokens=10, effort="none")
        with _patched_openai_errors() as (_, _, _, status):
            with mock.patch.object(
                self.client,
                "_models_for",
                return_value=["model-a"],
            ), mock.patch.object(
                self.client,
                "_call_with_retries",
                side_effect=[("", status(400)), ("recovered", None)],
            ) as call:
                self.assertEqual(
                    self.client._answer_with_spec(spec, "factual", "prompt"),
                    "recovered",
                )
        self.assertEqual(call.call_args_list[0].args[4], "none")
        self.assertIsNone(call.call_args_list[1].args[4])

    def test_answer_with_spec_falls_back_to_next_model(self):
        spec = SimpleNamespace(system="system", max_tokens=10, effort="none")
        with _patched_openai_errors() as (_, _, _, status):
            with mock.patch.object(
                self.client,
                "_models_for",
                return_value=["model-a", "model-b"],
            ), mock.patch.object(
                self.client,
                "_call_with_retries",
                side_effect=[("", status(404)), ("second model", None)],
            ):
                self.assertEqual(
                    self.client._answer_with_spec(spec, "factual", "prompt"),
                    "second model",
                )

    def test_answer_with_spec_raises_when_every_model_fails(self):
        spec = SimpleNamespace(system="system", max_tokens=10, effort="none")
        with _patched_openai_errors() as (_, _, _, status):
            with mock.patch.object(
                self.client, "_models_for", return_value=["model-a", "model-b"]
            ), mock.patch.object(
                self.client,
                "_call_with_retries",
                side_effect=[("", status(404)), ("", status(404))],
            ):
                with self.assertRaises(RuntimeError):
                    self.client._answer_with_spec(spec, "factual", "prompt")


if __name__ == "__main__":
    unittest.main()
