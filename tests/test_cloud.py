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

    def test_call_treats_none_choices_and_bad_content_as_empty(self):
        self.client._client = mock.Mock()
        cases = [
            SimpleNamespace(choices=None, usage=None),
            SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None),
            # A list-of-dicts content part (newer SDK shape) is not plain text.
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=[{"type": "text"}]))],
                usage=None,
            ),
            "not a response object",
        ]
        for response in cases:
            with self.subTest(response=response):
                self.client._client.chat.completions.create.return_value = response
                self.assertEqual(self.client._call("model-a", "system", "user", 10), "")

    def test_usage_totals_normalizes_malformed_values(self):
        self.assertEqual(cloud.usage_totals(None), {"prompt": 0, "completion": 0, "total": 0})
        self.assertEqual(
            cloud.usage_totals({"prompt_tokens": 3, "completion_tokens": 4}),
            {"prompt": 3, "completion": 4, "total": 7},
        )
        self.assertEqual(
            cloud.usage_totals(
                SimpleNamespace(prompt_tokens="9", completion_tokens=-2, total_tokens=True)
            ),
            {"prompt": 0, "completion": 0, "total": 0},
        )
        self.assertEqual(
            cloud.usage_totals(SimpleNamespace(total_tokens=12))["total"], 12
        )

    def test_response_text_handles_empty_and_malformed_payloads(self):
        self.assertEqual(cloud.response_text(_response("  hi  ")), "hi")
        self.assertEqual(cloud.response_text(_response(choices=False)), "")
        self.assertEqual(cloud.response_text(None), "")

    def test_answer_stops_when_the_per_task_budget_is_spent(self):
        self.client.task_budget = 0.01
        with mock.patch.object(
            self.client, "_call_with_retries", return_value=("", None)
        ) as call, mock.patch.object(cloud.time, "sleep"), mock.patch.object(
            cloud.time, "time", side_effect=[0.0, 100.0, 100.0, 100.0]
        ):
            self.assertEqual(self.client.answer("factual", "Why?"), "")
        self.assertEqual(call.call_count, 0)  # stopped before any paid call
        self.assertEqual(self.client.budget_exceeded_count(), 1)


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
        sleep.assert_called_once()
        # Equal-jitter backoff: at least half of base*2^0 and never more.
        delay = sleep.call_args.args[0]
        self.assertGreaterEqual(delay, 0.25)
        self.assertLessEqual(delay, 0.5)
        self.assertEqual(self.client.retry_count(), 1)

    def test_server_errors_back_off_between_retries(self):
        with _patched_openai_errors() as (_, _, _, status):
            self.client._call = mock.Mock(side_effect=status(503))
            with mock.patch.object(cloud.time, "sleep") as sleep:
                text, error = self.client._call_with_retries(
                    "model-a", "system", "user", 10, "none"
                )
        self.assertEqual(text, "")
        self.assertIsNone(error)
        self.assertEqual(self.client._call.call_count, 3)
        delays = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(len(delays), 2)  # no sleep after the final attempt
        self.assertLess(delays[0], delays[1])  # exponential growth
        self.assertGreaterEqual(delays[0], 0.25)
        self.assertLessEqual(delays[1], 1.0)
        self.assertEqual(self.client.retry_count(), 2)

    def test_retry_sleep_is_clipped_to_the_remaining_budget(self):
        with _patched_openai_errors() as (conn, _, _, _):
            self.client._call = mock.Mock(side_effect=[conn("down"), "ok"])
            with mock.patch.object(cloud.time, "sleep") as sleep:
                text, _ = self.client._call_with_retries(
                    "model-a",
                    "system",
                    "user",
                    10,
                    "none",
                    deadline=cloud.time.time() + 0.05,
                )
        self.assertEqual(text, "ok")
        self.assertLessEqual(sleep.call_args.args[0], 0.05)

    def test_no_further_attempt_is_made_once_the_budget_is_spent(self):
        with _patched_openai_errors() as (conn, _, _, _):
            self.client._call = mock.Mock(side_effect=conn("down"))
            with mock.patch.object(cloud.time, "sleep"):
                text, error = self.client._call_with_retries(
                    "model-a",
                    "system",
                    "user",
                    10,
                    "none",
                    deadline=cloud.time.time() - 1,
                )
        self.assertEqual((text, error), ("", None))
        self.assertEqual(self.client._call.call_count, 1)
        self.assertEqual(self.client.retry_count(), 0)

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

    def test_answer_with_spec_returns_empty_when_every_model_is_empty(self):
        spec = SimpleNamespace(system="system", max_tokens=10, effort="none")
        with mock.patch.object(
            self.client, "_models_for", return_value=["model-a", "model-b"]
        ), mock.patch.object(
            self.client, "_call_with_retries", return_value=("", None)
        ):
            self.assertEqual(
                self.client._answer_with_spec(spec, "factual", "prompt"), ""
            )


if __name__ == "__main__":
    unittest.main()
