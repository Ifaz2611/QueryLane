"""Tests for local calibration, gating, and answer post-processing."""

import threading
import unittest
from unittest import mock

from agent import local


class _FakeLlama:
    def __init__(self, content="Answer", completion_tokens=16):
        self.content = content
        self.completion_tokens = completion_tokens
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {"completion_tokens": self.completion_tokens},
        }


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self.target = target
        self.daemon = daemon
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class LocalTestBase(unittest.TestCase):
    _STATE = (
        "_speed_ok",
        "_time_spent",
        "_tps",
        "_llm",
        "_import_ok",
        "_preload_started",
        "_calibrated",
    )

    def setUp(self):
        self._original = {name: getattr(local, name) for name in self._STATE}
        local.reset_budgets()
        local._speed_ok = None
        local._tps = 0.0
        local._llm = None
        local._import_ok = None
        local._preload_started = False
        local._calibrated = threading.Event()

    def tearDown(self):
        for name, value in self._original.items():
            setattr(local, name, value)


class CalibrationTests(LocalTestBase):
    def test_calibration_enables_local_when_fast_enough(self):
        fake = _FakeLlama(completion_tokens=16)
        # started, gen_start, gen_end, success log, finally
        with mock.patch.object(local, "_get_llm", return_value=fake), mock.patch.object(
            local.time, "time", side_effect=[0.0, 100.0, 110.0, 110.0, 110.0]
        ), mock.patch("builtins.print"):
            local._calibrate()

        self.assertTrue(local._speed_ok)
        self.assertAlmostEqual(local._tps, 1.6)
        self.assertAlmostEqual(local._time_spent, 110.0)
        self.assertTrue(local._calibrated.is_set())
        self.assertEqual(len(fake.calls), 2)  # warmup is not measured

    def test_calibration_disables_local_when_too_slow(self):
        fake = _FakeLlama(completion_tokens=1)
        with mock.patch.object(local, "_get_llm", return_value=fake), mock.patch.object(
            local.time, "time", side_effect=[0.0, 100.0, 110.0, 110.0, 110.0]
        ), mock.patch("builtins.print"):
            local._calibrate()

        self.assertFalse(local._speed_ok)
        self.assertAlmostEqual(local._tps, 0.1)
        self.assertTrue(local._calibrated.is_set())

    def test_calibration_failure_disables_local_and_records_time(self):
        with mock.patch.object(
            local, "_get_llm", side_effect=RuntimeError("no model")
        ), mock.patch.object(local.time, "time", side_effect=[10.0, 12.5]), mock.patch(
            "builtins.print"
        ):
            local._calibrate()

        self.assertFalse(local._speed_ok)
        self.assertAlmostEqual(local._time_spent, 2.5)
        self.assertTrue(local._calibrated.is_set())

    def test_calibration_uses_reported_completion_tokens(self):
        fake = _FakeLlama(completion_tokens=0)
        with mock.patch.object(local, "_get_llm", return_value=fake), mock.patch.object(
            local.time, "time", side_effect=[0.0, 100.0, 110.0, 110.0, 110.0]
        ), mock.patch("builtins.print"):
            local._calibrate()

        # A zero/garbage count falls back to the requested 16 tokens rather
        # than dividing by zero or treating a fast-looking response as real.
        self.assertAlmostEqual(local._tps, 1.6)


class ReadinessTests(LocalTestBase):
    def test_preload_starts_only_one_calibration_thread(self):
        thread = mock.Mock(side_effect=lambda target, daemon: _ImmediateThread(target))
        with mock.patch.object(local.os.path, "exists", return_value=True), mock.patch.object(
            local, "_calibrate"
        ) as calibrate, mock.patch.object(local.threading, "Thread", thread):
            local.preload()
            local.preload()

        self.assertTrue(local._preload_started)
        calibrate.assert_called_once_with()

    def test_preload_is_a_noop_without_weights(self):
        with mock.patch.object(local.os.path, "exists", return_value=False), mock.patch.object(
            local, "_calibrate"
        ) as calibrate, mock.patch.object(local.threading, "Thread") as thread:
            local.preload()

        self.assertFalse(local._preload_started)
        calibrate.assert_not_called()
        thread.assert_not_called()

    def test_wait_ready_returns_false_on_timeout(self):
        with mock.patch.object(local, "preload"):
            self.assertFalse(local.wait_ready(timeout=0.01))

    def test_ready_and_category_gates(self):
        local._speed_ok = True
        local._time_spent = 0.0
        local._tps = 0.5

        self.assertTrue(local.ready())
        self.assertTrue(local.category_enabled("math"))
        self.assertFalse(local.category_enabled("factual"))
        self.assertFalse(local.category_enabled("summarization"))

        local._time_spent = local.TIME_BUDGET
        self.assertFalse(local.ready())
        self.assertFalse(local.category_enabled("math"))

    def test_available_requires_weights_import_speed_and_budget(self):
        with mock.patch.object(local.os.path, "exists", return_value=False):
            self.assertFalse(local.available())

        with mock.patch.object(local.os.path, "exists", return_value=True):
            local._import_ok = False
            self.assertFalse(local.available())

            local._import_ok = True
            local._speed_ok = False
            self.assertFalse(local.available())

            local._speed_ok = True
            local._time_spent = local.TIME_BUDGET
            self.assertFalse(local.available())

            local._time_spent = 0.0
            self.assertTrue(local.available())


class LocalAnswerTests(LocalTestBase):
    def _run_answer(self, content, category="factual", tps=1.0):
        fake = _FakeLlama(content)
        local._llm = fake
        local._tps = tps
        local._speed_ok = True
        with mock.patch.object(local, "ready", return_value=True), mock.patch.object(
            local.time, "time", side_effect=[0.0, 1.0]
        ):
            answer = local.answer(category, "A prompt")
        return answer, fake

    def test_answer_returns_empty_when_not_ready(self):
        with mock.patch.object(local, "ready", return_value=False):
            self.assertEqual(local.answer("factual", "A prompt"), "")

    def test_math_answer_is_evaluated_with_pal(self):
        answer, fake = self._run_answer(
            "EXPR: 2 + 3 * 4 | UNIT: items", category="math"
        )
        self.assertEqual(answer, "Answer: 14 items")
        self.assertAlmostEqual(local._time_spent, 1.0)
        self.assertEqual(len(fake.calls), 1)

    def test_bad_math_expression_escalates(self):
        answer, _ = self._run_answer("EXPR: __import__('os') | UNIT: none", "math")
        self.assertEqual(answer, "")

    def test_generation_cap_scales_with_measured_speed(self):
        _, fake = self._run_answer("Answer", "factual", tps=1.0)
        # factual cap is 200, but 75 seconds at 1 tok/s is the hard ceiling.
        self.assertEqual(fake.calls[0]["max_tokens"], 75)

    def test_answer_passes_system_and_prompt_to_model(self):
        _, fake = self._run_answer("Answer", "sentiment", tps=10.0)
        messages = fake.calls[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "A prompt"})
        self.assertEqual(fake.calls[0]["temperature"], 0)

    def test_answer_normalizes_ner_output(self):
        answer, _ = self._run_answer("Germany: Organization", "ner", tps=10.0)
        self.assertEqual(answer, "Germany: Location")


class AnswerGuardTests(unittest.TestCase):
    def test_empty_answer_is_rejected(self):
        self.assertFalse(local.answer_ok("factual", "   "))

    def test_sentiment_neutral_escalates(self):
        self.assertFalse(local.answer_ok("sentiment", "Neutral, it is a box."))
        self.assertTrue(local.answer_ok("sentiment", "Positive, it works well."))

    def test_ner_requires_typed_lines(self):
        self.assertFalse(local.answer_ok("ner", "Maria"))
        self.assertFalse(local.answer_ok("ner", "Maria: Unknown"))
        self.assertTrue(local.answer_ok("ner", "Maria: Person"))

    def test_ner_completeness_guard_detects_missing_entities(self):
        prompt = "Extract all named entities and their types from: Maria joined Fireworks AI in May 2020."
        complete = "Maria: Person\nFireworks AI: Organization\nMay 2020: Date"
        self.assertTrue(local.ner_complete(prompt, complete))
        self.assertFalse(local.ner_complete(prompt, "Maria: Person\nMay 2020: Date"))

    def test_math_requires_an_answer_line(self):
        self.assertFalse(local.answer_ok("math", "I cannot compute that."))
        self.assertTrue(local.answer_ok("math", "Working...\nAnswer: 42"))

    def test_other_categories_only_require_nonempty_text(self):
        self.assertTrue(local.answer_ok("factual", "Paris is the capital."))


class NormalizeAndSpanTests(unittest.TestCase):
    def test_normalize_rewrites_document_entities(self):
        self.assertEqual(
            local.normalize("ner", "Paris Agreement: Organization"),
            "Paris Agreement: Event",
        )

    def test_normalize_rewrites_bare_countries_only(self):
        self.assertEqual(local.normalize("ner", "Germany: Organization"), "Germany: Location")
        self.assertEqual(
            local.normalize("ner", "Bank of England: Organization"),
            "Bank of England: Organization",
        )

    def test_capitalized_spans_do_not_merge_across_sentences(self):
        spans = local._capitalized_spans("Maria met Bob; Paris is nice")
        self.assertEqual(spans, ["Maria", "Bob", "Paris"])

    def test_capitalized_spans_keep_connectors(self):
        spans = local._capitalized_spans("The Bank of England met in New York")
        self.assertEqual(spans, ["Bank of England", "New York"])


class CategoryBudgetAndProofTests(LocalTestBase):
    def test_single_category_exhaustion_does_not_disable_other_categories(self):
        """A slow category that exhausts its fallback budget must not lock out others."""
        local._speed_ok = True
        local._tps = 2.0  # speed clears bars for math, ner, factual
        local._time_spent = 50.0  # well below global TIME_BUDGET (300s)

        self.assertTrue(local.category_enabled("factual"))
        self.assertTrue(local.category_enabled("math"))
        self.assertTrue(local.category_enabled("ner"))

        # Consume factual's budget
        budget = local.category_budget("factual")
        with local._budget_lock:
            local._category_time_spent["factual"] = budget

        # Factual is now disabled due to fallback budget exhaustion
        self.assertFalse(local.category_enabled("factual"))
        # But math and ner remain enabled with plenty of budget!
        self.assertTrue(local.category_enabled("math"))
        self.assertTrue(local.category_enabled("ner"))

    def test_global_budget_exhaustion_disables_all(self):
        """When global TIME_BUDGET is reached, all categories are disabled."""
        local._speed_ok = True
        local._tps = 2.0
        local._time_spent = local.TIME_BUDGET

        self.assertFalse(local.ready())
        self.assertFalse(local.category_enabled("math"))
        self.assertFalse(local.category_enabled("factual"))

    def test_category_budget_custom_env_override(self):
        """Category budget can be configured through environment variables."""
        with mock.patch.dict(local.os.environ, {"LOCAL_CATEGORY_BUDGET_FACTUAL": "45.5"}):
            self.assertEqual(local.category_budget("factual"), 45.5)

    def test_category_time_tracking_in_answer(self):
        """answer() records elapsed time both globally and per category."""
        fake = _FakeLlama("Answer: 42")
        local._llm = fake
        local._tps = 1.0
        local._speed_ok = True

        with mock.patch.object(local, "ready", return_value=True), mock.patch.object(
            local.time, "time", side_effect=[10.0, 15.0]
        ):
            local.answer("factual", "What is the answer?")

        self.assertEqual(local.category_time_spent("factual"), 5.0)
        self.assertEqual(local._time_spent, 5.0)
        self.assertEqual(local.category_time_spent("math"), 0.0)

    def test_answer_returns_empty_when_category_budget_exhausted(self):
        """answer() returns empty string if category fallback budget is exhausted."""
        fake = _FakeLlama("Answer: 42")
        local._llm = fake
        local._tps = 1.0
        local._speed_ok = True

        with local._budget_lock:
            local._category_time_spent["factual"] = local.category_budget("factual") + 1.0

        with mock.patch.object(local, "ready", return_value=True):
            ans = local.answer("factual", "What is the answer?")
        self.assertEqual(ans, "")

    def test_unproven_summarization_and_logic_disabled_by_default(self):
        """Summarization and logic must remain disabled until proven by fine-tune."""
        local._speed_ok = True
        local._tps = 100.0  # Even at blistering speeds

        self.assertFalse(local.category_proven("summarization"))
        self.assertFalse(local.category_proven("logic"))
        self.assertFalse(local.category_enabled("summarization"))
        self.assertFalse(local.category_enabled("logic"))

        # answer() returns empty when unproven
        fake = _FakeLlama("A summary")
        local._llm = fake
        with mock.patch.object(local, "ready", return_value=True):
            self.assertEqual(local.answer("summarization", "Some text"), "")
            self.assertEqual(local.answer("logic", "A puzzle"), "")

    def test_proven_category_can_be_enabled(self):
        """When proven via environment flag, category_proven returns True."""
        with mock.patch.dict(local.os.environ, {"PROVE_LOCAL_SUMMARIZATION": "1"}):
            self.assertTrue(local.category_proven("summarization"))
            local._speed_ok = True
            local._tps = 100.0
            self.assertTrue(local.category_enabled("summarization"))


if __name__ == "__main__":
    unittest.main()
