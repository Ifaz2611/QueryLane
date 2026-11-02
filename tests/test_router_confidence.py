"""Tests for the router confidence score and low-confidence cloud routing."""

import unittest
from unittest import mock

from agent import router
from agent.router import (
    ALL_CATEGORIES,
    classify,
    classify_with_confidence,
    route,
    route_with_confidence,
)


SENTIMENT_PROMPT = (
    "Classify the sentiment of this review: The battery is great and the screen is sharp."
)


class ConfidenceTests(unittest.TestCase):
    def test_confident_prompt_scores_above_threshold(self):
        category, confidence = classify_with_confidence(SENTIMENT_PROMPT)
        self.assertEqual(category, "sentiment")
        self.assertGreaterEqual(confidence, router.MIN_CONFIDENCE)
        self.assertLessEqual(confidence, 1.0)

    def test_unknown_prompt_has_low_confidence_factual_guess(self):
        category, confidence = classify_with_confidence(
            "Tell me something interesting."
        )
        self.assertEqual(category, "factual")
        self.assertLess(confidence, router.MIN_CONFIDENCE)

    def test_every_category_can_report_a_confidence(self):
        prompts = {
            "factual": "What is the capital of Japan?",
            "math": "A train travels 180 km in 2.5 hours. What is its average speed?",
            "sentiment": SENTIMENT_PROMPT,
            "summarization": "Summarize this paragraph in one sentence: The council met.",
            "ner": "Extract all named entities and their types from: Maria joined Acme.",
            "code_debug": "Fix the bug in this function: def f(x): return x +",
            "logic": "Anna is taller than Ben. Who is the tallest?",
            "code_gen": "Write a Python function that reverses a list.",
        }
        self.assertEqual(set(prompts), set(ALL_CATEGORIES))
        for expected, prompt in prompts.items():
            with self.subTest(category=expected):
                category, confidence = classify_with_confidence(prompt)
                self.assertEqual(category, expected)
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)


class RouteWithConfidenceTests(unittest.TestCase):
    def test_low_confidence_local_category_is_forced_to_cloud(self):
        with mock.patch.object(router, "LOCAL_CATEGORIES", {"sentiment"}), (
            mock.patch.object(router, "MIN_CONFIDENCE", 0.999)
        ):
            category, target, confidence = route_with_confidence(SENTIMENT_PROMPT)
        self.assertEqual(category, "sentiment")
        self.assertEqual(target, "cloud")
        self.assertLess(confidence, 0.999)

    def test_confident_local_category_keeps_local_target(self):
        with mock.patch.object(router, "LOCAL_CATEGORIES", {"sentiment"}), (
            mock.patch.object(router, "MIN_CONFIDENCE", 0.10)
        ):
            category, target, _ = route_with_confidence(SENTIMENT_PROMPT)
        self.assertEqual(category, "sentiment")
        self.assertEqual(target, "local")

    def test_route_stays_backward_compatible(self):
        self.assertEqual(len(route(SENTIMENT_PROMPT)), 2)
        self.assertEqual(route(SENTIMENT_PROMPT)[0], classify(SENTIMENT_PROMPT))
        self.assertEqual(len(route(SENTIMENT_PROMPT, with_confidence=True)), 3)


if __name__ == "__main__":
    unittest.main()
