"""Fast, dependency-light regression tests for QueryLane's routing core.

Run with:
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import agent.main as main_module
from agent import local, pal, specialists
from agent.cloud import strip_instruction
from agent.router import ALL_CATEGORIES, classify
from finetune import gen_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RouterTests(unittest.TestCase):
    def test_dev_set_routes_to_expected_categories(self):
        with open(os.path.join(ROOT, "eval", "tasks_dev.json"), encoding="utf-8") as fh:
            tasks = json.load(fh)
        for task in tasks:
            with self.subTest(task_id=task["task_id"]):
                self.assertEqual(classify(task["prompt"]), task["category"])
                self.assertIn(task["category"], ALL_CATEGORIES)

    def test_unknown_prompt_falls_back_to_factual(self):
        self.assertEqual(classify("Tell me something interesting."), "factual")


class PalTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(pal.safe_eval("2 + 3 * 4"), 14)
        self.assertAlmostEqual(pal.safe_eval("2 ** -3"), 0.125)
        self.assertEqual(pal.safe_eval("abs(-5) + max(1, 2)"), 7)

    def test_answer_formatting(self):
        cases = {
            "EXPR: 240 * (1 - 0.15) - 60 | UNIT: items": "Answer: 144 items",
            "EXPR: 180 / 2.5 | UNIT: km/h": "Answer: 72 km/h",
            "EXPR: 40 / (1 - 0.20) | UNIT: $": "Answer: $50",
            "EXPR: 2 ** 10 | UNIT: none": "Answer: 1024",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pal.answer_from_expression(text), expected)

    def test_rejects_unsafe_or_non_numeric(self):
        for expression in (
            "1e309",          # non-finite constant
            "10 ** 100000",   # exponent bomb
            "(-1) ** 0.5",    # complex result
            "'a' * 3",        # string constant
            "__import__('os')",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    pal.safe_eval(expression)

    def test_bad_expression_returns_none(self):
        self.assertIsNone(pal.answer_from_expression("EXPR: 1e309 | UNIT: none"))
        self.assertIsNone(pal.answer_from_expression("no expression here"))


class InstructionStrippingTests(unittest.TestCase):
    def test_sentiment_prefix(self):
        self.assertEqual(
            strip_instruction(
                "sentiment",
                "Classify the sentiment of this review: The battery is great.",
            ),
            "The battery is great.",
        )

    def test_ner_prefix(self):
        self.assertEqual(
            strip_instruction(
                "ner",
                "Extract all named entities and their types from: Maria joined Fireworks AI.",
            ),
            "Maria joined Fireworks AI.",
        )

    def test_unusual_prompt_is_untouched(self):
        prompt = "What is the sentiment of the market?"
        self.assertEqual(strip_instruction("sentiment", prompt), prompt)


class NormalizeTests(unittest.TestCase):
    def test_document_entity_is_event(self):
        self.assertEqual(
            local.normalize("ner", "Paris Agreement: Organization"),
            "Paris Agreement: Event",
        )

    def test_country_entity_is_location(self):
        self.assertEqual(local.normalize("ner", "Germany: Organization"), "Germany: Location")

    def test_other_categories_untouched(self):
        self.assertEqual(
            local.normalize("factual", "Germany: Organization"),
            "Germany: Organization",
        )


class SpecialistCueTests(unittest.TestCase):
    def test_positive_cues_do_not_match_negated_words(self):
        for text in ("The bed was uncomfortable.", "The result was imperfect.",
                     "The glove fits.", "The service was unfriendly."):
            with self.subTest(text=text):
                self.assertIsNone(specialists._POS_CUE_RE.search(text))

    def test_positive_cues_match(self):
        self.assertIsNotNone(specialists._POS_CUE_RE.search("The bed was comfortable."))

    def test_negative_cues_do_not_match_positive_context(self):
        for text in ("It crushes ice in seconds.", "The latest model is great."):
            with self.subTest(text=text):
                self.assertIsNone(specialists._NEG_CUE_RE.search(text))

    def test_negative_cues_match(self):
        for text in ("The box was completely crushed.", "The package arrived late."):
            with self.subTest(text=text):
                self.assertIsNotNone(specialists._NEG_CUE_RE.search(text))

    def test_date_regex_keeps_full_month_year(self):
        matches = [m.group(0) for m in specialists._DATE_RE.finditer("in May 2020")]
        self.assertEqual(matches, ["May 2020"])
        self.assertEqual(
            [m.group(0) for m in specialists._DATE_RE.finditer("maybe later")],
            [],
        )


class DataValidationTests(unittest.TestCase):
    def test_neutral_rows_with_quotes_are_rejected(self):
        rows = gen_data._to_rows(
            "sentiment",
            [{
                "review": "Great product.",
                "label": "neutral",
                "praise_quote": "Great",
                "complaint_quote": "",
            }],
        )
        self.assertEqual(rows, [])

    def test_valid_neutral_row_is_kept(self):
        rows = gen_data._to_rows(
            "sentiment",
            [{
                "review": "The box contains one charger.",
                "label": "neutral",
                "praise_quote": "",
                "complaint_quote": "",
            }],
        )
        self.assertEqual(len(rows), 1)

    def test_ner_rows_with_empty_entities_are_rejected(self):
        rows = gen_data._to_rows(
            "ner",
            [{
                "sentence": "Maria joined Fireworks AI.",
                "entities": [{"text": "", "type": "Person"}],
            }],
        )
        self.assertEqual(rows, [])

    def test_train_all_contains_every_category(self):
        data_dir = os.path.join(ROOT, "finetune", "data")
        with open(os.path.join(data_dir, "train_all.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        categories = {row["category"] for row in rows}
        self.assertEqual(
            categories,
            {"sentiment", "ner", "summarization", "factual", "math"},
        )
        expected = 0
        for category in categories:
            with open(os.path.join(data_dir, f"{category}.jsonl"), encoding="utf-8") as fh:
                expected += sum(1 for line in fh if line.strip())
        self.assertEqual(len(rows), expected)


class MainLoopTests(unittest.TestCase):
    class FakeCloud:
        @classmethod
        def from_env(cls):
            return cls()

        def answer(self, category, prompt):
            return f"Answer: {category}"

        def call_count(self):
            return 1

        def total_tokens(self):
            return 7

    def test_main_writes_every_task_and_accepts_bare_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [
                        {"task_id": "a", "prompt": "What is 2 + 2?"},
                        {"task_id": "b", "prompt": "Write a Python function to add."},
                    ],
                    fh,
                )
            env = {"TASKS_PATH": tasks_path, "RESULTS_PATH": "results_bare.json"}
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with mock.patch.dict(os.environ, env), mock.patch.object(
                    main_module, "CloudClient", self.FakeCloud
                ):
                    self.assertEqual(main_module.main(), 0)
            finally:
                os.chdir(old_cwd)
            with open(os.path.join(tmp, "results_bare.json"), encoding="utf-8") as fh:
                results = json.load(fh)
            self.assertEqual([r["task_id"] for r in results], ["a", "b"])
            self.assertTrue(all(r["answer"] for r in results))

    def test_main_rejects_non_array_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump({"task_id": "a"}, fh)
            with mock.patch.dict(os.environ, {"TASKS_PATH": tasks_path}):
                self.assertEqual(main_module.main(), 1)


class JsonAssetTests(unittest.TestCase):
    def test_all_json_assets_parse(self):
        paths = [
            os.path.join(ROOT, "eval", "tasks_dev.json"),
            os.path.join(ROOT, "testio", "input", "tasks.json"),
            os.path.join(ROOT, "testio", "input_dev", "tasks.json"),
            os.path.join(ROOT, "testio", "input_local", "tasks.json"),
            os.path.join(ROOT, "testio", "input_stress", "tasks.json"),
            os.path.join(ROOT, "models", "ner", "config.json"),
            os.path.join(ROOT, "models", "sst2", "config.json"),
        ]
        for path in paths:
            with self.subTest(path=path):
                with open(path, encoding="utf-8") as fh:
                    json.load(fh)


if __name__ == "__main__":
    unittest.main()
