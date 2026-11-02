"""Tests for the per-category evaluation dashboard."""

import json
import os
import tempfile
import unittest

from eval.dashboard import (
    build_dashboard,
    deterministic_grade,
)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.sample_tasks = [
            {"task_id": "math-01", "category": "math", "prompt": "240 items, sells 15% and 60. How many remain?"},
            {"task_id": "math-02", "category": "math", "prompt": "Speed calculation"},
            {"task_id": "sent-01", "category": "sentiment", "prompt": "Review text"},
            {"task_id": "fact-01", "category": "factual", "prompt": "Capital of Japan"},
        ]
        self.sample_answers = [
            {"task_id": "math-01", "category": "math", "answer": "Answer: 144 items", "routed": "local", "tokens": 0},
            {"task_id": "math-02", "category": "math", "answer": "Answer: 72 km/h", "routed": "cloud", "tokens": 120},
            {"task_id": "sent-01", "category": "sentiment", "answer": "Positive, great item", "routed": "specialist", "tokens": 0},
            {"task_id": "fact-01", "category": "factual", "answer": "Tokyo on Honshu island", "routed": "cloud", "tokens": 90},
        ]

    def test_build_dashboard_metrics(self):
        graded_verdicts = {
            "math-01": True,
            "math-02": True,
            "sent-01": True,
            "fact-01": False,
        }
        report = build_dashboard(self.sample_tasks, self.sample_answers, graded_verdicts=graded_verdicts)

        self.assertEqual(report.total_tasks, 4)
        self.assertEqual(report.total_passed, 3)
        self.assertEqual(report.total_failed, 1)
        self.assertAlmostEqual(report.overall_accuracy, 75.0)
        self.assertFalse(report.overall_gate_passed)  # 75% < 80%
        self.assertEqual(report.total_tokens, 210)  # 120 + 90
        self.assertEqual(report.avg_tokens_per_task, 52.5)  # 210 / 4

        # Math category checks
        math_m = report.categories["math"]
        self.assertEqual(math_m.total_tasks, 2)
        self.assertEqual(math_m.passed_tasks, 2)
        self.assertEqual(math_m.failed_tasks, 0)
        self.assertEqual(math_m.accuracy, 100.0)
        self.assertTrue(math_m.gate_passed)
        self.assertEqual(math_m.tokens, 120)
        self.assertEqual(math_m.routed_lanes, {"local": 1, "cloud": 1})

        # Factual category checks
        fact_m = report.categories["factual"]
        self.assertEqual(fact_m.total_tasks, 1)
        self.assertEqual(fact_m.passed_tasks, 0)
        self.assertEqual(fact_m.failed_tasks, 1)
        self.assertEqual(fact_m.accuracy, 0.0)
        self.assertFalse(fact_m.gate_passed)
        self.assertEqual(fact_m.tokens, 90)

    def test_table_formatting(self):
        report = build_dashboard(self.sample_tasks, self.sample_answers)
        table_output = report.format_table()
        self.assertIn("QUERYLANE EVALUATION DASHBOARD", table_output)
        self.assertIn("math", table_output)
        self.assertIn("sentiment", table_output)
        self.assertIn("factual", table_output)
        self.assertIn("TOTAL / OVERALL", table_output)

    def test_markdown_formatting(self):
        report = build_dashboard(self.sample_tasks, self.sample_answers)
        md_output = report.format_markdown()
        self.assertIn("# QueryLane Evaluation Dashboard", md_output)
        self.assertIn("| Category | Tasks | Passed |", md_output)
        self.assertIn("**math**", md_output)
        self.assertIn("## Lane Distribution", md_output)

    def test_save_markdown_and_json(self):
        report = build_dashboard(self.sample_tasks, self.sample_answers)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "dash.md")
            json_path = os.path.join(tmpdir, "dash.json")

            report.save_markdown(md_path)
            report.save_json(json_path)

            self.assertTrue(os.path.exists(md_path))
            self.assertTrue(os.path.exists(json_path))

            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("summary", data)
            self.assertIn("categories", data)
            self.assertEqual(data["summary"]["total_tasks"], 4)

    def test_deterministic_grade_fallback(self):
        task_math = {"task_id": "math-01", "category": "math"}
        ok, _ = deterministic_grade(task_math, "Answer: 144 items")
        self.assertTrue(ok)

        task_fact = {"task_id": "fact-01", "category": "factual"}
        ok, _ = deterministic_grade(task_fact, "The capital is Tokyo on Honshu island.")
        self.assertTrue(ok)
        ok_fail, _ = deterministic_grade(task_fact, "Kyoto")
        self.assertFalse(ok_fail)


if __name__ == "__main__":
    unittest.main()
