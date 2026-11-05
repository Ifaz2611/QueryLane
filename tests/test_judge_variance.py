"""Tests for the multi-run judge variance report (no API calls)."""

import unittest

from eval.judge import _majority_verdicts, summarize_variance


def _outcome(passed, source="llm"):
    return {"passed": passed, "reason": "", "source": source}


class JudgeVarianceTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [
            {"task_id": "a", "category": "math"},
            {"task_id": "b", "category": "code_debug"},
            {"task_id": "c", "category": "code_gen"},
            {"task_id": "d", "category": "factual"},
        ]
        self.runs = {
            "a": [_outcome(True, "deterministic")] * 3,
            "b": [
                _outcome(True),
                _outcome(False),
                _outcome(True),
            ],
            "c": [_outcome(False)] * 3,
            "d": [_outcome(True, "deterministic")] * 3,
        }

    def test_variance_reports_per_run_accuracy_and_flips(self):
        summary = summarize_variance(self.runs, self.tasks, runs=3)

        self.assertEqual(summary["runs"], 3)
        self.assertEqual(summary["deterministic_tasks"], 2)
        self.assertEqual(summary["llm_tasks"], 2)
        self.assertEqual(summary["accuracy_per_run"], [75.0, 50.0, 75.0])
        self.assertAlmostEqual(summary["accuracy_mean"], 66.67, places=2)
        self.assertAlmostEqual(summary["accuracy_stdev"], 11.79, places=2)
        self.assertEqual(summary["accuracy_min"], 50.0)
        self.assertEqual(summary["accuracy_max"], 75.0)
        self.assertEqual(summary["stable_tasks"], 3)
        self.assertEqual([item["task_id"] for item in summary["unstable_tasks"]], ["b"])
        self.assertEqual(summary["unstable_tasks"][0]["passes"], 2)
        self.assertEqual(summary["unstable_tasks"][0]["runs"], 3)

    def test_single_run_has_no_spread(self):
        runs = {task_id: outcomes[:1] for task_id, outcomes in self.runs.items()}
        summary = summarize_variance(runs, self.tasks, runs=1)
        self.assertEqual(summary["accuracy_stdev"], 0.0)
        self.assertEqual(summary["accuracy_per_run"], [75.0])
        self.assertEqual(len(summary["unstable_tasks"]), 0)

    def test_majority_verdict_uses_strict_majority(self):
        verdicts = _majority_verdicts(self.runs)
        self.assertTrue(verdicts["a"])
        self.assertTrue(verdicts["b"])  # 2/3
        self.assertFalse(verdicts["c"])
        self.assertTrue(verdicts["d"])

        tie = {"x": [_outcome(True), _outcome(False)]}
        self.assertFalse(_majority_verdicts(tie)["x"])

    def test_undecided_is_counted_separately(self):
        runs = {
            "a": [_outcome(False, "undecided")] * 2,
            "b": [_outcome(False)] * 2,
            "c": [_outcome(True)] * 2,
            "d": [_outcome(True)] * 2,
        }
        summary = summarize_variance(runs, self.tasks, runs=2)
        self.assertEqual(summary["undecided_tasks"], 1)
        self.assertEqual(summary["deterministic_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
