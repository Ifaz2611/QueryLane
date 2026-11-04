"""Tests for structured logging, metrics export, duplicates, and --dry-run."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import agent.main as main_module
from agent import obs


class FakeCloud:
    """Cloud stand-in whose token counter grows like the real client's."""

    def __init__(self, answer="cloud answer"):
        self._answer = answer
        self.answered = []
        self._tokens = 7

    @classmethod
    def from_env(cls):
        return cls()

    def answer(self, category, prompt):
        self.answered.append((category, prompt))
        self._tokens += 7  # one ~7-token answer per task, as billed
        return self._answer

    def call_count(self):
        return len(self.answered)

    def total_tokens(self):
        return self._tokens

    def tokens_by_category(self):
        return {"factual": 3, "sentiment": 4}

    def retry_count(self):
        return 1


@contextlib.contextmanager
def _stdout():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def _json_lines(buffer):
    """Parse the JSON event lines, ignoring human-readable status prints."""
    rows = []
    for line in buffer.getvalue().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


class ObsTests(unittest.TestCase):
    def test_log_event_is_silent_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True), _stdout() as buffer:
            obs.log_event("task", task_id="a", lane="cloud")
        self.assertEqual(buffer.getvalue(), "")

    def test_log_event_writes_one_json_line_per_event(self):
        env = {"AGENT_LOG_JSON": "1"}
        with mock.patch.dict(os.environ, env, clear=True), _stdout() as buffer:
            obs.log_event("task", task_id="a", lane="cloud", tokens=3)
            obs.log_event("run_summary", tasks=1)
        rows = _json_lines(buffer)
        self.assertEqual([row["event"] for row in rows], ["task", "run_summary"])
        self.assertEqual(rows[0]["task_id"], "a")
        self.assertEqual(rows[0]["level"], "info")
        self.assertEqual(rows[0]["tokens"], 3)
        self.assertNotIn("task_id", rows[1])
        self.assertIn("ts", rows[0])

    def test_log_event_serializes_unknown_objects_instead_of_raising(self):
        env = {"AGENT_LOG_JSON": "yes"}
        with mock.patch.dict(os.environ, env, clear=True), _stdout() as buffer:
            obs.log_event("weird", payload=object())
        row = _json_lines(buffer)[0]
        self.assertIn("payload", row)

    def test_flag_values(self):
        for value, expected in (
            ("1", True),
            ("on", True),
            ("0", False),
            ("off", False),
            ("disabled", False),
        ):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"X": value}, clear=True):
                    self.assertEqual(obs.flag("X"), expected)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(obs.flag("X"))


class RunMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = obs.RunMetrics()

    def test_record_task_counts_lanes_categories_and_outcomes(self):
        self.metrics.record_task("sentiment", "specialist", tokens=0)
        self.metrics.record_task("factual", "cloud", tokens=90, empty=True)
        self.metrics.record_task(
            "math", "escalated", tokens=40, escalated=True, low_confidence=True
        )
        snap = self.metrics.snapshot()

        self.assertEqual(snap["tasks"], 3)
        self.assertEqual(snap["lanes"], {"specialist": 1, "cloud": 1, "escalated": 1})
        self.assertEqual(snap["categories"], {"sentiment": 1, "factual": 1, "math": 1})
        self.assertEqual(snap["escalations"], {"escalated": 1})
        self.assertEqual(snap["empty_answers"], 1)
        self.assertEqual(snap["low_confidence_routes"], 1)
        self.assertEqual(snap["tokens"]["total"], 130)
        self.assertEqual(snap["tokens"]["by_category"], {"factual": 90, "math": 40})
        self.assertEqual(snap["tokens"]["per_task"], 43.33)

    def test_record_task_ignores_negative_and_non_int_tokens(self):
        self.metrics.record_task("factual", "cloud", tokens=-5)
        self.metrics.record_task("factual", "cloud", tokens=None)
        self.assertEqual(self.metrics.snapshot()["tokens"]["total"], 0)

    def test_duplicates_and_missing_prompts_are_recorded(self):
        self.metrics.record_duplicates(2, 2)
        self.metrics.record_missing_prompts(1)
        snap = self.metrics.snapshot()
        self.assertEqual(snap["duplicates"], {"repeated_task_ids": 2, "skipped_rows": 2})
        self.assertEqual(snap["missing_prompts"], 1)

    def test_snapshot_merges_cloud_counters_and_tolerates_their_absence(self):
        snap = self.metrics.snapshot(cloud=FakeCloud(), elapsed=1.5)
        self.assertEqual(snap["cloud"]["calls"], 0)
        self.assertEqual(snap["cloud"]["tokens"], 7)
        self.assertEqual(snap["cloud"]["retries"], 1)
        self.assertEqual(
            snap["cloud"]["tokens_by_category"], {"factual": 3, "sentiment": 4}
        )
        self.assertEqual(snap["elapsed_seconds"], 1.5)

        bare = self.metrics.snapshot(cloud=object())
        self.assertEqual(bare["cloud"]["calls"], 0)
        self.assertEqual(bare["cloud"]["tokens_by_category"], {})

    def test_save_writes_json_and_creates_directories(self):
        self.metrics.record_task("factual", "cloud", tokens=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "metrics.json")
            snapshot = self.metrics.save(path, cloud=FakeCloud())
            with open(path, encoding="utf-8") as fh:
                on_disk = json.load(fh)
        self.assertEqual(on_disk["tasks"], 1)
        self.assertEqual(on_disk, snapshot)


class DryRunTests(unittest.TestCase):
    TASKS = [
        {"task_id": "a", "prompt": "What is the capital of France?"},
        {"task_id": "b", "prompt": "Write a Python function that adds two numbers."},
    ]

    def _write_tasks(self, tmp):
        tasks_path = os.path.join(tmp, "tasks.json")
        with open(tasks_path, "w", encoding="utf-8") as fh:
            json.dump(self.TASKS, fh)
        return tasks_path

    def test_dry_run_routes_without_touching_the_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = self._write_tasks(tmp)
            output_path = os.path.join(tmp, "results.json")
            env = {"TASKS_PATH": tasks_path, "RESULTS_PATH": output_path}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                main_module.CloudClient, "from_env"
            ) as from_env, mock.patch.object(
                main_module.local_model, "preload"
            ) as preload:
                self.assertEqual(main_module.main(["--dry-run"]), 0)
            from_env.assert_not_called()
            preload.assert_not_called()
            with open(output_path, encoding="utf-8") as fh:
                results = json.load(fh)

        self.assertEqual([row["task_id"] for row in results], ["a", "b"])
        self.assertEqual([row["answer"] for row in results], ["", ""])

    def test_dry_run_env_flag_matches_the_cli_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = self._write_tasks(tmp)
            env = {
                "TASKS_PATH": tasks_path,
                "RESULTS_PATH": os.path.join(tmp, "results.json"),
                "AGENT_DRY_RUN": "1",
            }
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                main_module.CloudClient, "from_env"
            ) as from_env:
                self.assertEqual(main_module.main([]), 0)
            from_env.assert_not_called()

    def test_dry_run_entry_reports_a_lane_and_no_answer(self):
        entry = main_module.dry_run_entry(
            {"task_id": "a", "prompt": "Summarize this in one sentence: The cat sat."}
        )
        self.assertEqual(entry["category"], "summarization")
        self.assertEqual(entry["routed"], entry["predicted"])
        self.assertIn(entry["routed"], {"cloud", "specialist", "local"})
        self.assertEqual(entry["tokens"], 0)
        self.assertEqual(entry["answer"], "")

    def test_unknown_cli_flags_are_ignored(self):
        dry_run, _ = main_module._parse_args(["-v", "--cov=agent", "-k", "foo"])
        self.assertFalse(dry_run)
        self.assertTrue(main_module._parse_args(["--dry-run"])[0])

    def test_help_prints_usage_without_touching_the_input_file(self):
        with mock.patch.dict(os.environ, {}, clear=True), _stdout() as buffer:
            self.assertEqual(main_module.main(["--help"]), 0)
        text = buffer.getvalue()
        self.assertIn("usage: querylane", text)
        self.assertIn("--dry-run", text)
        self.assertIn("METRICS_PATH", text)

    def test_predict_lane_short_circuits_on_low_confidence(self):
        with mock.patch.object(main_module, "MIN_CONFIDENCE", 0.70):
            self.assertEqual(
                main_module.predict_lane("sentiment", "local", 0.10, "x"), "cloud"
            )

    def test_predict_lane_prefers_specialists_then_local_then_cloud(self):
        with mock.patch.object(main_module, "MIN_CONFIDENCE", 0.70), mock.patch(
            "agent.specialists.available", return_value=True
        ):
            self.assertEqual(
                main_module.predict_lane("sentiment", "cloud", 0.95, "x"), "specialist"
            )
        with mock.patch.object(main_module, "MIN_CONFIDENCE", 0.70), mock.patch(
            "agent.specialists.available", return_value=False
        ), mock.patch("agent.specialists.ner_enabled", return_value=False), mock.patch.object(
            main_module.local_model, "available", return_value=True
        ), mock.patch.object(
            main_module.local_model, "category_enabled", return_value=True
        ):
            self.assertEqual(
                main_module.predict_lane("factual", "local", 0.95, "x"), "local"
            )
        with mock.patch.object(main_module, "MIN_CONFIDENCE", 0.70), mock.patch(
            "agent.specialists.available", return_value=False
        ), mock.patch(
            "agent.specialists.ner_enabled", return_value=False
        ), mock.patch.object(
            main_module.local_model, "available", return_value=False
        ):
            self.assertEqual(
                main_module.predict_lane("factual", "local", 0.95, "x"), "cloud"
            )


class DuplicateTaskIdTests(unittest.TestCase):
    def test_dedupe_keeps_first_row_and_counts_repeats(self):
        tasks = [
            {"task_id": "a", "prompt": "first"},
            {"task_id": "b", "prompt": "other"},
            {"task_id": "a", "prompt": "second"},
        ]
        unique, duplicates = main_module._dedupe_tasks(tasks)
        self.assertEqual([task["prompt"] for task in unique], ["first", "other"])
        self.assertEqual(duplicates, 1)

    def test_main_answers_each_duplicate_once_and_keeps_every_row(self):
        cloud = FakeCloud("shared answer")
        tasks = [
            {"task_id": "a", "prompt": "What is 2 + 2?"},
            {"task_id": "a", "prompt": "What is 2 + 2?"},
            {"task_id": "b", "prompt": "Name a planet."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            output_path = os.path.join(tmp, "results.json")
            metrics_path = os.path.join(tmp, "nested", "metrics.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump(tasks, fh)
            env = {
                "TASKS_PATH": tasks_path,
                "RESULTS_PATH": output_path,
                "METRICS_PATH": metrics_path,
            }
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                main_module, "CloudClient"
            ) as client_cls, mock.patch.object(main_module.local_model, "preload"):
                client_cls.from_env.return_value = cloud
                self.assertEqual(main_module.main([]), 0)
            with open(output_path, encoding="utf-8") as fh:
                results = json.load(fh)
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)

        self.assertEqual([row["task_id"] for row in results], ["a", "a", "b"])
        self.assertTrue(all(row["answer"] == "shared answer" for row in results))
        self.assertEqual(len(cloud.answered), 2)  # 'a' was billed once
        self.assertEqual(
            metrics["duplicates"], {"repeated_task_ids": 1, "skipped_rows": 1}
        )

    def test_metrics_are_written_and_the_run_summary_is_logged(self):
        tasks = [{"task_id": "a", "prompt": "Name a planet."}]
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            output_path = os.path.join(tmp, "results.json")
            metrics_path = os.path.join(tmp, "metrics.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump(tasks, fh)
            env = {
                "TASKS_PATH": tasks_path,
                "RESULTS_PATH": output_path,
                "METRICS_PATH": metrics_path,
                "AGENT_LOG_JSON": "1",
            }
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                main_module, "CloudClient"
            ) as client_cls, mock.patch.object(
                main_module.local_model, "preload"
            ), _stdout() as buffer:
                client_cls.from_env.return_value = FakeCloud("answer")
                self.assertEqual(main_module.main([]), 0)
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
            events = _json_lines(buffer)

        self.assertEqual(metrics["tasks"], 1)
        self.assertEqual(metrics["lanes"], {"cloud": 1})
        self.assertEqual(metrics["tokens"]["total"], 7)
        task_events = [event for event in events if event["event"] == "task"]
        self.assertEqual(task_events[0]["task_id"], "a")
        self.assertEqual(task_events[0]["category"], "factual")
        self.assertEqual(task_events[0]["lane"], "cloud")
        self.assertIn("elapsed_seconds", task_events[0])
        self.assertEqual(task_events[0]["tokens"], 7)
        summaries = [event for event in events if event["event"] == "run_summary"]
        self.assertEqual(summaries[0]["tasks"], 1)

    def test_missing_prompt_is_treated_as_an_empty_string(self):
        entry = main_module.dry_run_entry({"task_id": "a", "prompt": None})
        self.assertEqual(entry["category"], "factual")
        self.assertEqual(entry["answer"], "")


if __name__ == "__main__":
    unittest.main()
