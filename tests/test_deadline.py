"""Tests for the global deadline and lane fallback behavior in agent/main.py."""

import json
import os
import tempfile
import unittest
from unittest import mock

import agent.main as main_module
from agent import local as local_model


class FakeCloud:
    def __init__(self, answer="cloud answer", error=None):
        self._answer = answer
        self._error = error

    @classmethod
    def from_env(cls):
        return cls()

    def answer(self, category, prompt):
        if self._error:
            raise self._error
        return self._answer

    def call_count(self):
        return 1

    def total_tokens(self):
        return 7


class _TimeoutFuture:
    def result(self, timeout=None):
        raise TimeoutError("deadline exceeded")


class _FakePool:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.shutdown_args = None
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        return _TimeoutFuture()

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_args = (wait, cancel_futures)


class DeadlineTests(unittest.TestCase):
    def test_remaining_uses_module_start_and_deadline(self):
        with mock.patch.object(main_module, "DEADLINE_SECONDS", 10), mock.patch.object(
            main_module, "_START", 95.0
        ), mock.patch.object(main_module.time, "time", return_value=100.0):
            self.assertEqual(main_module._remaining(), 5.0)

    def test_main_writes_every_task_when_workers_miss_the_deadline(self):
        pool = _FakePool()
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            output_path = os.path.join(tmp, "results.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [
                        {"task_id": "a", "prompt": "What is 2 + 2?"},
                        {"task_id": "b", "prompt": "Name a planet."},
                    ],
                    fh,
                )
            with mock.patch.dict(
                os.environ,
                {"TASKS_PATH": tasks_path, "RESULTS_PATH": output_path},
            ), mock.patch.object(
                main_module, "ThreadPoolExecutor", return_value=pool
            ), mock.patch.object(
                main_module, "CloudClient", FakeCloud
            ), mock.patch.object(
                local_model, "preload"
            ):
                self.assertEqual(main_module.main(), 0)

            with open(output_path, encoding="utf-8") as fh:
                results = json.load(fh)

        self.assertEqual([row["task_id"] for row in results], ["a", "b"])
        self.assertEqual([row["answer"] for row in results], ["", ""])
        self.assertEqual(pool.shutdown_args, (False, True))


class LaneFallbackTests(unittest.TestCase):
    def test_local_route_falls_back_when_cutoff_is_near(self):
        cloud = FakeCloud("cloud answer")
        with mock.patch.object(
            main_module, "route", return_value=("factual", "local")
        ), mock.patch.object(main_module, "_remaining", return_value=1.0):
            entry = main_module.answer_task(cloud, {"task_id": "a", "prompt": "Why?"})

        self.assertEqual(entry["answer"], "cloud answer")
        self.assertEqual(entry["routed"], "cloud-fallback")

    def test_suspicious_local_answer_escalates_to_cloud(self):
        cloud = FakeCloud("cloud answer")
        with mock.patch.object(
            main_module, "route", return_value=("factual", "local")
        ), mock.patch.object(
            main_module, "_remaining", return_value=1000.0
        ), mock.patch.object(
            local_model, "available", return_value=True
        ), mock.patch.object(
            local_model, "wait_ready", return_value=True
        ), mock.patch.object(
            local_model, "category_enabled", return_value=True
        ), mock.patch.object(
            local_model, "answer", return_value="suspicious"
        ), mock.patch.object(
            local_model, "answer_ok", return_value=False
        ):
            entry = main_module.answer_task(cloud, {"task_id": "a", "prompt": "Why?"})

        self.assertEqual(entry["answer"], "cloud answer")
        self.assertEqual(entry["routed"], "escalated")

    def test_local_exception_falls_through_to_cloud(self):
        cloud = FakeCloud("cloud answer")
        with mock.patch.object(
            main_module, "route", return_value=("factual", "local")
        ), mock.patch.object(
            main_module, "_remaining", return_value=1000.0
        ), mock.patch.object(
            local_model, "available", return_value=True
        ), mock.patch.object(
            local_model, "wait_ready", return_value=True
        ), mock.patch.object(
            local_model, "category_enabled", return_value=True
        ), mock.patch.object(
            local_model, "answer", side_effect=RuntimeError("crashed")
        ), mock.patch("builtins.print"):
            entry = main_module.answer_task(cloud, {"task_id": "a", "prompt": "Why?"})

        self.assertEqual(entry["answer"], "cloud answer")

    def test_specialist_exception_falls_through_to_cloud(self):
        cloud = FakeCloud("cloud answer")
        with mock.patch.object(
            main_module, "route", return_value=("sentiment", "cloud")
        ), mock.patch(
            "agent.specialists.available", return_value=True
        ), mock.patch(
            "agent.specialists.sentiment_answer", side_effect=RuntimeError("boom")
        ), mock.patch("builtins.print"):
            entry = main_module.answer_task(
                cloud, {"task_id": "a", "prompt": "Classify the sentiment: bad."}
            )

        self.assertEqual(entry["answer"], "cloud answer")

    def test_multipart_math_is_forced_to_cloud(self):
        cloud = FakeCloud("cloud answer")
        prompt = "What is 2 + 2, and what is 3 + 3?"
        with mock.patch.object(
            main_module, "route", return_value=("math", "local")
        ), mock.patch.object(
            main_module.pal, "is_multipart", return_value=True
        ):
            entry = main_module.answer_task(cloud, {"task_id": "a", "prompt": prompt})

        self.assertEqual(entry["routed"], "cloud")
        self.assertEqual(entry["answer"], "cloud answer")


class MainFailureTests(unittest.TestCase):
    def test_bad_cloud_configuration_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = os.path.join(tmp, "tasks.json")
            with open(tasks_path, "w", encoding="utf-8") as fh:
                json.dump([{"task_id": "a", "prompt": "Why?"}], fh)
            with mock.patch.dict(os.environ, {"TASKS_PATH": tasks_path}), mock.patch.object(
                main_module.CloudClient,
                "from_env",
                side_effect=ValueError("no key"),
            ), mock.patch("builtins.print"):
                self.assertEqual(main_module.main(), 1)

    def test_cloud_failure_still_returns_a_string_answer(self):
        cloud = FakeCloud(error=RuntimeError("cloud down"))
        with mock.patch.object(
            main_module, "route", return_value=("factual", "cloud")
        ), mock.patch("builtins.print"):
            entry = main_module.answer_task(cloud, {"task_id": "a", "prompt": "Why?"})

        self.assertEqual(entry["task_id"], "a")
        self.assertEqual(entry["answer"], "")


if __name__ == "__main__":
    unittest.main()
