"""End-to-end tests for ``python -m agent.main`` with a mocked cloud client.

The OpenAI SDK talks to a tiny local HTTP server that implements the subset of
the chat-completions contract the agent uses. That exercises the real
entrypoint, environment parsing, routing, lane cascade, PAL parsing, output
contract, and process exit code without spending any Fireworks tokens.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_PATH = os.path.join(ROOT, "testio", "input", "tasks.json")
MOCK_MODEL = "accounts/fireworks/models/mock-e2e"


class _MockOpenAI:
    """Minimal OpenAI-compatible server that records every request."""

    def __init__(self):
        self.requests = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="mock-openai", daemon=True
        )

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {}

                owner.requests.append(payload)
                messages = payload.get("messages") or []
                system = " ".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "system"
                )
                user = " ".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "user"
                )

                if "EXPR:" in system and "UNIT:" in system:
                    content = "EXPR: 2 + 2 | UNIT: none"
                elif "sentiment" in system.lower():
                    content = "Positive. The mocked review is clearly favorable."
                elif "Entity: Type" in system or "named entity" in system.lower():
                    content = "Maria Sanchez: Person"
                else:
                    content = "Mocked cloud answer."

                prompt_tokens = max(1, len(system + user) // 4)
                completion_tokens = max(1, len(content) // 4)
                body = json.dumps(
                    {
                        "id": "chatcmpl-e2e",
                        "object": "chat.completion",
                        "created": 0,
                        "model": payload.get("model", MOCK_MODEL),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep test output clean
                return

        return Handler

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _clean_env(updates):
    env = os.environ.copy()
    for key in (
        "AGENT_DRY_RUN",
        "AGENT_LOG_JSON",
        "AGENT_DEBUG",
        "METRICS_PATH",
        "TASKS_PATH",
        "RESULTS_PATH",
        "LOCAL_MODEL_PATH",
        "LOCAL_CATEGORIES",
        "NER_SPECIALIST",
    ):
        env.pop(key, None)
    # A developer proxy must not intercept the loopback mock server.
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    env.update(updates)
    return env


def _run_entrypoint(env, timeout=120):
    return subprocess.run(
        [sys.executable, "-m", "agent.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class EntrypointE2ETests(unittest.TestCase):
    def test_entrypoint_answers_testio_batch_with_mocked_cloud(self):
        with tempfile.TemporaryDirectory() as tmp, _MockOpenAI() as server:
            results_path = os.path.join(tmp, "results.json")
            env = _clean_env(
                {
                    "TASKS_PATH": TASKS_PATH,
                    "RESULTS_PATH": results_path,
                    "FIREWORKS_API_KEY": "test-key",
                    "FIREWORKS_BASE_URL": server.base_url,
                    "ALLOWED_MODELS": MOCK_MODEL,
                    "LOCAL_MODEL_PATH": os.path.join(tmp, "missing.gguf"),
                    "NER_SPECIALIST": "off",
                    "AGENT_WORKERS": "2",
                    "CLOUD_TASK_BUDGET": "30",
                    "AGENT_DEADLINE": "120",
                }
            )

            proc = _run_entrypoint(env)
            self.assertEqual(
                proc.returncode,
                0,
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )

            with open(TASKS_PATH, encoding="utf-8") as fh:
                expected_ids = [task["task_id"] for task in json.load(fh)]
            with open(results_path, encoding="utf-8") as fh:
                results = json.load(fh)

            self.assertEqual([row["task_id"] for row in results], expected_ids)
            self.assertTrue(all(row["answer"].strip() for row in results))
            # The mocked EXPR answer must be evaluated through the real PAL path.
            math_row = next(row for row in results if row["task_id"] == "practice-02")
            self.assertEqual(math_row["answer"], "Answer: 4")

            self.assertGreaterEqual(len(server.requests), len(expected_ids))
            self.assertTrue(
                all(request.get("model") == MOCK_MODEL for request in server.requests)
            )
            self.assertTrue(
                any("messages" in request for request in server.requests),
                "the entrypoint did not send chat-completion requests",
            )

    def test_dry_run_entrypoint_spends_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.json")
            env = _clean_env(
                {
                    "TASKS_PATH": TASKS_PATH,
                    "RESULTS_PATH": results_path,
                    "AGENT_DRY_RUN": "1",
                    "LOCAL_MODEL_PATH": os.path.join(tmp, "missing.gguf"),
                }
            )

            proc = _run_entrypoint(env)
            self.assertEqual(
                proc.returncode,
                0,
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            self.assertIn("dry run", proc.stdout.lower())

            with open(results_path, encoding="utf-8") as fh:
                results = json.load(fh)
            self.assertEqual(len(results), 8)
            self.assertTrue(all(row["answer"] == "" for row in results))


if __name__ == "__main__":
    unittest.main()
