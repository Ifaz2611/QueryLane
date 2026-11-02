"""Versioned paraphrase regression suite for the lexical router.

Every `eval/router_paraphrases_v*.json` file is discovered automatically, so a
new version can be added without touching the test runner.
"""

import glob
import json
import os
import re
import unittest

from agent.router import ALL_CATEGORIES, classify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_RE = re.compile(r"router_paraphrases_v(\d+)\.json$")


class ParaphraseSuiteTests(unittest.TestCase):
    def test_versioned_suites_exist_and_route_correctly(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "eval", "router_paraphrases_v*.json")))
        self.assertTrue(paths, "no router paraphrase suite found")

        for path in paths:
            match = SUITE_RE.search(os.path.basename(path))
            self.assertIsNotNone(match, path)
            with self.subTest(suite=os.path.basename(path)):
                with open(path, encoding="utf-8") as fh:
                    suite = json.load(fh)

                self.assertEqual(suite["version"], int(match.group(1)))
                self.assertIn("description", suite)
                self.assertGreaterEqual(len(suite["prompts"]), 64)
                self.assertEqual(
                    sorted(suite["categories"]), sorted(ALL_CATEGORIES)
                )

                ids = [entry["id"] for entry in suite["prompts"]]
                prompts = [entry["prompt"] for entry in suite["prompts"]]
                self.assertEqual(len(ids), len(set(ids)), "duplicate prompt ids")
                self.assertEqual(
                    len(prompts), len(set(prompts)), "duplicate prompt text"
                )

                for entry in suite["prompts"]:
                    with self.subTest(prompt_id=entry["id"]):
                        self.assertIn(entry["category"], ALL_CATEGORIES)
                        self.assertEqual(classify(entry["prompt"]), entry["category"])


if __name__ == "__main__":
    unittest.main()
