"""Tests for scripts/check_secrets.py (the pre-commit/CI secret gate)."""

import tempfile
import unittest
from pathlib import Path

from scripts import check_secrets


class SecretPatternTests(unittest.TestCase):
    def test_forbidden_env_names(self):
        self.assertTrue(check_secrets.is_forbidden_env(Path(".env")))
        self.assertTrue(check_secrets.is_forbidden_env(Path(".env.local")))
        self.assertTrue(check_secrets.is_forbidden_env(Path("config/.env.production")))
        self.assertFalse(check_secrets.is_forbidden_env(Path(".env.example")))

    def test_high_signal_prefixes_are_detected(self):
        samples = {
            "Fireworks API key": "fw_" + "a" * 24,
            "OpenAI API key": "sk-" + "b" * 24,
            "GitHub token": "ghp_" + "c" * 24,
            "AWS access key": "AKIA" + "D" * 16,
            # Split so the scanner does not flag this test file itself.
            "private key block": "-----BEGIN RSA " + "PRIVATE KEY-----",
            "secret assignment": "API_KEY=" + "e" * 24,
        }
        for label, sample in samples.items():
            with self.subTest(label=label):
                findings = check_secrets.scan_text(Path("x.txt"), sample)
                self.assertIn(label, [finding[2] for finding in findings])

    def test_placeholders_are_ignored(self):
        text = "\n".join(
            [
                "FIREWORKS_API_KEY=fw_your_key_here",
                "API_KEY=your-example-placeholder-value",
                "TOKEN=${GITHUB_TOKEN}",
            ]
        )
        self.assertEqual(check_secrets.scan_text(Path("x.txt"), text), [])


class SecretFileTests(unittest.TestCase):
    def test_env_example_allows_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.example"
            path.write_text("FIREWORKS_API_KEY=fw_your_key_here", encoding="utf-8")
            self.assertEqual(check_secrets.scan_file(path), [])

    def test_env_example_still_catches_a_real_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.example"
            path.write_text("FIREWORKS_API_KEY=fw_" + "a" * 24, encoding="utf-8")
            findings = check_secrets.scan_file(path)
        self.assertEqual(findings[0][2], "Fireworks API key")

    def test_real_env_file_is_rejected_even_without_a_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SAFE=yes", encoding="utf-8")
            findings = check_secrets.scan_file(path)
        self.assertEqual(findings[0][2], "real .env file must not be committed")

    def test_binary_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"\0fw_" + b"a" * 24)
            self.assertEqual(check_secrets.scan_file(path), [])


if __name__ == "__main__":
    unittest.main()
