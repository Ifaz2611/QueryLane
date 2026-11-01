"""Fail a commit or CI run when it contains credentials or a real .env file.

This deliberately uses a small, high-signal pattern set instead of entropy
heuristics: false positives on lockfiles and docs are more annoying than a
few targeted prefixes. Run it from pre-commit (paths are passed in) or from
CI with ``--all`` (scans every git-tracked file).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEMPT_NAMES = {".env.example"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "source",
}
SKIP_SUFFIXES = {
    ".gguf",
    ".onnx",
    ".safetensors",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
}
MAX_BYTES = 2_000_000

PATTERNS = (
    ("Fireworks API key", re.compile(r"\bfw_[A-Za-z0-9]{16,}\b")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|password)"
            r"\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{20,})"
        ),
    ),
)
PLACEHOLDER_RE = re.compile(
    r"(?i)your|example|sample|dummy|placeholder|changeme|replace|xxx|"
    r"<[^>]+>|\$\{|\benv\.|redacted"
)


def is_forbidden_env(path: Path) -> bool:
    """True for real dotenv files; ``.env.example`` is the only exception."""
    name = path.name.lower()
    if name in EXEMPT_NAMES:
        return False
    return name == ".env" or name.startswith(".env.")


def scan_text(path: Path, text: str) -> list[tuple[str, int, str]]:
    """Return ``(path, line, label)`` for each credential-looking line."""
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            for match in pattern.finditer(line):
                value = match.group(1) if match.groups() else match.group(0)
                if PLACEHOLDER_RE.search(value):
                    continue
                findings.append((str(path), line_number, label))
                break
    return findings


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    if is_forbidden_env(path):
        return [(str(path), 0, "real .env file must not be committed")]
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        data = path.read_bytes()
    except OSError:
        return []
    if b"\0" in data[:1024]:
        return []
    return scan_text(path, data.decode("utf-8", errors="ignore"))


def _git_tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file()]
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [ROOT / name for name in names if name]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to scan (pre-commit passes these)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan every git-tracked file instead of explicit paths",
    )
    args = parser.parse_args(argv)

    if args.paths:
        paths = [_resolve(path) for path in args.paths]
    elif args.all:
        paths = _git_tracked_files()
    else:
        parser.error("pass one or more files, or --all")

    findings = []
    checked = 0
    for path in paths:
        if not path.is_file() or should_skip(path):
            continue
        checked += 1
        findings.extend(scan_file(path))

    if findings:
        print("secret scan failed:", file=sys.stderr)
        for path, line_number, label in findings:
            location = f"{path}:{line_number}" if line_number else path
            print(f"  {location}: {label}", file=sys.stderr)
        print(
            "Remove the credential, rotate it if it was real, and add the file "
            "to .gitignore. Use .env.example for placeholders.",
            file=sys.stderr,
        )
        return 1

    print(f"secret scan: {checked} file(s) checked, no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
