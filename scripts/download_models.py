"""Download and verify model artifacts with cryptographic checksums.

Replaces manual placement instructions with automated, verified downloads from
official Hugging Face repositories for both ONNX specialists and local GGUF models.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from typing import Dict, List, NamedTuple, Optional


class ModelSpec(NamedTuple):
    name: str
    target_relpath: str
    url: str
    sha256: str
    size_bytes: int
    description: str
    group: str


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "ner": ModelSpec(
        name="ner",
        target_relpath="ner/model.onnx",
        url="https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/model.onnx",
        sha256="963039b81eec5b33e23d84826ccdf1e8f8ada776f320e692113034cfae384617",
        size_bytes=431172619,
        description="dslim/bert-base-NER int8 ONNX (NER specialist)",
        group="onnx",
    ),
    "sst2": ModelSpec(
        name="sst2",
        target_relpath="sst2/model.onnx",
        url="https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx",
        sha256="252cf7048af94a1599019fef35961b2bd3d6db13df0b0a4b032b92baeae31939",
        size_bytes=267955711,
        description="DistilBERT-SST2 ONNX (Sentiment specialist)",
        group="onnx",
    ),
    "gguf-3b": ModelSpec(
        name="gguf-3b",
        target_relpath="qwen2.5-3b-instruct-q4_k_m.gguf",
        url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        sha256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
        size_bytes=2104932768,
        description="Qwen2.5-3B-Instruct Q4_K_M GGUF (Local generative model)",
        group="gguf",
    ),
    "gguf-1.5b": ModelSpec(
        name="gguf-1.5b",
        target_relpath="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        sha256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        size_bytes=1117320736,
        description="Qwen2.5-1.5B-Instruct Q4_K_M GGUF (Lightweight local generative)",
        group="gguf",
    ),
}


def compute_sha256(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_file(filepath: str, expected_sha256: str) -> bool:
    """True if file exists and its sha256 matches expected_sha256."""
    if not os.path.exists(filepath):
        return False
    return compute_sha256(filepath).lower() == expected_sha256.lower()


def download_spec(
    spec: ModelSpec,
    base_dir: str = "models",
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Download a single model artifact and verify its sha256."""
    target_path = os.path.join(base_dir, spec.target_relpath)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    if not force and os.path.exists(target_path):
        if not quiet:
            print(f"[{spec.name}] Checking existing file at {target_path}...", end=" ", flush=True)
        if verify_file(target_path, spec.sha256):
            if not quiet:
                print("OK (checksum matches)")
            return True
        elif not quiet:
            print("MISMATCH (re-downloading)")

    tmp_path = target_path + ".tmp"
    if not quiet:
        print(f"[{spec.name}] Downloading {spec.description}")
        print(f"  URL:  {spec.url}")
        print(f"  Dest: {target_path}")
        print(f"  Size: {spec.size_bytes / (1024 * 1024):.1f} MB")

    headers = {"User-Agent": "QueryLane-Model-Downloader/0.16.0"}
    req = urllib.request.Request(spec.url, headers=headers)

    hasher = hashlib.sha256()
    downloaded = 0
    start_time = time.time()
    last_print = 0.0

    try:
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", spec.size_bytes))
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)

                now = time.time()
                if not quiet and (now - last_print > 0.4 or downloaded >= total_size):
                    elapsed = max(0.001, now - start_time)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed
                    pct = (downloaded / total_size) * 100 if total_size else 0
                    print(
                        f"\r  Progress: {downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB "
                        f"({pct:.1f}%) [{speed_mb:.1f} MB/s]",
                        end="",
                        flush=True,
                    )
                    last_print = now

        if not quiet:
            print()

        actual_sha256 = hasher.hexdigest()
        if actual_sha256.lower() != spec.sha256.lower():
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(
                f"Checksum mismatch for {spec.name}!\n"
                f"  Expected: {spec.sha256}\n"
                f"  Actual:   {actual_sha256}"
            )

        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(tmp_path, target_path)

        if not quiet:
            print(f"[{spec.name}] Verified and saved successfully.")
        return True

    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"[{spec.name}] FAILED: {exc}", file=sys.stderr)
        return False


def resolve_models(targets: List[str]) -> List[ModelSpec]:
    """Map target strings ('all', 'onnx', 'gguf', names) to ModelSpecs."""
    selected: Dict[str, ModelSpec] = {}
    for t in targets:
        t = t.lower().strip()
        if t in ("all", "*"):
            selected.update(MODEL_CATALOG)
        elif t == "onnx":
            selected.update({k: v for k, v in MODEL_CATALOG.items() if v.group == "onnx"})
        elif t == "gguf":
            selected.update({k: v for k, v in MODEL_CATALOG.items() if v.group == "gguf"})
        elif t in MODEL_CATALOG:
            selected[t] = MODEL_CATALOG[t]
        else:
            raise ValueError(f"Unknown model identifier '{t}'. Valid options: {list(MODEL_CATALOG.keys()) + ['all', 'onnx', 'gguf']}")
    return list(selected.values())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify QueryLane ONNX and GGUF model artifacts."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["onnx"],
        help="Models to download: onnx (default), gguf, all, ner, sst2, gguf-3b, gguf-1.5b.",
    )
    parser.add_argument(
        "--dest-dir",
        default="models",
        help="Target base directory for models (default: models/).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify checksums of existing files without downloading missing ones.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if already present with valid checksum.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress output.",
    )

    args = parser.parse_args(argv)

    try:
        specs = resolve_models(args.models)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    all_ok = True
    if args.verify_only:
        print(f"Verifying {len(specs)} model artifact(s)...")
        for spec in specs:
            target = os.path.join(args.dest_dir, spec.target_relpath)
            if not os.path.exists(target):
                print(f"[{spec.name}] MISSING: {target}")
                all_ok = False
            else:
                ok = verify_file(target, spec.sha256)
                if ok:
                    print(f"[{spec.name}] OK: {target}")
                else:
                    print(f"[{spec.name}] CORRUPTED: checksum mismatch for {target}")
                    all_ok = False
        return 0 if all_ok else 1

    print(f"Processing {len(specs)} model artifact(s) into '{args.dest_dir}'...")
    for spec in specs:
        success = download_spec(
            spec,
            base_dir=args.dest_dir,
            force=args.force,
            quiet=args.quiet,
        )
        if not success:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
