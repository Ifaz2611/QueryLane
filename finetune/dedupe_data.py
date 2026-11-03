"""Deduplicate exact and near-identical prompts in QueryLane's SFT JSONL files.

Two files are shipped under finetune/data: train_all.jsonl and
train_round2.jsonl. Investigation (2026-11-03) showed they contain the exact
same 2,819 rows in a different order — plus 8 factual prompts that appear
twice inside the dataset with diverging answers. This tool:

- loads one or more JSONL datasets (default: the two files above),
- drops EXACT duplicate prompts (normalized: casefolded, punctuation and
  whitespace collapsed, so straight/curly apostrophe variants collide),
- detects NEAR-identical prompts within the same category using token-set
  overlap as a cheap prefilter and difflib.SequenceMatcher as the ratio,
- optionally drops rows the validator rejects (finetune.validate_data), and
- writes a canonical deduplicated JSONL plus a machine-readable report that
  documents the relationship between the input files.

Usage:
    python -m finetune.dedupe_data                      # canonical artifact + report
    python -m finetune.dedupe_data a.jsonl b.jsonl --drop-near --out merged.jsonl
Exit code 0 unless a file fails to load at all.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher

from finetune.validate_data import check_row, normalize_prompt, strip_category_prefix

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_FILES = (
    os.path.join(DATA_DIR, "train_all.jsonl"),
    os.path.join(DATA_DIR, "train_round2.jsonl"),
)
DEFAULT_OUT = os.path.join(DATA_DIR, "train_dedup.jsonl")
DEFAULT_REPORT = os.path.join(DATA_DIR, "dedupe_report.json")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_with_lines(path: str) -> tuple:
    """Read a JSONL file keeping (line_no, row) pairs; returns (entries, issues)."""
    entries, issues = [], []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            entries.append({"file": os.path.basename(path), "path": path, "line": line_no, "row": row})
    return entries, issues


def _token_set(text: str) -> set:
    return set(normalize_prompt(text).split())


def find_near_pairs(entries, ratio_threshold: float, jaccard_gate: float) -> list:
    """Near-identical same-category prompt pairs. Token-set overlap is the
    cheap prefilter (robust to single-word swaps and reordering); the
    difflib.SequenceMatcher ratio on the prefix-stripped prompt decides."""
    per_index = []
    inverted = {}
    for idx, entry in enumerate(entries):
        row = entry["row"]
        category = row.get("category")
        content = strip_category_prefix(category, row["messages"][1]["content"])
        tokens = _token_set(content)
        per_index.append((category, content, tokens))
        for token in tokens:
            inverted.setdefault(token, []).append(idx)

    reported = []
    seen_reported = set()
    for idx, (category, content, tokens) in enumerate(per_index):
        overlap = Counter()
        for token in tokens:
            for other in inverted.get(token, ()):
                if other > idx:
                    overlap[other] += 1
        for other, inter in overlap.items():
            other_category, other_content, other_tokens = per_index[other]
            if category != other_category:
                continue
            union = len(tokens) + len(other_tokens) - inter
            if not union or inter / union < jaccard_gate:
                continue
            ratio = SequenceMatcher(None, content, other_content).ratio()
            if ratio < ratio_threshold:
                continue
            key = (idx, other) if idx < other else (other, idx)
            if key in seen_reported:
                continue
            seen_reported.add(key)
            reported.append(
                {
                    "a": {"file": entries[idx]["file"], "line": entries[idx]["line"], "index": idx},
                    "b": {"file": entries[other]["file"], "line": entries[other]["line"], "index": other},
                    "category": category,
                    "ratio": round(ratio, 4),
                }
            )
    return reported


# --------------------------------------------------------------------------- #
# Relationship analysis and CLI                                               #
# --------------------------------------------------------------------------- #
def _row_signature(row) -> str:
    """Canonical row signature for multiset comparison between files."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _pair_relationship(entries_a: list, entries_b: list) -> dict:
    """How do two files relate as multiset of full rows?"""
    sig_a = Counter(_row_signature(entry["row"]) for entry in entries_a)
    sig_b = Counter(_row_signature(entry["row"]) for entry in entries_b)
    common = sum((sig_a & sig_b).values())
    only_a = sum((sig_a - sig_b).values())
    only_b = sum((sig_b - sig_a).values())
    if common and not only_a and not only_b:
        relationship = "identical-multiset (same rows, order differs)"
    elif not common:
        relationship = "disjoint"
    elif not only_b:
        relationship = "a contains all of b"
    elif not only_a:
        relationship = "b contains all of a"
    else:
        relationship = "overlapping"
    return {"common_rows": common, "only_in_first": only_a, "only_in_second": only_b, "relationship": relationship}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deduplicate QueryLane SFT JSONL files.")
    parser.add_argument("files", nargs="*", help=f"JSONL inputs (default: {', '.join(DEFAULT_FILES)})")
    parser.add_argument("--out", default=DEFAULT_OUT, help="deduplicated union JSONL (empty string skips)")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="JSON report path (empty string skips)")
    parser.add_argument("--near-threshold", type=float, default=0.92, help="difflib ratio gate for near-dupes")
    parser.add_argument("--jaccard-gate", type=float, default=0.5, help="token-overlap prefilter for near-dupes")
    parser.add_argument("--drop-near", action="store_true", help="also drop near-identical rows (first wins)")
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="also drop rows the validator rejects (see finetune.validate_data)",
    )
    parser.add_argument("--list", type=int, default=10, help="pairs/groups to list in the console summary")
    args = parser.parse_args(argv)

    files = args.files or list(DEFAULT_FILES)
    if not files:
        print("no input files given", file=sys.stderr)
        return 2

    inputs = []
    all_entries = []
    for path in files:
        entries, load_issues = _load_with_lines(path)
        if load_issues:
            for issue in load_issues:
                print(f"warning: {issue}", file=sys.stderr)
        inputs.append({"path": path, "sha256": file_sha256(path), "rows": len(entries)})
        # In-file duplicate prompt groups, per file, for the record.
        per_file = Counter(normalize_prompt(e["row"]["messages"][1]["content"]) for e in entries)
        inputs[-1]["duplicate_prompt_groups_in_file"] = sum(1 for c in per_file.values() if c > 1)
        all_entries.extend(entries)
        print(f"loaded {len(entries)} rows from {path}")

    # 1) optional validator-driven invalid-row drop
    invalid = []
    if args.drop_invalid:
        survivors = []
        for entry in all_entries:
            row_errors = [i for i in check_row(entry["row"], line_no=entry["line"]) if i.level == "error"]
            if row_errors:
                invalid.append(
                    {
                        "file": entry["file"],
                        "line": entry["line"],
                        "category": entry["row"].get("category"),
                        "errors": [f"{i.code}: {i.detail}" for i in row_errors],
                    }
                )
            else:
                survivors.append(entry)
        for item in invalid[: args.list]:
            print(f"invalid: {item['file']}:{item['line']} {item['errors'][0]}")
        print(f"dropped {len(invalid)} invalid rows; {len(survivors)} remain")
        all_entries = survivors

    # 2) exact duplicate prompts across ALL inputs, first occurrence wins
    kept = []
    dropped_exact = []
    first_seen = {}
    for entry in all_entries:
        key = normalize_prompt(entry["row"]["messages"][1]["content"])
        if key in first_seen:
            dropped_exact.append(
                {
                    "file": entry["file"],
                    "line": entry["line"],
                    "category": entry["row"].get("category"),
                    "duplicate_of": dict(first_seen[key]),
                }
            )
        else:
            first_seen[key] = {"file": entry["file"], "line": entry["line"]}
            kept.append(entry)
    print(f"dropped {len(dropped_exact)} exact duplicate rows; {len(kept)} remain")

    # 3) near-identical prompts among the survivors (report; drop only on flag)
    near_pairs = find_near_pairs(kept, args.near_threshold, args.jaccard_gate)
    dropped_near = []
    if near_pairs and args.drop_near:
        drop_idx = {pair["b"]["index"] for pair in near_pairs}
        dropped_near = [
            {"file": kept[idx]["file"], "line": kept[idx]["line"], "category": kept[idx]["row"].get("category")}
            for idx in sorted(drop_idx, reverse=True)
        ]
        kept = [entry for idx, entry in enumerate(kept) if idx not in drop_idx]
        print(f"dropped {len(dropped_near)} near-duplicate rows; {len(kept)} remain")
    for pair in near_pairs[: args.list]:
        print(
            f"near-dupe ({pair['ratio']:.3f}) [{pair['category']}] "
            f"{pair['a']['file']}:{pair['a']['line']} <-> {pair['b']['file']}:{pair['b']['line']}"
        )
    # 4) document the input files' relationship (the two default files first)
    by_file = {info["path"]: [e for e in all_entries if e["path"] == info["path"]] for info in inputs}
    relationships = []
    for i, info_a in enumerate(inputs):
        for info_b in inputs[i + 1 :]:
            relation = _pair_relationship(by_file[info_a["path"]], by_file[info_b["path"]])
            relation["files"] = [os.path.basename(info_a["path"]), os.path.basename(info_b["path"])]
            relationships.append(relation)
            print(f"relationship {relation['files']}: {relation['relationship']}")

    # 5) write the canonical deduplicated union + machine-readable report
    output_path = args.out or None
    out_info = None
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            for entry in kept:
                fh.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
        out_info = {"path": output_path, "rows": len(kept), "category_counts": dict(
            Counter(entry["row"].get("category") for entry in kept)
        )}
        print(f"wrote {len(kept)} rows -> {output_path}")
    report = {
        "generated_by": "finetune.dedupe_data",
        "inputs": inputs,
        "input_relationships": relationships,
        "exact_duplicate_rows_dropped": dropped_exact,
        "exact_duplicates_total": len(dropped_exact),
        "near_duplicate_pairs": near_pairs,
        "near_duplicates_total": len(near_pairs),
        "near_duplicate_rows_dropped": dropped_near,
        "invalid_rows_dropped": invalid,
        "options": {
            "near_threshold": args.near_threshold,
            "jaccard_gate": args.jaccard_gate,
            "drop_near": args.drop_near,
            "drop_invalid": args.drop_invalid,
        },
        "output": out_info,
    }
    if near_pairs and not args.drop_near:
        print("note: near-duplicate pairs are reported but kept; rerun with --drop-near to remove the b-side")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"wrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


