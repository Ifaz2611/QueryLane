"""Tests for the SFT data toolchain: validate_data, dedupe_data, eval_split,
and the reproducibility hooks in gen_data.

These tests are dependency-light (stdlib + agent only); no network, GPU or
model downloads. Run with the repo's usual commands:
    python -m pytest tests -q
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from finetune import dedupe_data, eval_split, gen_data, validate_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "finetune", "data")


def write_jsonl(path: str, rows: list) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for row_ in rows:
            fh.write(json.dumps(row_, ensure_ascii=False) + "\n")
    return path


def system(category: str) -> str:
    return validate_data.SYSTEM_PROMPTS[category]


def row(category: str, user: str, answer: str) -> dict:
    return {
        "category": category,
        "messages": [
            {"role": "system", "content": system(category)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def sentiment_row(review="The battery lasts two days and charges fast.", praise="lasts two days"):
    return row(
        "sentiment",
        f"Classify the sentiment of this review: {review}",
        f'Positive, the review expresses satisfaction ("{praise}") with no complaints.',
    )


def ner_row():
    return row(
        "ner",
        "Extract all named entities and their types from: Maria joined Fireworks AI in San Jose on 3 March.",
        "Maria: Person\nFireworks AI: Organization\nSan Jose: Location\n3 March: Date",
    )


def summarization_row():
    passage = "The transit project will add 12 buses and cut wait times by 30 percent, at a cost of 4 million."
    return row(
        "summarization",
        f"Summarize the following in exactly one sentence: {passage}",
        "The project adds 12 buses and cuts waits 30 percent despite 4 million in added costs.",
    )


def factual_row():
    return row(
        "factual",
        "What is FTP and where is it commonly used?",
        "FTP stands for File Transfer Protocol; it moves files between hosts over a network.",
    )


def math_row(expr="2 + 3 * 4", unit="items"):
    return row("math", "Nia buys 2 snacks and 3 drinks.", f"EXPR: {expr} | UNIT: {unit}")


def all_valid_rows():
    return [
        sentiment_row(),
        ner_row(),
        summarization_row(),
        factual_row(),
        math_row(),
    ]


def codes(issues):
    return {issue.code for issue in issues}


class ValidationTests(unittest.TestCase):
    def test_valid_rows_are_clean(self):
        for row_ in all_valid_rows():
            with self.subTest(category=row_["category"]):
                issues = validate_data.check_row(row_, line_no=1)
                self.assertEqual(issues, [], str(issues))

    def test_unknown_category_rejected(self):
        issues = validate_data.check_row({"category": "poetry", "messages": []}, line_no=7)
        self.assertIn("schema/category", codes(issues))
        self.assertEqual(issues[0].line, 7)

    def test_wrong_role_order_rejected(self):
        broken = {"category": "factual", "messages": list(reversed(factual_row()["messages"]))}
        self.assertIn("schema/roles", codes(validate_data.check_row(broken)))

    def test_empty_answer_rejected(self):
        broken = factual_row()
        broken["messages"][2]["content"] = "   "
        self.assertIn("schema/content", codes(validate_data.check_row(broken)))

    def test_system_prompt_skew_rejected(self):
        broken = factual_row()
        broken["messages"][0]["content"] += " Also be poetic."
        self.assertIn("schema/system-prompt", codes(validate_data.check_row(broken)))

    def test_sentiment_nonverbatim_quote_rejected(self):
        broken = sentiment_row(praise="charges instantly")
        self.assertIn("sentiment/verbatim-quote", codes(validate_data.check_row(broken)))

    def test_sentiment_unknown_label_rejected(self):
        broken = sentiment_row()
        broken["messages"][2]["content"] = "This review is quite good."
        self.assertIn("sentiment/label", codes(validate_data.check_row(broken)))

    def test_sentiment_neutral_must_use_exact_template(self):
        broken = sentiment_row()
        broken["messages"][2]["content"] = "Neutral review, nothing special."
        self.assertIn("sentiment/template", codes(validate_data.check_row(broken)))

    def test_ner_unknown_type_and_missing_entity_rejected(self):
        broken = ner_row()
        broken["messages"][2]["content"] = "Maria: Personx\nFireworks AI: Organization"
        self.assertIn("ner/type", codes(validate_data.check_row(broken)))
        broken2 = ner_row()
        broken2["messages"][2]["content"] = "Maria Lopez: Person"
        self.assertIn("ner/verbatim", codes(validate_data.check_row(broken2)))

    def test_ner_empty_entity_rejected(self):
        broken = ner_row()
        broken["messages"][2]["content"] = ": Person\nMaria: Person"
        self.assertIn("ner/empty-text", codes(validate_data.check_row(broken)))

    def test_summarization_unknown_constraint_rejected(self):
        broken = summarization_row()
        user = broken["messages"][1]["content"]
        broken["messages"][1]["content"] = "TL;DR in one one line: " + user.split(": ", 1)[1]
        self.assertIn("summarization/unknown-constraint", codes(validate_data.check_row(broken)))

    def test_summarization_length_violation_rejected(self):
        broken = summarization_row()
        broken["messages"][2]["content"] += " Another sentence here."
        self.assertIn("summarization/length", codes(validate_data.check_row(broken)))

    def test_summarization_hallucinated_number_rejected(self):
        broken = summarization_row()
        broken["messages"][2]["content"] = "The project adds 88 buses. Costs rose 4 million."
        found = codes(validate_data.check_row(broken))
        self.assertIn("summarization/numbers-not-in-passage", found)
        self.assertIn("summarization/length", found)  # two sentences violates 'exactly one sentence'

    def test_factual_overlong_answer_rejected(self):
        broken = factual_row()
        broken["messages"][2]["content"] = "x" * 700
        self.assertIn("factual/too-long", codes(validate_data.check_row(broken)))

    def test_math_unsafe_negative_or_malformed_rejected(self):
        cases = (
            ("EXPR: __import__('os') | UNIT: $", "math/eval"),
            ("EXPR: 3 - 10 | UNIT: items", "math/negative"),
            ("EXPR: 1e309 | UNIT: none", "math/eval"),
            ("The answer is roughly twelve.", "math/format"),
        )
        for answer, expected_code in cases:
            with self.subTest(answer=answer):
                broken = math_row()
                broken["messages"][2]["content"] = answer
                self.assertIn(expected_code, codes(validate_data.check_row(broken)))

    def test_duplicate_prompt_warning(self):
        seen = {}
        issues = validate_data.check_row(factual_row(), line_no=1, seen_prompts=seen)
        self.assertEqual([i for i in issues if i.level == "error"], [])
        duplicate = factual_row()
        duplicate["messages"][2]["content"] = "FTP is File Transfer Protocol."
        issues = validate_data.check_row(duplicate, line_no=2, seen_prompts=seen)
        self.assertIn("duplicate-prompt", codes(issues))
        self.assertEqual([i for i in issues if i.level == "error"], [])


class DedupeTests(unittest.TestCase):
    def test_across_file_exact_dupes_dropped_and_relationship_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_path = write_jsonl(os.path.join(tmp, "a.jsonl"), [sentiment_row(), factual_row(), math_row()])
            # b: same rows in a different order plus one duplicate of the first row
            b_path = write_jsonl(
                os.path.join(tmp, "b.jsonl"), [math_row(), sentiment_row(), factual_row(), sentiment_row()]
            )
            out_path = os.path.join(tmp, "dedup.jsonl")
            report_path = os.path.join(tmp, "report.json")
            rc = dedupe_data.main([a_path, b_path, "--out", out_path, "--report", report_path])
            self.assertEqual(rc, 0)
            with open(out_path, encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(out_rows), 3)  # duplicate sentiment kept once
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            # 4 drops: 3 cross-file copies of a's rows + 1 in-file duplicate in b
            self.assertEqual(report["exact_duplicates_total"], 4)
            dropped = report["exact_duplicate_rows_dropped"][0]
            self.assertEqual(dropped["file"], "b.jsonl")
            relation = report["input_relationships"][0]
            self.assertEqual(relation["common_rows"], 3)
            self.assertEqual(relation["only_in_second"], 1)

    def test_identical_permutation_files_reported_as_identical_multiset(self):
        rows = all_valid_rows()
        with tempfile.TemporaryDirectory() as tmp:
            a_path = write_jsonl(os.path.join(tmp, "a.jsonl"), rows)
            b_path = write_jsonl(os.path.join(tmp, "b.jsonl"), list(reversed(rows)))
            report_path = os.path.join(tmp, "report.json")
            dedupe_data.main([a_path, b_path, "--out", os.path.join(tmp, "dedup.jsonl"), "--report", report_path])
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            relation = report["input_relationships"][0]
            self.assertIn("identical-multiset", relation["relationship"])
            self.assertEqual(relation["only_in_second"], 0)

    def test_near_duplicate_pair_detected_and_optionally_dropped(self):
        base = row(
            "math",
            "In a trivia contest Team A scored 45 points in the opening round, doubled that in the second "
            "round, and then lost 12 points in the third round before earning a final bonus of 8 points. "
            "What was Team A's total score across all rounds?",
            "EXPR: 45 + 45 * 2 - 12 + 8 | UNIT: points",
        )
        twin = row(
            "math",
            "In a trivia contest Team A scored 45 points in the opening round, doubled that in the second "
            "round, and then lost 12 points in the third round before earning a final bonus of 9 points. "
            "What was Team A's total score across all rounds?",
            "EXPR: 45 + 45 * 2 - 12 + 9 | UNIT: points",
        )
        with tempfile.TemporaryDirectory() as tmp:
            a_path = write_jsonl(os.path.join(tmp, "a.jsonl"), [base, factual_row()])
            b_path = write_jsonl(os.path.join(tmp, "b.jsonl"), [twin])
            report_path = os.path.join(tmp, "report.json")
            dedupe_data.main([a_path, b_path, "--out", "", "--report", report_path])
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertGreaterEqual(report["near_duplicates_total"], 1)
            self.assertEqual(report["near_duplicate_pairs"][0]["category"], "math")
            out_path = os.path.join(tmp, "dedup_near.jsonl")
            dedupe_data.main([a_path, b_path, "--out", out_path, "--report", "", "--drop-near"])
            with open(out_path, encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(out_rows), 2)  # near twin (b-side) removed

    def test_drop_invalid_removes_validator_rejected_rows(self):
        broken = summarization_row()
        broken["messages"][1]["content"] = "TL;DR in one one line: " + broken["messages"][1]["content"].split(": ", 1)[1]
        with tempfile.TemporaryDirectory() as tmp:
            a_path = write_jsonl(os.path.join(tmp, "a.jsonl"), [factual_row(), broken])
            out_path = os.path.join(tmp, "dedup.jsonl")
            dedupe_data.main([a_path, "--out", out_path, "--report", "", "--drop-invalid"])
            with open(out_path, encoding="utf-8") as fh:
                out_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(out_rows), 1)
            self.assertEqual(out_rows[0]["category"], "factual")


class SplitTests(unittest.TestCase):
    def _synthetic_dataset(self):
        rows = []
        for category in ("sentiment", "ner", "summarization", "factual", "math"):
            for i in range(10):
                rows.append(row(category, f"prompt {category} variant {i}", "answer " + str(i)))
        return rows

    def test_split_is_deterministic_and_stratified(self):
        rows = self._synthetic_dataset()
        train_a, val_a = eval_split.stratified_split(rows, val_frac=0.2, seed=7)
        train_b, val_b = eval_split.stratified_split(rows, val_frac=0.2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        per_cat = {}
        for r in val_a:
            per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
        self.assertEqual(set(per_cat), {"sentiment", "ner", "summarization", "factual", "math"})
        self.assertTrue(all(1 <= n <= 3 for n in per_cat.values()))

    def test_split_honors_val_frac_and_has_no_overlap(self):
        rows = self._synthetic_dataset()
        train, val = eval_split.stratified_split(rows, val_frac=0.2, seed=1)
        self.assertEqual(len(train) + len(val), len(rows))
        train_keys = {validate_data.normalize_prompt(r["messages"][1]["content"]) for r in train}
        val_keys = {validate_data.normalize_prompt(r["messages"][1]["content"]) for r in val}
        self.assertFalse(train_keys & val_keys)
        self.assertGreaterEqual(len(val), 8)  # at least ~2 per category
        self.assertLessEqual(len(val) / len(rows), 0.3)

    def test_duplicate_prompts_never_straddle_the_split(self):
        rows = [factual_row(), factual_row(), math_row(), math_row(unit="none")]
        rows[0]["messages"][2]["content"] += " Variant wording."
        rows[2]["messages"][2]["content"] = "EXPR: 1 | UNIT: none"
        train, val = eval_split.stratified_split(rows, val_frac=0.5, seed=3)
        train_keys = {validate_data.normalize_prompt(r["messages"][1]["content"]) for r in train}
        val_keys = {validate_data.normalize_prompt(r["messages"][1]["content"]) for r in val}
        self.assertFalse(train_keys & val_keys)


class ScoringTests(unittest.TestCase):
    def test_math_value_score_ignores_expression_form(self):
        score, _ = eval_split.score_answer("math", "q", "EXPR: 40 / 0.8 | UNIT: $", "EXPR: 50 - 5 | UNIT: $")
        self.assertEqual(score, 0.0)  # 45 != 50
        score, detail = eval_split.score_answer("math", "q", "EXPR: 40 / 0.8 | UNIT: $", "EXPR: 2 * 25 | UNIT: $")
        self.assertEqual(score, 1.0)  # 50 computed two different ways
        self.assertTrue(detail["unit_ok"])
        self.assertTrue(detail["executable"])

    def test_math_wrong_value_and_unit(self):
        score, detail = eval_split.score_answer("math", "q", "EXPR: 2 + 2 | UNIT: none", "EXPR: 2 + 3 | UNIT: items")
        self.assertEqual(score, 0.0)
        self.assertTrue(detail["executable"])
        self.assertFalse(detail["unit_ok"])

    def test_sentiment_label_accuracy(self):
        score, detail = eval_split.score_answer("sentiment", "q", "Positive, x", "positive.")
        self.assertEqual(score, 1.0)
        self.assertEqual(detail["pred_label"], "positive")
        score, _ = eval_split.score_answer("sentiment", "q", "Positive, x", "Negative, y")
        self.assertEqual(score, 0.0)

    def test_ner_f1_and_exact_match(self):
        score, detail = eval_split.score_answer(
            "ner", "q", "Ada: Person\nIBM: Organization", "ada: person\nibm: organization"
        )
        self.assertTrue(detail["exact_match"])
        self.assertGreater(score, 0.99)
        score2, detail2 = eval_split.score_answer("ner", "q", "Ada: Person\nIBM: Organization", "Ada: Person")
        self.assertFalse(detail2["exact_match"])
        self.assertAlmostEqual(score2, 2 / 3, places=2)

    def test_summarization_compliance_and_numbers(self):
        user = "Summarize the following in exactly one sentence: Sales rose 40 percent to 9 million."
        score, detail = eval_split.score_answer(
            "summarization", user, "Sales rose 40 percent.", "Sales rose 40 percent to 9 million."
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(detail["number_recall"], 1.0)
        score2, _ = eval_split.score_answer("summarization", user, "Sales rose 40 percent.", "It went up. And again.")
        self.assertEqual(score2, 0.0)  # two sentences violate the constraint

    def test_factual_token_f1(self):
        score = eval_split.score_answer("factual", "q", "the sky is blue", "the sky is very blue today")[0]
        self.assertGreater(score, 0.4)
        self.assertLess(score, 1.0)


class GenDataTests(unittest.TestCase):
    def test_pick_model_prefers_kimi_then_explicit(self):
        self.assertEqual(gen_data._pick_model(["a/other", "b/kimi-final"], ""), "b/kimi-final")
        self.assertEqual(gen_data._pick_model(["a/other", "b/kimi"], "explicit/id"), "explicit/id")
        self.assertEqual(gen_data._pick_model(["a/only"], ""), "a/only")

    def test_gen_prompt_is_rng_deterministic(self):
        import random

        args_a = gen_data._gen_prompt("summarization", "a research finding", 10, random.Random("42:summarization:0"))
        args_b = gen_data._gen_prompt("summarization", "a research finding", 10, random.Random("42:summarization:0"))
        args_c = gen_data._gen_prompt("summarization", "a research finding", 10, random.Random("42:summarization:1"))
        self.assertEqual(args_a, args_b)
        self.assertNotEqual(args_a, args_c)
        self.assertIn("STRICT JSON", args_a)

    def test_manifest_records_model_and_outputs_without_secrets(self):
        import shutil
        import tempfile

        class FakeArgs:
            per_category = 10
            categories = "math"
            temperature = 0.9
            seed = 1

        with tempfile.TemporaryDirectory() as tmp:
            math_path = os.path.join(DATA_DIR, "math.jsonl")
            if os.path.exists(math_path):
                shutil.copy(math_path, os.path.join(tmp, "math.jsonl"))
            else:
                write_jsonl(os.path.join(tmp, "math.jsonl"), [math_row()])
            path = gen_data._write_manifest(
                tmp, "test/model", "https://api.example.test/v1", FakeArgs(), {"math": 10}, 10, 999
            )
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["model"], {"id": "test/model", "base_url_host": "api.example.test"})
            self.assertEqual(manifest["params"]["seed"], 1)
            self.assertIn("sha256", manifest["generator"])
            self.assertIn("math.jsonl", manifest["outputs"])
            self.assertNotIn("FIREWORKS_API_KEY", json.dumps(manifest))  # never leak secrets


class ShippedDataTests(unittest.TestCase):
    """Regression guards over the real dataset files (cheap: no near-scan)."""

    def test_train_all_round2_are_identical_multisets(self):
        import collections

        def load(name):
            with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]

        def sig(r):
            return json.dumps(r, sort_keys=True, ensure_ascii=False)

        counter_a = collections.Counter(map(sig, load("train_all.jsonl")))
        counter_b = collections.Counter(map(sig, load("train_round2.jsonl")))
        self.assertEqual(counter_a, counter_b)

    def test_shipped_train_all_has_only_the_documented_findings(self):
        rows, parse_issues = validate_data.load_jsonl(os.path.join(DATA_DIR, "train_all.jsonl"))
        self.assertEqual(parse_issues, [])
        allowed_errors = {"summarization/numbers-not-in-passage", "summarization/unknown-constraint"}
        allowed_warnings = {"duplicate-prompt", "ner/document-as-organization"}
        for issue in validate_data.validate_rows(rows):
            if issue.level == "error":
                self.assertIn(issue.code, allowed_errors, str(issue))
            else:
                self.assertIn(issue.code, allowed_warnings, str(issue))

    def test_deduped_artifact_is_clean(self):
        path = os.path.join(DATA_DIR, "train_dedup.jsonl")
        if not os.path.exists(path):
            self.skipTest("train_dedup.jsonl not built yet (run python -m finetune.dedupe_data)")
        rows, parse_issues = validate_data.load_jsonl(path)
        issues = validate_data.validate_rows(rows)
        self.assertEqual(parse_issues, [])
        self.assertEqual([i for i in issues if i.level == "error"], [], str(issues[:5]))
        self.assertEqual([i for i in issues if i.code == "duplicate-prompt"], [])
        # 2819 unique rows - 2 invalid rows - 8 duplicate prompts - 34 near-duplicate rows
        self.assertEqual(len(rows), 2775)

    def test_dedupe_report_documents_the_relationship(self):
        path = os.path.join(DATA_DIR, "dedupe_report.json")
        if not os.path.exists(path):
            self.skipTest("dedupe_report.json not built yet (run python -m finetune.dedupe_data)")
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        relation = report["input_relationships"][0]
        self.assertIn("identical-multiset", relation["relationship"])
        self.assertEqual(report["input_relationships"][0]["only_in_second"], 0)
        self.assertEqual(report["output"]["rows"], 2775)


if __name__ == "__main__":
    unittest.main()





