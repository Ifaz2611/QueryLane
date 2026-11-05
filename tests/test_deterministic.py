"""Tests for the deterministic dev-set grader (eval/deterministic.py)."""

import json
import os
import unittest

from eval import deterministic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_PATH = os.path.join(ROOT, "eval", "tasks_dev.json")
CHECKS_PATH = os.path.join(ROOT, "eval", "checks_dev.json")

REFERENCE_ANSWERS = {
    "fact-01": "Tokyo, on the island of Honshu.",
    "fact-02": (
        "Photosynthesis converts light energy into chemical energy. Plants use "
        "carbon dioxide and water to make glucose and release oxygen."
    ),
    "fact-03": (
        "HyperText Transfer Protocol. It is used to transfer web data between "
        "clients and servers."
    ),
    "fact-04": "George Orwell wrote 1984, published in 1949.",
    "fact-05": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
    "math-01": "Answer: 144 items",
    "math-02": "Answer: 72 km/h",
    "math-03": "Answer: $50",
    "math-04": "Answer: $75",
    "math-05": "Answer: 750 grams",
    "sent-01": "Mixed. The battery life is great, but the screen scratches easily.",
    "sent-02": "Positive. They love the blender, the ice crushing, and the easy cleanup.",
    "sent-03": "Negative. The package arrived late and the box was crushed.",
    "sent-04": "Neutral. The review is factual and expresses neither praise nor complaint.",
    "sent-05": (
        "Mixed. The room was spotless and the staff friendly, but the wifi was "
        "unreliable."
    ),
    "summ-01": (
        "The council approved a six-month bicycle lane pilot on two downtown "
        "streets by a 7-4 vote."
    ),
    "summ-02": "New battery charges fast, lasts long, aiding electric vehicles.",
    "summ-03": (
        "Remote and hybrid work has become widespread. Cities, landlords, and "
        "transit systems are adapting to the change."
    ),
    "summ-04": (
        "Honeybees use the waggle dance to communicate the direction and "
        "distance of food."
    ),
    "summ-05": (
        "Revenue rose 12% on subscriptions while hardware fell 8% and margin "
        "stayed near 14%."
    ),
    "ner-01": (
        "Maria Sanchez: Person\nFireworks AI: Organization\nBerlin: Location\n"
        "last March: Date"
    ),
    "ner-02": (
        "14 February 2020: Date\nSarah Connor: Person\nToronto: Location\n"
        "Zurich: Location\nNestle: Organization"
    ),
    "ner-03": (
        "Apple: Organization\nVision Pro: Product\nWWDC: Event\n"
        "Cupertino: Location\nTim Cook: Person"
    ),
    "ner-04": (
        "Treaty of Versailles: Event\nFrance: Location\n1919: Date\n"
        "Germany: Location\nAllied Powers: Organization"
    ),
    "ner-05": (
        "Amina Yusuf: Person\nWorld Health Organization: Organization\n"
        "Geneva: Location\n3 June: Date"
    ),
    "logic-01": "Jo owns the dog and Sam cannot own the bird. Answer: Sam owns the cat.",
    "logic-02": (
        "Anna is before Ben, Carl is not last, and Ben is not second. "
        "Answer: Ben finished last."
    ),
    "logic-03": "Answer: Yellow, red, blue, green.",
    "logic-04": "Omar > Maya > Nina > Petra. Answer: Maya.",
    "logic-05": "Answer: Hana sits in seat 2.",
    "gen-01": (
        "def second_largest(values):\n"
        "    unique = sorted(set(values), reverse=True)\n"
        "    return unique[1] if len(unique) > 1 else None\n"
    ),
    "gen-02": (
        "def is_palindrome(text):\n"
        "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n"
    ),
    "gen-03": (
        "def fizzbuzz(n):\n"
        "    result = []\n"
        "    for value in range(1, n + 1):\n"
        "        if value % 15 == 0:\n"
        "            result.append('FizzBuzz')\n"
        "        elif value % 3 == 0:\n"
        "            result.append('Fizz')\n"
        "        elif value % 5 == 0:\n"
        "            result.append('Buzz')\n"
        "        else:\n"
        "            result.append(str(value))\n"
        "    return result\n"
    ),
    "gen-04": (
        "def merge(left, right):\n"
        "    out = []\n"
        "    i = j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            out.append(left[i]); i += 1\n"
        "        else:\n"
        "            out.append(right[j]); j += 1\n"
        "    return out + left[i:] + right[j:]\n"
    ),
    "gen-05": (
        "def word_counts(sentence):\n"
        "    counts = {}\n"
        "    for word in sentence.lower().split():\n"
        "        counts[word] = counts.get(word, 0) + 1\n"
        "    return counts\n"
    ),
}


class DeterministicGraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TASKS_PATH, encoding="utf-8") as fh:
            cls.tasks = json.load(fh)
        cls.by_id = {task["task_id"]: task for task in cls.tasks}
        with open(CHECKS_PATH, encoding="utf-8") as fh:
            cls.checks = json.load(fh)

    def evaluate(self, task_id, answer):
        return deterministic.evaluate(self.by_id[task_id], answer)

    def test_checks_cover_every_task_except_judge_only_code_debug(self):
        judge_only = {
            task["task_id"]
            for task in self.tasks
            if task["category"] == "code_debug"
        }
        self.assertEqual(set(self.checks), set(self.by_id) - judge_only)
        self.assertTrue(judge_only)  # keep the fallback exercised

    def test_all_deterministic_reference_answers_pass(self):
        for task_id, answer in REFERENCE_ANSWERS.items():
            with self.subTest(task_id=task_id):
                verdict = self.evaluate(task_id, answer)
                if task_id.startswith("gen-"):
                    # Structural checks prove the code exists and parses, but
                    # correctness still belongs to the LLM judge.
                    self.assertIsNone(verdict.passed)
                    self.assertFalse(verdict.authoritative)
                else:
                    self.assertTrue(verdict.passed, verdict.reason)
                    self.assertTrue(verdict.authoritative)

    def test_empty_answer_always_fails(self):
        for task_id in ("fact-01", "gen-01", "dbg-01"):
            with self.subTest(task_id=task_id):
                verdict = self.evaluate(task_id, "   ")
                self.assertFalse(verdict.passed)
                self.assertTrue(verdict.authoritative)

    def test_sentiment_requires_the_exact_label_and_justification(self):
        self.assertTrue(self.evaluate("sent-01", REFERENCE_ANSWERS["sent-01"]).passed)
        wrong = self.evaluate(
            "sent-01", "Positive. The battery is great and the screen is fine."
        )
        self.assertFalse(wrong.passed)
        self.assertIn("not accepted", wrong.reason)

        unjustified = self.evaluate("sent-02", "Positive.")
        self.assertFalse(unjustified.passed)
        self.assertIn("justification", unjustified.reason)

    def test_sentiment_accepts_both_polarities_as_mixed(self):
        verdict = self.evaluate(
            "sent-01",
            "Positive about the battery, negative about the screen; the review "
            "is two-sided.",
        )
        self.assertTrue(verdict.passed, verdict.reason)

    def test_summarization_sentence_and_word_counts_are_exact(self):
        too_long = self.evaluate(
            "summ-01", REFERENCE_ANSWERS["summ-01"] + " Another sentence."
        )
        self.assertFalse(too_long.passed)
        self.assertIn("sentence", too_long.reason)

        eleven_words = self.evaluate(
            "summ-02",
            "A new battery charges very quickly and lasts much longer for cars.",
        )
        self.assertFalse(eleven_words.passed)
        self.assertIn("at most 10 words", eleven_words.reason)

    def test_ner_requires_every_entity_and_an_accepted_type(self):
        missing = self.evaluate(
            "ner-01",
            "Maria Sanchez: Person\nFireworks AI: Organization\nlast March: Date",
        )
        self.assertFalse(missing.passed)
        self.assertIn("Berlin", missing.reason)

        mistyped = self.evaluate(
            "ner-01",
            "Maria Sanchez: Person\nFireworks AI: Organization\n"
            "Berlin: Organization\nlast March: Date",
        )
        self.assertFalse(mistyped.passed)

        misc_is_fine_for_products = self.evaluate(
            "ner-03",
            "Apple: Organization\nVision Pro: Misc\nWWDC: Misc\n"
            "Cupertino: Location\nTim Cook: Person",
        )
        self.assertTrue(misc_is_fine_for_products.passed, misc_is_fine_for_products.reason)

    def test_math_requires_the_exact_number_and_unit(self):
        self.assertTrue(self.evaluate("math-01", "Answer: 144 items").passed)
        self.assertFalse(self.evaluate("math-01", "Answer: 143 items").passed)
        self.assertFalse(self.evaluate("math-01", "Answer: 144").passed)

    def test_logic_checks_only_the_final_answer_span(self):
        # The reasoning mentions both names, but the conclusion is wrong.
        verdict = self.evaluate(
            "logic-01",
            "Sam cannot own the bird and Jo owns the dog.\nAnswer: Lee owns the cat.",
        )
        self.assertFalse(verdict.passed)

    def test_code_structural_failures_are_deterministic(self):
        missing = self.evaluate("gen-01", "Use a loop and return the second value.")
        self.assertFalse(missing.passed)
        self.assertTrue(missing.authoritative)

        wrong_name = self.evaluate("gen-03", "def fizz(n):\n    return []\n")
        self.assertFalse(wrong_name.passed)
        self.assertIn("fizzbuzz", wrong_name.reason)

    def test_unknown_task_returns_undecided(self):
        verdict = deterministic.evaluate(
            {"task_id": "custom-1", "category": "factual"}, "some answer"
        )
        self.assertIsNone(verdict.passed)
        self.assertFalse(verdict.authoritative)

    def test_fallback_check_lookup_by_task_id(self):
        # Dashboard callers often pass slim task dicts without the check inline.
        task = {"task_id": "math-02", "category": "math"}
        verdict = deterministic.evaluate(task, "Answer: 72 km/h")
        self.assertTrue(verdict.passed, verdict.reason)


if __name__ == "__main__":
    unittest.main()
