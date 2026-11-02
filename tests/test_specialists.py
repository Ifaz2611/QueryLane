"""Regression tests for the NER and sentiment specialists.

The ONNX weights are optional, so these tests fake tokenizer output and
classifier decisions. They cover the precision gates that decide whether the
specialist answers or escalates to cloud.
"""

import os
import re
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
except ImportError:  # numpy is an optional specialist dependency
    np = None

from agent import specialists

# A four-class model schema, as shipped by bert-base-NER.
ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def _encode(passage, token_labels, confidence=0.99):
    """Build the (encoding, ids, logits, id2label) tuple `_run` returns."""
    tokens = list(re.finditer(r"\S+", passage))
    encoding = SimpleNamespace(offsets=[(m.start(), m.end()) for m in tokens])
    ids = list(range(len(tokens)))
    logits = []
    n_labels = len(ID2LABEL)
    target = np.log(confidence * (n_labels - 1) / (1.0 - confidence))
    for index in range(len(tokens)):
        row = np.zeros(n_labels)
        label = token_labels.get(index, "O")
        row[[k for k, v in ID2LABEL.items() if v == label][0]] = target
        logits.append(row)
    return encoding, ids, np.array(logits), ID2LABEL


class NerEnableTests(unittest.TestCase):
    def test_enabled_by_default_when_model_is_available(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            specialists, "available", return_value=True
        ):
            os.environ.pop("NER_SPECIALIST", None)
            self.assertTrue(specialists.ner_enabled())

    def test_explicit_off_still_disables(self):
        with mock.patch.dict(
            os.environ, {"NER_SPECIALIST": "off"}, clear=False
        ), mock.patch.object(specialists, "available", return_value=True):
            self.assertFalse(specialists.ner_enabled())

    def test_missing_model_disables(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            specialists, "available", return_value=False
        ):
            os.environ.pop("NER_SPECIALIST", None)
            self.assertFalse(specialists.ner_enabled())


@unittest.skipIf(np is None, "numpy is an optional specialist dependency")
class NerDecodeTests(unittest.TestCase):
    passage = "Maria joined Acme Corp in Berlin on March 3, 2024."

    def _entities(self, *, token_labels, passage=None, confidence=0.99):
        text = passage if passage is not None else self.passage
        encoding, ids, logits, id2label = _encode(text, token_labels, confidence)
        return specialists._decode_entities(
            text, encoding, ids, logits, id2label
        )

    def test_maps_model_tags_to_task_schema_and_adds_dates(self):
        entities = self._entities(
            token_labels={
                0: "B-PER",
                2: "B-ORG",
                3: "I-ORG",
                5: "B-LOC",
            }
        )
        self.assertIsNotNone(entities)
        self.assertEqual(
            [(text, type_name) for text, type_name, _, _ in entities],
            [
                ("Maria", "Person"),
                ("Acme Corp", "Organization"),
                ("Berlin", "Location"),
                ("March 3, 2024", "Date"),
            ],
        )

    def test_low_confidence_token_escalates(self):
        self.assertIsNone(
            self._entities(
                token_labels={0: "B-PER", 2: "B-ORG", 3: "I-ORG", 5: "B-LOC"},
                confidence=0.55,
            )
        )

    def test_missing_candidate_span_escalates(self):
        # Berlin is capitalized but the model never tags it, so the answer
        # would be incomplete; the completeness guard must escalate.
        self.assertIsNone(
            self._entities(token_labels={0: "B-PER", 2: "B-ORG", 3: "I-ORG"})
        )

    def test_misc_without_a_cue_escalates(self):
        passage = "Zorblax arrived yesterday."
        self.assertIsNone(
            self._entities(token_labels={0: "B-MISC"}, passage=passage)
        )

    def test_ner_answer_formats_entity_lines(self):
        encoding, ids, logits, id2label = _encode(
            self.passage, {0: "B-PER", 2: "B-ORG", 3: "I-ORG", 5: "B-LOC"}
        )
        with mock.patch.object(
            specialists, "_run", return_value=(encoding, ids, logits, id2label)
        ):
            answer = specialists.ner_answer(
                "Extract all named entities and their types from: " + self.passage
            )
        self.assertEqual(
            answer,
            "Maria: Person\nAcme Corp: Organization\nBerlin: Location\n"
            "March 3, 2024: Date",
        )


class NerTypeTests(unittest.TestCase):
    def test_nearby_launch_words_do_not_turn_a_person_into_a_product(self):
        passage = "Aurora Retail unveiled a product in Dubai with CFO Sofia Lindqvist."
        start = passage.index("Sofia Lindqvist")
        self.assertEqual(
            specialists._infer_entity_type(
                "Sofia Lindqvist", passage, start, start + 15, "PER"
            ),
            "Person",
        )

    def test_misc_span_uses_its_own_cue_before_context(self):
        passage = "At the World AgriTech Forum, HarvestAI was unveiled as a product."
        start = passage.index("World AgriTech Forum")
        self.assertEqual(
            specialists._infer_entity_type(
                "World AgriTech Forum", passage, start, start + 19, "MISC"
            ),
            "Event",
        )

    def test_country_name_overrides_a_wrong_model_type(self):
        passage = "The office in Germany handles the account."
        start = passage.index("Germany")
        self.assertEqual(
            specialists._infer_entity_type(
                "Germany", passage, start, start + 7, "ORG"
            ),
            "Location",
        )


class SentimentTests(unittest.TestCase):
    def _answer(self, prompt, label="POSITIVE", confidence=0.99):
        with mock.patch.object(
            specialists,
            "_classify_clause",
            return_value=(label, confidence),
        ):
            return specialists.sentiment_answer(prompt)

    def test_negated_negative_is_positive(self):
        answer = self._answer(
            "Classify the sentiment of this review: not bad at all"
        )
        self.assertTrue(answer.startswith("Positive"))

    def test_negated_positive_is_negative(self):
        answer = self._answer(
            "Classify the sentiment of this review: not great",
            label="NEGATIVE",
        )
        self.assertTrue(answer.startswith("Negative"))

    def test_mixed_review_reports_both_sides(self):
        answer = self._answer(
            "Classify the sentiment of this review: "
            "Great sound but the battery dies fast."
        )
        self.assertTrue(answer.startswith("Mixed"))
        self.assertIn("Great sound", answer)
        self.assertIn("battery dies", answer)

    def test_marketing_quote_is_not_reviewer_praise(self):
        answer = self._answer(
            'Classify the sentiment of this review: '
            'The box says "premium quality" but it feels cheap.',
            label="NEGATIVE",
        )
        self.assertTrue(answer.startswith("Negative"))
        self.assertIn("cheap", answer)

    def test_quote_only_claim_escalates(self):
        self.assertEqual(
            self._answer(
                'Classify the sentiment of this review: '
                'The box says "premium quality".'
            ),
            "",
        )

    def test_endorsed_quote_counts_as_evidence(self):
        answer = self._answer(
            'Classify the sentiment of this review: '
            'The review says: "Absolutely wonderful."'
        )
        self.assertTrue(answer.startswith("Positive"))
        self.assertIn('"Absolutely wonderful."', answer)
        self.assertNotIn("The review says", answer)

    def test_hedged_review_escalates(self):
        self.assertEqual(
            self._answer(
                "Classify the sentiment of this review: "
                "It is okay, nothing special."
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
