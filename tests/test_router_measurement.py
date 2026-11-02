"""Tests for the router measurement harness."""

import unittest
from eval.measure_router import SimpleTfidfNaiveBayes, REAL_DISTRIBUTION, run_benchmark


class RouterMeasurementTests(unittest.TestCase):
    def test_real_distribution_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(REAL_DISTRIBUTION.values()), 1.0)
        self.assertEqual(len(REAL_DISTRIBUTION), 8)

    def test_small_classifier_fit_and_predict(self):
        texts = [
            "Calculate 2 + 2",
            "Solve the math problem",
            "Extract all named entities from this text",
            "Find persons and locations mentioned below",
        ]
        labels = ["math", "math", "ner", "ner"]
        clf = SimpleTfidfNaiveBayes()
        clf.fit(texts, labels)
        self.assertEqual(clf.predict("Calculate 5 * 10"), "math")
        self.assertEqual(clf.predict("Extract entities from paragraph"), "ner")

    def test_run_benchmark_lexical_beats_or_ties_classifier(self):
        results = run_benchmark()
        lexical_acc = results["Lexical Router"]["weighted_acc"]
        clf_acc = results["Small Classifier (CV)"]["weighted_acc"]
        self.assertGreaterEqual(lexical_acc, clf_acc)
        self.assertEqual(lexical_acc, 100.0)


if __name__ == "__main__":
    unittest.main()
