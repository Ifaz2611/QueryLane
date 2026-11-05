"""Token-cost regression: the dev set must stay under its recorded budget."""

import os
import unittest

from eval import cost_regression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS_PATH = os.path.join(ROOT, "eval", "token_budgets.json")


class TokenBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.budgets = cost_regression.load_budgets(BUDGETS_PATH)
        cls.results = cost_regression.run_all()

    def test_every_configuration_is_under_its_recorded_budget(self):
        recorded = self.budgets["configurations"]
        for name in cost_regression.CONFIGURATIONS:
            with self.subTest(configuration=name):
                self.assertIn(name, recorded)
                budget = recorded[name]["budget_tokens"]
                measured = self.results[name]["tokens"]
                self.assertLessEqual(
                    measured,
                    budget,
                    f"{name} costs {measured} tokens, over recorded budget {budget}; "
                    "a lost zero-token lane or a grown prompt is the likely cause",
                )

    def test_cheaper_lane_configurations_cost_less(self):
        cloud_only = self.results["cloud-only"]["tokens"]
        specialists = self.results["specialists"]["tokens"]
        hybrid = self.results["hybrid"]["tokens"]
        self.assertLess(specialists, cloud_only)
        self.assertLess(hybrid, specialists)

    def test_zero_token_lanes_actually_win_tasks(self):
        self.assertEqual(self.results["cloud-only"]["lanes"], {"cloud": 40})

        specialist_lanes = self.results["specialists"]["lanes"]
        self.assertEqual(specialist_lanes.get("specialist"), 10)
        self.assertEqual(specialist_lanes.get("cloud"), 30)

        hybrid_lanes = self.results["hybrid"]["lanes"]
        self.assertEqual(hybrid_lanes.get("specialist"), 10)
        # Factual routes are below the router confidence bar, so only math is
        # eligible for the local lane in the current dev set.
        self.assertEqual(hybrid_lanes.get("local"), 5)

    def test_every_task_still_gets_an_answer(self):
        for name, result in self.results.items():
            with self.subTest(configuration=name):
                self.assertEqual(result["tasks"], 40)
                self.assertGreaterEqual(result["calls"], 0)

    def test_budget_file_names_the_estimator_and_all_configurations(self):
        self.assertEqual(self.budgets["schema_version"], 1)
        self.assertIn("characters/4", self.budgets["estimator"])
        self.assertEqual(
            set(self.budgets["configurations"]),
            set(cost_regression.CONFIGURATIONS),
        )


if __name__ == "__main__":
    unittest.main()
