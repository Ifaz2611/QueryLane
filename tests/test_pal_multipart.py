"""Regression tests for multipart-math detector.

In v16, a broad regex (matching bare ',? and the') false-positived on ordinary
word-problem prose like ', and the rest on Tuesday', converting ~120-token PAL
math into ~200-token step-by-step generations and costing ~250 extra Fireworks
tokens.

The detector must strictly reject single-quantity prose with 'and the' and only
trigger when a prompt genuinely requests multiple distinct quantities.
"""

import unittest
from unittest import mock

import agent.main as main_module
from agent import pal


class MultipartMathRegressionTests(unittest.TestCase):
    """Test suite covering the v16 regression cases and multipart detection."""

    def test_single_quantity_and_the_prose_is_not_multipart(self):
        """Prose containing 'and the' must NOT be detected as multipart.

        These patterns directly caused the v16 token regression.
        """
        single_part_cases = [
            # The classic v16 trigger:
            "A store has 240 items. It sells 15% on Monday, and the rest on Tuesday. How many items were sold on Tuesday?",
            "A bakery prepared 120 loaves of bread in the morning and the remaining 80 loaves in the afternoon. How many loaves were made?",
            "Alice bought 5 apples at $2 each and the cashier gave a 10% discount. How much did she pay?",
            "A runner ran 6 miles on Saturday and the rest of his 15-mile goal on Sunday. How many miles did he run on Sunday?",
            "A school ordered 300 textbooks in September and the rest of the 500 books in October. How many books arrived in October?",
            "There were 80 people at the seminar and the organizer handed out 2 brochures per person. How many brochures were handed out in total?",
            "A cyclist traveled 25 miles in 2 hours and the return trip took 3 hours. What was the average speed for the whole trip?",
            "Bob deposited $1,000 into a savings account and the bank offered a 4% annual interest rate. How much interest is earned in one year?",
            "A recipe needs 4 cups of flour and the chef only has 2.5 cups. How many more cups are needed?",
            "The initial temperature was 15 degrees and the temperature dropped by 7 degrees by midnight. What was the midnight temperature?",
            "A train traveled 180 km in 3 hours and the second leg of 120 km took 2 hours. What was the overall average speed?",
            "A factory produced 450 units on day one and the defect rate was 2%. How many defective units were produced?",
            "The company had $50,000 in revenue and the operational expenses were $32,000. What is the net operating profit?",
            "We planted 18 trees on Saturday and the remaining 14 trees on Sunday. How many trees were planted in total?",
            "A water container holds 50 liters. 15 liters were drained in the morning, and the rest in the evening. How many liters were drained in the evening?",
        ]
        for prompt in single_part_cases:
            with self.subTest(prompt=prompt[:60]):
                self.assertFalse(
                    pal.is_multipart(prompt),
                    f"Prompt unexpectedly flagged as multipart: {prompt}",
                )

    def test_true_multipart_questions_are_detected(self):
        """Prompts genuinely requesting two quantities must be flagged as multipart."""
        multipart_cases = [
            # Second interrogatives ('and what', 'and how many/much/long/far/old')
            "What is the total price of the items, and what is the sales tax?",
            "How many apples did Alice buy, and how many oranges did Bob buy?",
            "What was the flight duration, and how long was the layover?",
            "How much does the jacket cost, and how much is the shipping fee?",
            "How far is the first checkpoint, and how far is the second checkpoint?",
            "How old is David, and how old is his younger brother?",
            # Multiple question marks
            "How many students passed the exam? How many failed?",
            "What is the radius of the circle? What is its circumference?",
            "What is the length of the rectangle? What is its width?",
            # 'what is/are the ... and the ...' construct asking for two quantities
            "What is the length of the box and the width of the box?",
            "What are the coordinates of point A and the coordinates of point B?",
            "What were the sales in March and the sales in April?",
        ]
        for prompt in multipart_cases:
            with self.subTest(prompt=prompt[:60]):
                self.assertTrue(
                    pal.is_multipart(prompt),
                    f"Prompt failed to be detected as multipart: {prompt}",
                )

    def test_case_insensitivity(self):
        """Detection should be case-insensitive."""
        self.assertTrue(
            pal.is_multipart("WHAT IS THE COST OF ADMISSION, AND WHAT IS THE PARKING FEE?")
        )
        self.assertFalse(
            pal.is_multipart(
                "A STORE HAS 200 APPLES AND THE REST ON TUESDAY. HOW MANY APPLES REMAIN?"
            )
        )

    def test_main_routing_multipart_vs_single_part(self):
        """Verify main task scheduler and answer_task logic for single vs multipart math."""
        # Single-part math with 'and the' should NOT be forced to cloud when local is targeted
        single_part = (
            "A store has 240 items. It sells 15% on Monday, and the rest on Tuesday. "
            "How many items were sold on Tuesday?"
        )
        self.assertFalse(pal.is_multipart(single_part))

        # Multipart math prompt
        multipart = "What is the speed of the car, and what is the total distance traveled?"
        self.assertTrue(pal.is_multipart(multipart))

        # Test submit_order in agent/main.py: multipart math goes to cloud priority (0, 0)
        # while single-part local math goes to local priority (1, 0)
        mock_cloud = mock.MagicMock()
        task_single = {"task_id": "math-s", "prompt": single_part}
        task_multi = {"task_id": "math-m", "prompt": multipart}

        with mock.patch("agent.main._route") as mock_route:
            mock_route.return_value = ("math", "local", 0.95)

            # For single-part, it should attempt local
            with mock.patch("agent.main.local_model") as mock_local:
                mock_local.available.return_value = True
                mock_local.wait_ready.return_value = True
                mock_local.category_enabled.return_value = True
                mock_local.answer.return_value = "Answer: 204 items"
                mock_local.answer_ok.return_value = True
                mock_local.normalize.side_effect = lambda cat, ans: ans

                res = main_module.answer_task(mock_cloud, task_single)
                self.assertEqual(res["routed"], "local")
                mock_local.answer.assert_called_once()
                mock_cloud.answer.assert_not_called()

            # For multipart, it must flip to cloud immediately
            with mock.patch("agent.main.local_model") as mock_local:
                mock_cloud.answer.return_value = "Speed: 60 km/h\nDistance: 120 km"
                mock_local.normalize.side_effect = lambda cat, ans: ans

                res = main_module.answer_task(mock_cloud, task_multi)
                self.assertEqual(res["routed"], "cloud")
                mock_local.answer.assert_not_called()
                mock_cloud.answer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
