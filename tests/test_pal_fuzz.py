"""Adversarial and fuzz regression tests for agent/pal.safe_eval.

The evaluator handles untrusted model output, so every rejected construct is
pinned here. The random fuzzer is deterministic (fixed seed) and asserts the
invariant that matters: safe_eval either raises or returns a finite, bounded
real number. It must never return a bool/string/complex/NaN/inf.
"""

import math
import random
import unittest
import warnings

from agent import pal

# Construct -> expression. Every one parses as Python but must be rejected by
# the AST whitelist or the numeric result checks.
REJECTED_CONSTRUCTS = {
    "string constant": "'a'",
    "bytes constant": "b'a'",
    "none constant": "None",
    "boolean true": "True",
    "boolean false": "False",
    "ellipsis": "...",
    "complex constant": "1j",
    "list": "[1, 2]",
    "tuple": "(1, 2)",
    "set": "{1, 2}",
    "dict": "{'a': 1}",
    "list comprehension": "[x for x in (1, 2)]",
    "set comprehension": "{x for x in (1, 2)}",
    "dict comprehension": "{x: x for x in (1, 2)}",
    "generator": "(x for x in (1, 2))",
    "attribute": "(1).real",
    "dunder attribute": "(1).__class__",
    "subscript": "[1][0]",
    "lambda": "(lambda: 1)()",
    "conditional": "1 if 1 else 2",
    "comparison": "1 < 2",
    "membership": "1 in (1, 2)",
    "identity": "1 is 1",
    "boolean and": "1 and 2",
    "boolean or": "1 or 2",
    "boolean not": "not 1",
    "bitwise and": "1 & 2",
    "bitwise or": "1 | 2",
    "bitwise xor": "1 ^ 2",
    "left shift": "1 << 2",
    "right shift": "1 >> 2",
    "bitwise invert": "~1",
    "matrix multiply": "1 @ 2",
    "walrus": "(x := 1)",
    "f-string": "f'{1}'",
    "starred call": "max(*[1, 2])",
    "keyword argument": "max(1, b=2)",
    "import call": "__import__('os')",
    "eval call": "eval('1')",
    "exec call": "exec('1')",
    "open call": "open('x')",
    "print call": "print(1)",
    "int call": "int('1')",
    "float call": "float('1')",
    "pow call": "pow(2, 3)",
    "attribute call": "math.sqrt(4)",
    "bare name": "x",
    "huge exponent": "2 ** 101",
    "negative huge exponent": "2 ** -101",
    "computed exponent": "2 ** (10 ** 2)",
    "named exponent": "2 ** x",
    "non-finite literal": "1e309",
    "negative non-finite literal": "-1e309",
    "overflowing product": "1e308 * 10",
    "out-of-range literal": "9999999999999999",
    "out-of-range power": "10 ** 16",
    "out-of-range product": "1e15 * 10",
    "complex result": "(-1) ** 0.5",
    "string multiplication": "'a' * 3",
    "oversized expression": "1+" * 200 + "1",
}

ALLOWED_CONSTRUCTS = {
    "precedence": ("2 + 3 * 4", 14),
    "negative exponent": ("2 ** -3", 0.125),
    "helpers": ("abs(-5) + max(1, 2)", 7),
    "min": ("min(1, 2, 3)", 1),
    "round": ("round(2.675, 2)", 2.67),
    "ceil": ("ceil(1.2)", 2),
    "floor": ("floor(1.8)", 1),
    "sqrt": ("sqrt(16)", 4),
    "unary plus": ("+5", 5),
    "unary minus": ("-5", -5),
    "grouped unary": ("-(2 + 3)", -5),
    "floor division": ("10 // 3", 3),
    "modulo": ("10 % 3", 1),
    "power": ("10 ** 3", 1000),
    "scientific notation": ("1e3", 1000),
    "boundary value": ("10 ** 15", 10**15),
}


class PalRejectionTests(unittest.TestCase):
    def test_every_disallowed_construct_is_rejected(self):
        for name, expression in REJECTED_CONSTRUCTS.items():
            with self.subTest(construct=name, expression=expression):
                with self.assertRaises(ValueError):
                    pal.safe_eval(expression)

    def test_rejected_constructs_never_reach_answer_formatting(self):
        for name, expression in REJECTED_CONSTRUCTS.items():
            with self.subTest(construct=name, expression=expression):
                self.assertIsNone(
                    pal.answer_from_expression(f"EXPR: {expression} | UNIT: none")
                )

    def test_arithmetic_errors_are_not_leaked_as_values(self):
        for expression, error in (
            ("1 / 0", ZeroDivisionError),
            ("1 // 0", ZeroDivisionError),
            ("1 % 0", ZeroDivisionError),
            ("sqrt(4, 2)", TypeError),
            ("ceil()", TypeError),
            ("round(1.2, 3, 4)", TypeError),
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(error):
                    pal.safe_eval(expression)

    def test_syntax_errors_raise(self):
        for expression in ("", "1 +", "(", "1 2"):
            with self.subTest(expression=expression):
                with self.assertRaises(SyntaxError):
                    pal.safe_eval(expression)

    def test_allowed_arithmetic_still_evaluates(self):
        for name, (expression, expected) in ALLOWED_CONSTRUCTS.items():
            with self.subTest(construct=name, expression=expression):
                value = pal.safe_eval(expression)
                if isinstance(expected, float):
                    self.assertAlmostEqual(value, expected)
                else:
                    self.assertEqual(value, expected)

    def test_expression_extraction_trims_trailing_prose(self):
        self.assertEqual(pal.extract_expression("EXPR: 2 + 3 items | UNIT: none"), "2 + 3")
        self.assertEqual(pal.extract_expression("EXPR: `2+2` | UNIT: none"), "2+2")
        self.assertIsNone(pal.extract_expression("no expression here"))
        self.assertIsNone(pal.extract_expression("EXPR:   | UNIT: none"))
        self.assertIsNone(pal.answer_from_expression("EXPR:   | UNIT: none"))


class PalFuzzTests(unittest.TestCase):
    # High-signal tokens: imports, attribute access, containers, operators that
    # are deliberately absent from the whitelist, and Python's truthy words.
    ALPHABET = [
        "__", "import", "os", "sys", "eval", "exec", "open", "getattr",
        "lambda", ":", ";", "[", "]", "(", ")", "{", "}", ".", ",", "'", '"',
        "1", "2", "0", "3", "+", "-", "*", "/", "**", "//", "%", "@", "&", "|",
        "^", "<<", ">>", "~", "and", "or", "not", "if", "else", "for", "in",
        "is", "=", "<", ">", "!", "True", "False", "None", "...", " ",
    ]

    def test_random_expressions_never_escape_the_numeric_sandbox(self):
        rng = random.Random(0x51AFE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for _ in range(2500):
                expression = "".join(
                    rng.choice(self.ALPHABET) for _ in range(rng.randint(1, 40))
                )
                try:
                    value = pal.safe_eval(expression)
                except Exception:
                    continue  # rejected: the only acceptable other outcome
                with self.subTest(expression=expression):
                    self.assertIsInstance(value, (int, float))
                    self.assertNotIsInstance(value, bool)
                    self.assertTrue(math.isfinite(value))
                    self.assertLessEqual(abs(value), 1e15)

    def test_random_expressions_never_break_answer_from_expression(self):
        rng = random.Random(0xC0FFEE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for _ in range(1000):
                expression = "".join(
                    rng.choice(self.ALPHABET) for _ in range(rng.randint(1, 40))
                )
                text = f"EXPR: {expression} | UNIT: none"
                result = pal.answer_from_expression(text)
                with self.subTest(text=text):
                    self.assertTrue(result is None or result.startswith("Answer: "))


if __name__ == "__main__":
    unittest.main()
