"""Program-aided math: safely evaluate a model-emitted arithmetic expression.

The local model answers math tasks with a single line 'EXPR: <expression>'.
Evaluating ~15 output tokens beats generating ~60 chain-of-thought tokens,
which matters at the grader's low tokens-per-second. The evaluator is an AST
whitelist: numbers, arithmetic operators, and a few math helpers. Anything
else fails closed and the task escalates to cloud.
"""

import ast
import math
import re

_EXPR_LINE_RE = re.compile(r"EXPR:\s*([^|\n]+)", re.IGNORECASE)
_UNIT_RE = re.compile(r"\|\s*UNIT:\s*([^\n|]*)", re.IGNORECASE)

# A single expression answers a single quantity. Questions asking for two
# quantities must take the step-by-step path (any model, local or cloud).
# The pattern requires a second INTERROGATIVE, never a bare conjunction:
# ", and the rest on Tuesday" is normal word-problem prose (the broad v16
# pattern matched it and cost ~250 tokens on the real set).
_MULTIPART_RE = re.compile(
    r"\band (?:what|how (?:much|many|long|far|old))\b|\?[^?]*\?|"
    r"\bwhat (?:is|are|was|were) the [^.?]{0,60}\band the\b",
    re.IGNORECASE | re.DOTALL,
)


def is_multipart(prompt: str) -> bool:
    return bool(_MULTIPART_RE.search(prompt))

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
)

# Model output is untrusted. Keep the evaluator from being used as a cheap
# denial-of-service (10**10**10) or from producing non-answer values (inf,
# complex, giant integers).
_MAX_EXPRESSION_LENGTH = 300
_MAX_EXPONENT = 100
_MAX_ABS_VALUE = 1e15

_ALLOWED_NAMES = {
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def extract_expression(text: str):
    """Pull the expression out of an 'EXPR: ...' answer line, if present.
    Trailing prose is trimmed token by token until the rest parses."""
    match = _EXPR_LINE_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).replace("`", "").splitlines()[0].strip()
    tokens = candidate.split()
    while tokens:
        attempt = " ".join(tokens)
        try:
            ast.parse(attempt, mode="eval")
            return attempt
        except SyntaxError:
            tokens.pop()
    return None


def _constant_value(node):
    """Return the numeric value of a Constant or a +/- Constant node."""
    if isinstance(node, ast.Constant):
        value = node.value
    elif (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
    ):
        value = node.operand.value
        if isinstance(node.op, ast.USub):
            value = -value
    else:
        raise ValueError("non-constant exponent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric value")
    return value


def safe_eval(expression: str):
    """Evaluate a pure-arithmetic expression. Raises on anything else."""
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("disallowed function call")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("non-numeric constant")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite constant")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = _constant_value(node.right)
            if abs(exponent) > _MAX_EXPONENT:
                raise ValueError("exponent too large")
    value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric result")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite result")
    if abs(value) > _MAX_ABS_VALUE:
        raise ValueError("result out of range")
    return value


def format_number(value) -> str:
    """Render 144.0 as '144' but keep true decimals."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric result")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite result")
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{round(value, 6):g}"
    return str(value)


def extract_unit(text: str) -> str:
    match = _UNIT_RE.search(text)
    if not match:
        return ""
    unit = match.group(1).strip().strip(".")
    return unit if len(unit) <= 20 else ""


def answer_from_expression(text: str):
    """Full pipeline: 'EXPR: <expr> | UNIT: <unit>' -> 'Answer: X unit'.
    None if the expression is missing or fails the safe evaluator."""
    expression = extract_expression(text)
    if not expression:
        return None
    try:
        value = safe_eval(expression)
        number = format_number(value)
    except Exception:
        return None
    unit = extract_unit(text)
    if not unit or unit.lower() in ("none", "n/a", "-"):
        return f"Answer: {number}"
    if unit in ("$", "€", "£"):
        return f"Answer: {unit}{number}"
    return f"Answer: {number} {unit}"
