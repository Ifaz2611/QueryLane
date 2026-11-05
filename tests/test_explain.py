"""Tests for agent.explain and agent.web_demo."""

import json

from agent.explain import explain_decision, format_explanation, main
from agent.web_demo import HTML_TEMPLATE


def test_explain_empty():
    info = explain_decision("")
    assert info["prompt"] == ""
    assert info["predicted_lane"] == "cloud"
    assert info["category"] == "factual"


def test_explain_ner():
    prompt = "Extract all named entities: Steve Jobs founded Apple in Cupertino."
    info = explain_decision(prompt)
    assert info["category"] == "ner"
    assert info["confidence"] >= 0.70
    assert "ner" in info["signals"]
    assert any("NER" in r or "ner" in r for r in info["reasons"])


def test_explain_math_single():
    prompt = "Calculate 15 * 8 - 4"
    info = explain_decision(prompt)
    assert info["category"] == "math"
    assert not info["is_multipart_math"]
    assert any("PAL" in r for r in info["reasons"])


def test_explain_math_multipart():
    prompt = (
        "Alice has 10 apples and Bob has 5 oranges. "
        "How many apples does Alice have and how many oranges does Bob have?"
    )
    info = explain_decision(prompt)
    assert info["category"] == "math"
    assert info["is_multipart_math"]
    assert info["predicted_lane"] == "cloud"


def test_format_explanation():
    prompt = "Write a function to sort an array"
    info = explain_decision(prompt)
    text = format_explanation(info)
    assert "QUERYLANE DECISION EXPLANATION" in text
    assert info["category"] in text
    assert info["predicted_lane"].upper() in text


def test_cli_json_output(capsys):
    rc = main(["--json", "Summarize this article in 2 sentences: ..."])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["category"] == "summarization"


def test_web_demo_template():
    # verify HTML_TEMPLATE is valid and contains QueryLane
    assert "QueryLane" in HTML_TEMPLATE
    assert "/api/explain" in HTML_TEMPLATE
