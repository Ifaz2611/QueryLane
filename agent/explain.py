"""Lane decision explainer for QueryLane.

Accepts an input prompt, runs QueryLane's zero-token lexical classification,
evaluates confidence scores and runtime specialist/local gates, and provides
a clear, structured explanation of the chosen processing lane and potential
escalation triggers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from agent import local as local_model
from agent import pal, specialists
from agent.main import predict_lane
from agent.router import (
    LOCAL_CATEGORIES,
    MIN_CONFIDENCE,
    _signals,
    route_with_confidence,
)


def explain_decision(prompt: str) -> Dict[str, Any]:
    """Analyze a prompt and explain how and why QueryLane routes it."""
    clean_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not clean_prompt:
        return {
            "prompt": "",
            "category": "factual",
            "confidence": 0.0,
            "confidence_ok": False,
            "signals": {},
            "target": "cloud",
            "predicted_lane": "cloud",
            "specialist_available": False,
            "local_available": False,
            "is_multipart_math": False,
            "reasons": ["Empty prompt defaults to safest cloud fallback lane."],
            "escalation_triggers": [],
            "estimated_token_cost": "0 (empty)",
        }

    signals = _signals(clean_prompt)
    category, target, confidence = route_with_confidence(clean_prompt)
    lane = predict_lane(category, target, confidence, clean_prompt)

    reasons: List[str] = []
    triggers: List[str] = []
    is_multipart = False

    # 1. Classification rationale
    if signals:
        top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        signal_desc = ", ".join(f"{cat} ({score:.2f})" for cat, score in top_signals)
        reasons.append(f"Lexical cues matched: {signal_desc}.")
        reasons.append(
            f"Classified as '{category}' with confidence {confidence:.2f} "
            f"(threshold: {MIN_CONFIDENCE:.2f})."
        )
    else:
        reasons.append(
            f"No specific category cues fired; defaulted to '{category}' with fallback confidence {confidence:.2f}."
        )

    # 2. Confidence gating
    if confidence < MIN_CONFIDENCE:
        reasons.append(
            f"Confidence ({confidence:.2f}) is below MIN_CONFIDENCE ({MIN_CONFIDENCE:.2f}). "
            "Task is short-circuited straight to cloud to safeguard accuracy."
        )
        triggers.append("Confidence below 0.70 forced immediate cloud routing.")

    # 3. Specialist checks
    spec_avail = False
    if category == "sentiment":
        spec_avail = specialists.available("sst2")
        if spec_avail:
            reasons.append(
                "Sentiment ONNX specialist (DistilBERT SST-2) is installed and active."
            )
            triggers.append(
                "Hedging language, mixed polarity ('but' / 'although'), or neutral tone escalates to cloud."
            )
        else:
            reasons.append(
                "Sentiment ONNX specialist weights not found in models/sst2; routed to cloud."
            )
    elif category == "ner":
        spec_avail = specialists.ner_enabled()
        if spec_avail:
            reasons.append(
                "NER ONNX specialist (BERT-base NER) is installed and active."
            )
            triggers.append(
                "Uncovered capitalized spans, ambiguous date formats, or low token confidence escalate to cloud."
            )
        else:
            reasons.append(
                "NER ONNX specialist disabled or weights not found in models/ner; routed to cloud."
            )

    # 4. Math checks
    if category == "math":
        is_multipart = pal.is_multipart(clean_prompt)
        if is_multipart:
            reasons.append(
                "Prompt asks for multiple numeric quantities; single-expression PAL is bypassed for step-by-step cloud generation."
            )
        else:
            reasons.append(
                "Single-quantity math detected; cloud will use compact Program-Aided Language (PAL) evaluation."
            )
            triggers.append(
                "PAL expression syntax error or invalid operator escalates to step-by-step reasoning."
            )

    # 5. Local model checks
    local_avail = local_model.available()
    local_enabled = local_model.category_enabled(category)
    too_long = local_model.is_prompt_too_long(clean_prompt, category)
    if too_long:
        reasons.append(
            f"Prompt length ({len(clean_prompt)} chars) exceeds local model context budget ({local_model.LOCAL_N_CTX} tokens). "
            "Routed directly to cloud to prevent context overflow and wasted compute."
        )
        triggers.append("Context window capacity exceeded; bypassed local inference.")
    elif category in LOCAL_CATEGORIES:
        if local_avail and local_enabled:
            reasons.append(
                f"Category '{category}' is enabled for local GGUF execution and cleared speed threshold."
            )
            triggers.append(
                "Local inference timeout or speed falling below category threshold escalates to cloud."
            )
        else:
            reasons.append(
                f"Category '{category}' is in LOCAL_CATEGORIES, but local GGUF model is not active or below speed bar."
            )

    # 6. Final lane summary
    if lane == "specialist":
        cost_desc = "0 Fireworks tokens ($0.00) -- Zero-token local ONNX inference"
    elif lane == "local":
        cost_desc = "0 Fireworks tokens ($0.00) -- Zero-token local GGUF inference"
    else:
        if category == "math" and not is_multipart:
            cost_desc = "~15 - 25 Fireworks tokens -- Compact PAL response"
        else:
            cost_desc = "~50 - 250 Fireworks tokens -- Standard cloud generation"

    return {
        "prompt": clean_prompt,
        "category": category,
        "confidence": confidence,
        "confidence_ok": confidence >= MIN_CONFIDENCE,
        "signals": signals,
        "target": target,
        "predicted_lane": lane,
        "specialist_available": spec_avail,
        "local_available": local_avail,
        "is_multipart_math": is_multipart,
        "is_prompt_too_long": too_long,
        "reasons": reasons,
        "escalation_triggers": triggers,
        "estimated_token_cost": cost_desc,
    }


def format_explanation(info: Dict[str, Any]) -> str:
    """Format explanation dictionary into human-readable text."""
    lane = info["predicted_lane"].upper()
    cat = info["category"]
    conf = info["confidence"]

    divider = "=" * 68
    subdiv = "-" * 68

    lines = [
        divider,
        "  QUERYLANE DECISION EXPLANATION",
        divider,
        f"Prompt:     {info['prompt'][:120]}{'...' if len(info['prompt']) > 120 else ''}",
        subdiv,
        f"Category:   {cat}  (confidence: {conf:.2f} / 1.00)",
        f"Target:     {info['target']}",
        f"Lane:       [{lane}]",
        f"Cost Est.:  {info['estimated_token_cost']}",
        subdiv,
        "Decision Rationale:",
    ]

    for r in info["reasons"]:
        lines.append(f"  * {r}")

    if info["escalation_triggers"]:
        lines.append("")
        lines.append("Escalation / Fallback Guards:")
        for t in info["escalation_triggers"]:
            lines.append(f"  ! {t}")

    lines.append(divider)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for explaining query routing."""
    parser = argparse.ArgumentParser(
        prog="querylane-explain",
        description="Explain QueryLane's routing and lane selection for a prompt.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Input query or prompt to explain. If omitted, read from stdin or enter interactive mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON analysis instead of formatted text.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the lightweight local web demo.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web demo (default: 8000).",
    )

    args = parser.parse_args(argv)

    if args.web:
        from agent.web_demo import run_server

        return run_server(port=args.port)

    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("QueryLane Route Explainer (Ctrl+C to exit)")
            print("Enter prompt to analyze:")
            try:
                prompt = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return 0

    info = explain_decision(prompt)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(format_explanation(info))

    return 0


if __name__ == "__main__":
    sys.exit(main())
