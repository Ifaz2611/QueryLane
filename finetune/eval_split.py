"""Held-out validation split + deterministic per-category metrics for SFT runs.

The historical evaluation was a single LLM judge run; several scores flipped
between reruns. This module makes model selection reproducible:

- ``stratified_split`` carves out a deterministic, category-stratified
  validation split. Rows sharing a normalized prompt (near-duplicates) are
  grouped, so the same prompt can never leak across the split boundary.
- ``per_token_eval_loss`` computes held-out loss per category, masked to the
  assistant's answer tokens (the tokens actually trained on), in a plain
  no-grad forward loop over batches.
- ``generate_answers`` + ``score_answer`` produce deterministic, judge-free
  accuracy metrics per category:
    sentiment       label accuracy (first-word label of the answer)
    ner             span/type F1 and exact set-match rate
    summarization   constraint-compliance rate + number retention vs gold
    factual         token-overlap F1 vs the gold answer
    math            expression executes and matches the gold value/unit
- ``evaluate_model`` ties it together, prints a table and writes
  ``val_metrics.json`` next to the checkpoints.

Only stdlib and finetune.validate_data are imported at module load so the
test suite (and CI) can use the split/scoring logic without torch; torch and
transformers are imported lazily inside the functions that need them.
"""

import json
import os
import random
import re
from collections import Counter

from agent import pal

try:
    from finetune.validate_data import (
        extract_numbers,
        normalize_prompt,
        parse_math_answer,
        parse_ner_answer,
        sentiment_answer_label,
        split_summary_user,
        summary_satisfies_constraint,
    )
except ImportError:  # flat layout on the pod: sibling module next to train.py
    from validate_data import (
        extract_numbers,
        normalize_prompt,
        parse_math_answer,
        parse_ner_answer,
        sentiment_answer_label,
        split_summary_user,
        summary_satisfies_constraint,
    )

# ChatML markers used by the Qwen template family; the assistant span is
# located by marker search first, with prefix-render comparison as fallback.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"
_EOS_MARKER = "<|im_end|>"

METRIC_NAMES = {
    "sentiment": "label_accuracy",
    "ner": "span_f1",
    "summarization": "constraint_compliance",
    "factual": "token_f1",
    "math": "value_accuracy",
}


# --------------------------------------------------------------------------- #
# Splitting                                                                   #
# --------------------------------------------------------------------------- #
def stratified_split(rows, val_frac: float = 0.05, seed: int = 42) -> tuple:
    """Deterministic category-stratified split; near-duplicate prompts stay
    on the same side. Returns (train_rows, val_rows)."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1): {val_frac}")
    # Group indices by (category, normalized prompt) so duplicate prompts
    # (including straight/curly apostrophe variants) cannot straddle splits.
    groups = {}
    for index, row in enumerate(rows):
        key = (row["category"], normalize_prompt(row["messages"][1]["content"]))
        groups.setdefault(key, []).append(index)
    per_category = {}
    for (category, _prompt), indices in groups.items():
        per_category.setdefault(category, []).append(indices)

    rng = random.Random(f"split:{seed}")
    train_indices, val_indices = [], []
    for category in sorted(per_category):
        index_groups = per_category[category]
        rng.shuffle(index_groups)
        total = sum(len(g) for g in index_groups)
        target = round(total * val_frac)
        taken = 0
        for group in index_groups:
            if taken < target:
                val_indices.extend(group)
                taken += len(group)
            else:
                train_indices.extend(group)
    return ([rows[i] for i in sorted(train_indices)], [rows[i] for i in sorted(val_indices)])


# --------------------------------------------------------------------------- #
# Deterministic scoring (judge-free)                                          #
# --------------------------------------------------------------------------- #
def _content_tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = _content_tokens(pred), _content_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _ner_set(text: str) -> set:
    return {
        (" ".join(entity.split()).casefold(), ent_type.casefold())
        for entity, ent_type in parse_ner_answer(text)
        if entity
    }


def _ner_f1(pred: str, gold: str) -> tuple:
    pred_set, gold_set = _ner_set(pred), _ner_set(gold)
    if not pred_set and not gold_set:
        return 1.0, True
    if not pred_set or not gold_set:
        return 0.0, pred_set == gold_set
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 0.0 if not common else 2 * precision * recall / (precision + recall)
    return f1, pred_set == gold_set


def _eval_number(expression: str):
    try:
        return float(pal.safe_eval(expression))
    except Exception:
        return None


def score_answer(category: str, user: str, gold: str, pred: str) -> tuple:
    """Deterministic score in [0, 1] plus detail dict for one (gold, pred) pair."""
    if category == "sentiment":
        gold_label = sentiment_answer_label(gold)
        pred_label = sentiment_answer_label(pred)
        return float(pred_label is not None and pred_label == gold_label), {
            "gold_label": gold_label,
            "pred_label": pred_label,
        }
    if category == "ner":
        f1, exact = _ner_f1(pred, gold)
        return f1, {"exact_match": exact}
    if category == "summarization":
        constraint, _passage = split_summary_user(user)
        compliant = constraint is not None and summary_satisfies_constraint(constraint, pred.strip())
        gold_numbers = extract_numbers(gold)
        kept = [n for n in gold_numbers if n in pred]
        recall = (len(kept) / len(gold_numbers)) if gold_numbers else 1.0
        return float(compliant), {"number_recall": recall, "constraint": constraint}
    if category == "math":
        gold_expr, gold_unit = parse_math_answer(gold)
        pred_expr, pred_unit = parse_math_answer(pred)
        gold_value = _eval_number(gold_expr) if gold_expr else None
        pred_value = _eval_number(pred_expr) if pred_expr else None
        value_ok = (
            gold_value is not None
            and pred_value is not None
            and abs(pred_value - gold_value) <= max(1e-6, abs(gold_value) * 1e-6)
        )
        return float(value_ok), {
            "executable": pred_value is not None,
            "unit_ok": (gold_unit or "none") == (pred_unit or "none"),
        }
    if category == "factual":
        return _token_f1(pred, gold), {}
    raise ValueError(f"unknown category: {category}")


def score_rows(rows, predictions) -> dict:
    """Aggregate deterministic scores per category over a list of predictions."""
    per_category = {}
    for row, pred in zip(rows, predictions):
        category = row["category"]
        score, detail = score_answer(category, row["messages"][1]["content"], row["messages"][2]["content"], pred)
        per_category.setdefault(category, []).append((score, detail))
    summary = {}
    for category, scored in sorted(per_category.items()):
        details = [detail for _score, detail in scored]
        entry = {"n": len(scored), METRIC_NAMES[category]: sum(s for s, _ in scored) / len(scored)}
        if category == "ner":
            entry["exact_match_rate"] = sum(1 for d in details if d["exact_match"]) / len(scored)
        if category == "summarization":
            entry["number_recall"] = sum(d["number_recall"] for d in details) / len(scored)
        if category == "math":
            entry["executable_rate"] = sum(1 for d in details if d["executable"]) / len(scored)
            entry["unit_accuracy"] = sum(1 for d in details if d["unit_ok"]) / len(scored)
        summary[category] = entry
    return summary


# --------------------------------------------------------------------------- #
# torch / transformers-backed evaluation (imported lazily)                    #
# --------------------------------------------------------------------------- #
def _render_prompt(tokenizer, messages, no_thinking: bool = False) -> str:
    kwargs = {"enable_thinking": False} if no_thinking else {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def _assistant_example(row, tokenizer, max_length: int) -> dict:
    """Tokenize one chat row with labels masked to the assistant answer tokens.

    The assistant span is located by ChatML marker search; if a template does
    not use those markers, the span is located by comparing the rendered
    prompt prefix. This way the reported loss measures the tokens the
    fine-tune actually optimizes (the answer), not the prompt echo.
    """
    full = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    labels = [-100] * len(input_ids)

    content_start = content_end = None
    marker_pos = full.find(_ASSISTANT_MARKER)
    if marker_pos != -1:
        content_start = marker_pos + len(_ASSISTANT_MARKER)
        eos_pos = full.find(_EOS_MARKER, content_start)
        content_end = eos_pos if eos_pos != -1 else len(full)
    else:
        prompt_text = _render_prompt(tokenizer, row["messages"][:2], no_thinking=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] == prompt_ids:
            content_start = len(prompt_ids)
            content_end = len(input_ids)
    if content_start is None:  # exotic template: degrade to full-sequence loss
        content_start, content_end = 0, len(input_ids)

    for idx, (start, end) in enumerate(offsets):
        if end > content_start and start < content_end and end > start:
            labels[idx] = input_ids[idx]
    return {"input_ids": input_ids, "labels": labels, "category": row["category"]}


def per_token_eval_loss(model, tokenizer, rows, batch_size: int = 8, max_length: int = 1024) -> tuple:
    """Mean per-token NLL over assistant answer tokens, per category.

    Returns ({category: {"loss", "rows", "answer_tokens"}}, overall_loss).
    """
    import torch

    examples = [_assistant_example(row, tokenizer, max_length) for row in rows]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    sums, totals, rows_count = {}, {}, Counter()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            width = max(len(ex["input_ids"]) for ex in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for i, ex in enumerate(batch):
                length = len(ex["input_ids"])
                input_ids[i, :length] = torch.tensor(ex["input_ids"], dtype=torch.long)
                attention[i, :length] = 1
                labels[i, :length] = torch.tensor(ex["labels"], dtype=torch.long)
            input_ids, attention, labels = input_ids.to(device), attention.to(device), labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            mask = shift_labels != -100
            for i, ex in enumerate(batch):
                token_loss = losses[i][mask[i]]
                count = int(token_loss.numel())
                if not count:
                    continue
                sums[ex["category"]] = sums.get(ex["category"], 0.0) + float(token_loss.sum())
                totals[ex["category"]] = totals.get(ex["category"], 0) + count
                rows_count[ex["category"]] += 1
    if was_training:
        model.train()

    per_category = {}
    grand_sum = grand_tokens = 0
    for category in sorted(sums):
        tokens = totals[category]
        per_category[category] = {
            "loss": sums[category] / tokens,
            "rows": rows_count[category],
            "answer_tokens": tokens,
        }
        grand_sum += sums[category]
        grand_tokens += tokens
    overall = (grand_sum / grand_tokens) if grand_tokens else None
    return per_category, overall


def generate_answers(
    model, tokenizer, rows, max_new_tokens: int = 256, batch_size: int = 8, no_thinking: bool = True
) -> list:
    """Greedy generation for each row's prompt; returns answers in row order."""
    import torch

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    original_padding = tokenizer.padding_side
    was_training = model.training
    model.eval()
    # Longest-first batching with left padding keeps generation positions sane
    # and the batches tight; answers are restored to row order afterwards.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["messages"][1]["content"]), reverse=True)
    answers = [""] * len(rows)
    try:
        tokenizer.padding_side = "left"
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                prompts = [_render_prompt(tokenizer, rows[i]["messages"][:2], no_thinking) for i in indices]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                )
                new_tokens = generated[:, encoded["input_ids"].shape[1]:]
                for index, text in zip(indices, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
                    answers[index] = text.strip()
    finally:
        tokenizer.padding_side = original_padding
        if was_training:
            model.train()
    return answers


def evaluate_model(
    model,
    tokenizer,
    rows,
    out_path=None,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    loss_batch_size: int = 8,
    max_length: int = 1024,
    no_thinking: bool = True,
) -> dict:
    """Per-category held-out loss + deterministic accuracy for ``rows``.

    Writes ``out_path`` when given and returns the merged metrics dict. The
    loss loop and the generation loop fail soft: if one errors out (e.g. OOM),
    the other's numbers are still reported under ``loss_error``/
    ``generation_error``.
    """
    metrics = {"n_rows": len(rows), "per_category": {}}
    try:
        losses, overall = per_token_eval_loss(
            model, tokenizer, rows, batch_size=loss_batch_size, max_length=max_length
        )
    except Exception as exc:
        losses, overall = {}, None
        metrics["loss_error"] = str(exc)
    try:
        predictions = generate_answers(
            model, tokenizer, rows, max_new_tokens=max_new_tokens, batch_size=batch_size, no_thinking=no_thinking
        )
        metrics["accuracy"] = score_rows(rows, predictions)
    except Exception as exc:
        metrics["generation_error"] = str(exc)

    for category in sorted(set(losses) | set(metrics.get("accuracy", {}))):
        entry = dict(losses.get(category, {}))
        entry.update(metrics.get("accuracy", {}).get(category, {}))
        metrics["per_category"][category] = entry
    if overall is not None:
        metrics["loss"] = overall

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"wrote {out_path}")

    header = f"{'category':<14}{'n':>5}  {'loss':>8}  {'metric':>24}"
    print(header)
    print("-" * len(header))
    for category, entry in metrics["per_category"].items():
        loss_text = f"{entry['loss']:.4f}" if "loss" in entry else "-"
        metric_name = METRIC_NAMES.get(category, "?")
        value = entry.get(metric_name)
        value_text = f"{value:.3f} ({metric_name})" if value is not None else "-"
        rows_n = entry.get("n", entry.get("rows", 0))
        print(f"{category:<14}{rows_n:>5}  {loss_text:>8}  {value_text:>24}")
    if metrics.get("loss") is not None:
        print(f"{'overall':<14}{len(rows):>5}  {metrics['loss']:>8.4f}")
    return metrics




