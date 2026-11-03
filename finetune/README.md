# QueryLane fine-tuning

Synthetic SFT data pipeline and training entry points for the local
specialist GGUFs. Pipeline order:

```
finetune.gen_data        generate  (Fireworks, seeded, writes gen_manifest.json)
        │
finetune.validate_data   validate every JSONL row (schema + content rules)
        │
finetune.dedupe_data     dedupe exact/near-identical prompts → train_dedup.jsonl
        │
finetune.train           LoRA SFT with a held-out split; per-category
        │                loss/accuracy → val_metrics.json + train_manifest.json
        │
finetune/convert.sh      merge adapter → GGUF → Q4_K_M
```

## Dataset files (finetune/data/)

| file | rows | status |
| --- | --- | --- |
| `{category}.jsonl` ×5 | 2,819 total | generator output per category (input to the combined build) |
| `train_all.jsonl` | 2,819 | combined, shuffled (seed 42) training file; rebuilt by every `gen_data` run |
| `train_round2.jsonl` | 2,819 | **permutation of `train_all.jsonl`** — the exact same 2,819 rows in a different order (verified 2026-11-03 with multiset comparison of canonical row signatures; see `data/dedupe_report.json`) |
| `train_dedup.jsonl` | 2,775 | **canonical training file**: union of the two above, exact duplicate prompts dropped, validator-rejected rows dropped, near-duplicate (≥0.92 ratio) rows dropped |

### The train_all / train_round2 relationship (documented 2026-11-03)

`train_round2.jsonl` was produced in a second generation round and
`train_all.jsonl` is the rebuilt combined file; investigation showed the two
are content-identical (same rows, different shuffle) — there are no
round-2-only rows to merge. Retained differences between the files as
shipped:

- byte-level: different line order (and row-level formatting) only;
- both contained 8 factual prompts twice with diverging answers (the
  generator's old dedupe key compared the raw first 120 characters, which
  missed apostrophe/case/punctuation variants — e.g. `'Moby-Dick'` vs
  `"Moby-Dick"`). The generator now dedupes on the normalized full prompt;
- both contained the 2 invalid rows listed below.

**Use `train_dedup.jsonl` for training.** `train_round2.jsonl` is kept for
provenance only and should not be concatenated with anything. The dedupe
report (`data/dedupe_report.json`) records hashes, the relationship, and
every dropped row; tests guard these facts (`tests/test_finetune_data.py`).

## Validation (run before every training)

```bash
python -m finetune.validate_data                     # all of finetune/data/*.jsonl
python -m finetune.validate_data finetune/data/train_dedup.jsonl --strict
```

Checks, per row: schema (`category` in the five categories, exactly
system/user/assistant turns, non-empty contents), the system prompt must
byte-match `agent.local._LOCAL_SYSTEM` (train/serve skew), and per-category
content rules mirroring the generator's acceptance rules — sentiment label
with the fixed answer template and quotes verbatim in the review; NER
`Entity: Type` lines with allowed types and verbatim entities; summarization
constraint compliance (sentence count / word budget) plus no numbers cited
that are absent from the passage (hallucination guard); factual answer
length; math `EXPR … | UNIT …` evaluated in the same PAL sandbox
(`agent.pal.safe_eval`) used at inference, finite and non-negative.
Cross-row: duplicate (case/punctuation-insensitive) prompt warnings.
Exit code 1 on any error; `--strict` promotes warnings to failures.

Current state of the shipped data (2026-11-03): `train_all.jsonl` has
**2 errors** (line 1511: a typo'd `TL;DR in one one line:` constraint; line
1873: a Camp Fire summary citing `85` casualties the passage never states)
and **9 warnings** (8 duplicate prompts + 1 document-noun entity labeled
Organization that inference-time `normalize()` would rewrite to Event).
`train_dedup.jsonl` passes with zero errors and zero duplicate-prompt
warnings.

## Deduplication

```bash
python -m finetune.dedupe_data                    # defaults: train_all + train_round2,
                                                  # writes train_dedup.jsonl + dedupe_report.json
python -m finetune.dedupe_data a.jsonl b.jsonl --out merged.jsonl --report r.json --drop-near --drop-invalid
```

Exact dedupe key = normalized full prompt (casefolded, punctuation and
whitespace collapsed), so the `Moby-Dick`-style variants collide. Near-dupe
candidates are prefiltered by token-set overlap (same category only,
`--jaccard-gate`, default 0.5) and confirmed by a
`difflib.SequenceMatcher` ratio on the prefix-stripped prompt
(`--near-threshold`, default 0.92); the b-side of each pair is listed in the
report and dropped only with `--drop-near`. `--drop-invalid` drops rows the
validator rejects. The report also records how the input files relate as
row multisets (the shipped pair: *identical-multiset — same rows, order
differs*).

Latest run (2026-11-03): 5,638 input rows → 4 invalid dropped → 2,825 exact
duplicates dropped → 34 near-duplicate rows dropped → **2,775 rows** in
`train_dedup.jsonl`.

## Held-out validation split and per-category metrics

```bash
python -m finetune.train --model Qwen/Qwen2.5-3B-Instruct \
    --data data/train_dedup.jsonl --out /workspace/ft-qwen3b --val-frac 0.05
```

- `--val-frac` (default 0.05) holds out a deterministic,
  category-stratified split (seed `--seed`, default 42). Grouping by
  normalized prompt keeps near-duplicates on one side, so nothing leaks
  across the boundary. `--val-data` can point at an explicit JSONL instead.
- During training: per-epoch HF eval loss on the held-out rows
  (`eval_strategy=epoch`, with `eval_strategy`/`evaluation_strategy` and
  `max_seq_length`/`max_length` compatibility fallbacks for different
  transformers/trl versions).
- After training (`finetune/eval_split.py`): per-category held-out loss
  masked to the assistant answer tokens, plus deterministic, judge-free
  accuracy — sentiment label accuracy, NER span F1 + exact-match rate,
  summarization constraint-compliance + number retention, factual token F1,
  math value accuracy (+ executable rate, unit accuracy). Written to
  `<out>/val_metrics.json`; a provenance manifest (`<out>/train_manifest.json`)
  records model, data sha256, row counts, args and library versions.
- `--skip-eval` skips the post-training pass; `--no-thinking` renders
  generation prompts with `enable_thinking=False` for Qwen3-family models.

This replaces model selection by single judge runs with reproducible
numbers (the LLM judge remains available via `eval/judge.py` for
end-to-end scoring).

## Reproducible dataset generation

```bash
set -a; source .env; set +a
python -m finetune.gen_data --per-category 600 --seed 42
```

- Everything local is deterministic: per-call RNGs are seeded
  (`f"{seed}:{category}:{call_index}"`) for topic/constraint choices and
  row ordering, an API `seed` field derived from `f"api:{seed}:…"` is sent
  with each request, and the combined-file shuffle uses `--seed`.
- `--model` pins the generator model (default: the `kimi` entry in
  `ALLOWED_MODELS`; e.g. `accounts/fireworks/models/kimi-k2p7-code`).
- `data/gen_manifest.json` records: the generator file's own sha256 and
  python/platform, model id + endpoint host (never the API key), all
  parameters, per-category kept counts, generator tokens spent, and the
  sha256 + size of every output JSONL. Regenerate and diff the manifest to
  detect drift; then run the validator.
- Known limit: the provider may not be bit-deterministic at temperature
  0.9 even with a seed; the manifest and validator make any drift visible
  rather than silent.

## Base-model candidates (verified 2026-11-03)

| candidate | verdict |
| --- | --- |
| `unsloth/Qwen2.5-1.5B-Instruct`, `unsloth/Qwen2.5-3B-Instruct` | in use (existing A/B winners per `SCORES.md` lineage) |
| `unsloth/Qwen3.5-2B-Instruct` — optional cell in `colab_unsloth.ipynb` | **does not exist**: `https://huggingface.co/api/models/unsloth/Qwen3.5-2B-Instruct` returns Repository-not-found (HTTP 401/404), and the id is absent from unsloth's own Qwen3.5 catalog (only `-GGUF`, `-MTP-GGUF`, `-NVFP4` mirrors exist there). Do not rely on it. |
| `Qwen/Qwen3.5-2B` / `unsloth/Qwen3.5-2B` | verified on the Hub (canonical: ~3.4M downloads, apache-2.0, sha `15852e8c16360a2fea060d615a32b45270f8a8fc`, created 2026-02-28; unsloth mirror sha `fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`). **Caveat:** the Qwen3.5 family is multimodal (`image-text-to-text`, `Qwen3_5ForConditionalGeneration`, `AutoProcessor`) — text-only SFT data is fine, but verify the Unsloth/transformers loader accepts it on the Colab image before spending session time; it defaults to an empty think block (`enable_thinking` supported). `colab_unsloth.ipynb` now uses `unsloth/Qwen3.5-2B` behind an executable Hub-existence guard. |

Zero-shot probes historically lost Qwen3-family 2B models to
Qwen2.5-1.5B in the grading sandbox (6.9 vs 8.2 tok/s); fine-tuning may
flip that, which is what the optional A/B cell is for.

## Colab notebooks

- `colab_unsloth.ipynb` — preferred (T4): Unsloth LoRA + built-in GGUF
  export; optional Qwen3.5 cell verifies the model id against the Hub
  first.
- `colab_finetune.ipynb` — fallback vanilla transformers/peft/TRL flow.

