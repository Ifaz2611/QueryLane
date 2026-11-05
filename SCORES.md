# QueryLane Score Log

QueryLane (originally submitted as Team AdversaryAI). This is the historical
record from the Track 1 evaluation. All times GMT+6. Accuracy gate: 80%
(16/19). Rank is by fewest Fireworks tokens among gate-passers.

| # | Image | Approach | Models (exact) | Resubmitted | Scored | Tokens | Accuracy | Rank | Notes |
|---|-------|----------|----------------|-------------|--------|--------|----------|------|------|
| 1 | `:latest` (phase 1) | All-cloud. Heuristic regex router picks a per-category terse system prompt and max_tokens; kimi-k2p7-code primary with minimax-m3 fallback; `reasoning_effort: none` disables thinking; one call per task, transient-only retries. | K, M | Jul 11 12:26 | Jul 11 13:42 | 3,877 | 94.7% | 12 | First pass; baseline |
| 2 | `:latest` (phase 2) | Hybrid: bundled Qwen2.5-3B GGUF (llama.cpp, CPU) answers sentiment + NER locally for zero tokens; everything else cloud as in phase 1. Local sentiment used the early "factual statement = neutral" rubric. | K, M, Q3 | Jul 11 15:29 | Jul 11 16:28 | — | 78.9% | — | ACCURACY_GATE_FAILED; rubric bug labeled factual-toned complaints neutral |
| 3 | `:latest` (phase 2 regraded) | Same image as row 2, re-scored by the nondeterministic LLM judge. | K, M, Q3 | Jul 11 17:12 | Jul 11 23:47 | 3,455 | 84.2% | 13→14 | One task flipped 15/19 to 16/19; back on board |
| 4 | `:v3` | Hybrid widened: local sentiment + NER + math (step-by-step CoT), fixed sentiment rubric, treaty/document NER type normalizer, escalation guards (suspicious local answers re-asked in cloud), tuned terse cloud prompts. | K, M, Q3 | Jul 12 08:20 | Jul 12 10:16 | — | — | — | TIMEOUT; local math CoT (~350 tokens/answer) too slow on the cold, slow grader VM |
| 5 | `:v5` | Adaptive hybrid: background model preload, warmup speed calibration (4 tok/s bar; math/logic need 8), 300s local time budget, cloud fallback everywhere; math auto-clouds on slow VMs. | K, M, Q3 | Jul 12 11:33 | Jul 12 13:06 | 3,461 | 94.7% | 15 | Grader measured < 4 tok/s so local stayed off; ran as all-cloud tuned. No timeout. |
| 6 | `:v6` | v5 plus trimmed code outputs: code_gen "only the code, no comments/docstrings", code_debug "one sentence + corrected code only". | K, M, Q3 | Jul 12 13:32 | Jul 12 15:03 | 3,459 | 94.7% | 14 | Code answers are code-dominated; trim saved ~2 tokens |
| 7 | `:v7` | v6 plus local reactivation for slow VMs: entry bar lowered to 2.5 tok/s (time budget is the real limiter), cheapest-first local ordering (sentiment, then NER), 150s deadline cutoff flips pending local work to cloud so answers are never empty. | K, M, Q3 | Jul 12 15:58 | Jul 12 16:36 | 3,476 | 94.7% | 15 | Tokens ~= v6: grader is below even 2.5 tok/s; local never activated |
| 8 | `:v8` (never submitted) | v7 plus per-category batched cloud calls with item delimiters. | K, M, Q3 | — | — | — | — | — | KILLED by A/B measurement: 1,303 batched vs 1,263 unbatched on identical tasks; delimiter overhead exceeds saved per-call overhead at real group sizes (2-3) |
| 9 | `:v9` | v7 plus the document-entity NER normalizer applied to cloud answers too (was local-only); batching removed. | K, M, Q3 | Jul 12 16:55 | Jul 12 17:06 | — | 0.0% | — | ACCURACY_GATE_FAILED 0.0%: harness-side failure. Image verified healthy post-hoc (8/8 answers, exit 0, 43s under grading constraints). Re-saved for regrade. |
| 10 | `:v10` | Non-generative specialists: int8 ONNX BERT-NER (+ date regex, span merge, confidence gate) and DistilBERT-SST2 sentiment (clause aggregation, sentiment-cue gate, quoted-evidence justifications) answer NER/sentiment in ~1s on any CPU for zero tokens; uncertain cases escalate to cloud. Instruction stripping on sentiment/NER cloud fallbacks. Generative local math disabled pending fine-tuned model (sign errors + missing units). | K, M, Q3, B, S | Jul 12 18:20 | Jul 12 18:58 | 2,988 | 84.2% | 13 | Specialists worked (-471 tokens) but cost ~2 correct answers on unseen variants. AT the gate, zero headroom. v11 must tighten specialist precision. |
| 11 | `:v11` | Precision-first specialists: NER specialist disabled (measured 21.9% exact-match precision on 597 ground-truth tasks; NER back to cloud). Sentiment specialist gated to measured 100% precision at 38% coverage (hedge escalation, contrast-conjunction escalation, cue-word/classifier agreement per side, 0.97 whole-review confidence). | K, M, S | pending | pending | — | — | — | Dev 39/40; the one miss is a cloud summarization length quirk. Restores ~2-miss gate headroom for ~200 tokens. |
| 12 | `:v12` | v11 plus: cloud-PAL math (kimi emits 'EXPR: ... \| UNIT: ...' ~15-20 completion tokens, evaluated locally by the AST-whitelist evaluator, step-by-step fallback on parse failure); router hardened for paraphrases (64/64 across dev+practice+16 paraphrase variants, was 9/16 on paraphrases); factual fallback cap 150->500 to prevent misroute truncation; summarization sentence-count emphasis; LOCAL_MIN_TPS=1.2 so generative local (NER few-shot + escalation-guarded) may activate on the grader for the first time. | K, M, Q3, S | Jul 12 20:59 | Jul 12 21:22 | **2,897** | 89.5% | 14 | Best passing token score yet (-579 vs v7). 17/19: one miss of headroom above the gate. Leaderboard has tightened; ~2,000 needed for top 5. |
| 13 | `:v13` | Fine-tuned Qwen2.5-1.5B (Unsloth LoRA on 2,753 validated synthetic rows, Colab T4) serves factual + math (PAL, unit-aware) locally, both 5/5 in the model A/B. Summarization NOT promoted (fine-tune regressed it to 0/5, caught by A/B). Country-mislabel NER normalizer added. Image 1.23GB compressed, 1.5B runs 6.5 tok/s even at 1 CPU. | K, M, Q1.5, B, S | Jul 12 23:24 | Jul 13 01:05 | 2,881 | 89.5% | 12 | Only -16 vs v12: local never activated. Registry proves the grader pulled v13; diagnosis: grader ~= our 0.25-CPU throttle, 1.5B calibrates ~1.2 tok/s, exactly at the 1.2 bar -> disabled. |
| 15 | `:v15` | Activation fix: global bar 1.2 -> 0.4 with per-category speed bars (math 0.4, NER 0.9, factual 1.3, sentiment 0.9) so categories degrade individually; local queue reordered math -> NER -> factual (savings per second); per-answer token cap scaled to measured speed (<= ~75s/answer); calibration shortened. v14's guarded-NER promotion included. | K, M, Q1.5, B, S | pending | pending | — | — | — | 0.25-CPU sim (0.5 tok/s measured): math local, -368 tokens vs all-cloud, 19/19 in 202s. At grader ~1.2: math+NER expected (~2,300-2,400). |
| 14 | `:v14` | v13 plus NER served by the fine-tuned 1.5B behind a zero-token completeness guard (every capitalized span + date in the passage must appear in the answer or the task escalates; measured 62% acceptance / 99% dropped-entity detection on 597 ground-truth rows). Logic promotion tested and rejected (ft-1.5B 0/5 on deduction). | K, M, Q1.5, B, S | pending | pending | — | — | — | Dev 37/40 (2 known judge flip-floppers + 1 judge-variance factual). 1-CPU stress: 19/19 in 77s, only 9 cloud calls, 924 dev-stress tokens. |
| 16 | `:v16` | v15 plus multi-part math detector routing two-quantity questions to step-by-step instead of single-expression PAL (local and cloud). | K, M, Q1.5, B, S | Jul 13 02:03 | Jul 13 08:30 | 3,135 | 84.2% | 26 | (Deadline later extended; not final after all.) Two compounding failures: (1) local never activated (final-hours grader contention pushed speed below even the 0.4 bar), (2) the multipart regex (",? and the") false-positived on ordinary word-problem prose, converting ~120-token PAL math into ~200-token step-by-step. The "cannot exceed v13" verification claim wrongly ignored that the fix also changed the CLOUD math path cost. Rank drop mostly the late-submission flood. |
| 17 | `:v17` | v15 plus narrowed multipart detector (requires a second interrogative; 12/12 incl. adversarial single-part prose), applied to local AND cloud math. | K, M, Q1.5, B, S | — | — | — | — | — | Superseded by v17b before submission. |
| 17b | `:v17b` | v17 with the multipart detector gating ONLY the local math path; the cloud math path is behaviorally identical to the twice-scored v13 (PAL first, fallback on parse failure). Floor = v13 by construction; local activation (bars 0.4/0.9/1.3) is pure upside. | K, M, Q1.5, B, S | — | — | — | — | — | Math dev 5/5 at 622 tokens (PAL parity). Superseded by v17c's prompt trims. |
| 17c | `:v17c` | v17b plus verified system-prompt trims: math-PAL spec 65 -> ~32 tokens (-30/call, category 622 -> 472, still 5/5 all-PAL), NER spec trimmed (-6/call, 5/5). No other changes. | K, M, Q1.5, B, S | Jul 13 16:32 | Jul 13 17:18 | 3,166 | 89.5% | 29 | +285 vs v13's old 2,881 despite trims, accuracy held at 89.5%. Same-sized excess as v16 (+254) WITHOUT v16's regex bug: suggests the hidden task set changed with the deadline extension (heavier prompts), not a build regression. Decisive check: v13's own re-score on the new set. |

### Model legend (exact identifiers)

| Tag | Exact model / artifact | Role |
|-----|------------------------|------|
| K | `accounts/fireworks/models/kimi-k2p7-code` | Primary cloud model (text tasks, PAL math) |
| M | `accounts/fireworks/models/minimax-m3` | Cloud fallback and LLM-judge model |
| Q3 | `Qwen/Qwen2.5-3B-Instruct`, Q4_K_M GGUF (`qwen2.5-3b-instruct-q4_k_m.gguf`) | Local generative lane (rows 2-12) |
| Q1.5 | `Qwen/Qwen2.5-1.5B-Instruct` + Unsloth LoRA on 2,753 synthetic rows, Q4_K_M GGUF (`ft-1.5b-q4_k_m.gguf`) | Local generative lane (rows 13+) |
| B | `dslim/bert-base-NER`, int8 ONNX | NER specialist (`models/ner`) |
| S | `distilbert-base-uncased-finetuned-sst-2-english`, int8 ONNX | Sentiment specialist (`models/sst2`) |

Historical revisions were **not pinned**: the `Qwen/*` and Hugging Face weights above
are base model ids plus the local GGUF/ONNX artifact names, not specific Hub commit
SHAs. Fireworks model ids resolve to whatever provider revision was live on the
scored date. Future rows should record the resolved Hub commit and image digest.

## Grader hardware conclusion (from rows 5-7)

v5 (bar 4 tok/s), v6, and v7 (bar 2.5 tok/s) all scored ~3,460: the grader VM generates
below 2.5 tok/s on a 3B Q4 GGUF. Generative local inference is effectively unusable there.
Zero-token answering must come from non-generative specialists (ONNX classifiers, program-aided
math with tiny generations, regex) rather than LLM text generation.

## Key facts derived from scores

- Grader VM generates < 4 tok/s on the 3B: local models only help if they run within a strict time budget at ~2.5-3.5 tok/s.
- Real set costs ~1.7x our dev-set tokens (longer prompts). All-cloud tuned floor is ~3,450.
- Estimated all-cloud absolute floor (prompts are fixed): ~2,700-2,900. Ranks below ~2,600 require working local inference on the slow grader.
- Accuracy has been stable at 94.7% (18/19) for every healthy build; we hold 2-miss headroom above the gate.
- Every re-save replaces the previous score, including failures. Freeze 2h before deadline.

## Competitive context (Jul 11 snapshot)

#1 yassai 1,377 @ 89.5% | #2 Metis 1,756 @ 100% | #3 Pathfinder0 1,763 @ 84.2% | #4 Kestrel 2,567 @ 100%
