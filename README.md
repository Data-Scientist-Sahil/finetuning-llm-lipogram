# no-e-eval — a requirement-level benchmark for lipogram ("no letter e") generation

**Part 1 — evaluation harness:** the committed results analyse Kevin Wu's released
`k33wu/qwen3-1.7b-no-e` SFT+RL adapter on `Qwen/Qwen3-1.7B`, against base-model
baselines. These are not results from a model trained by Sahil.

**Part 2 — verifiable-reward training (Run B):** `train/` and the Kaggle notebook
continue training from Kevin Wu's SFT-only adapter with a judge-free reward. One
100-step run (Run B) has now been run on a Kaggle T4 and evaluated; see
[Run B results](#run-b--100-step-verifiable-reward-run-kaggle-t4). The trained
adapter weights are not committed (size); its config and training log are in
`results/idea2_runB/`.

**Review provenance:** Claude assigned the 18 qualitative sample scores as a proxy.
They are AI-generated review, not independent human review or Sahil's manual
ratings, and have not been independently judge-validated. No judge API cross-check
was run. Legacy `manual_*` filenames are retained for compatibility.

A small, reproducible evaluation harness that stress-tests the claim behind
[Kevin Wu's "Teaching Qwen to write without the letter e"](https://kevin-wu.ca/blog/qwn/):
*does the trained model actually answer the question without an 'e', or does the
apparent improvement come from quietly saying less (or switching scripts)?*

Kevin scores answers with a single **usable rate** (no-'e' AND coherent AND
responsive, judged 1–5 by an LLM). That one number can be satisfied by
**shrinking** the answer — his median answer length fell 52 → 18 tokens — because
a shorter answer is easier to keep 'e'-free and locally coherent. This harness
splits "usable" into separately-measured axes so that failure mode is visible.

## What it measures (independently)

| Axis | How | Model needed? |
|---|---|---|
| **letter_compliance** | answer contains no `e`/`E` | no (regex) |
| **pure-English / script-escape** | fraction of letters that are Latin; catches CJK/Cyrillic escapes | no |
| **requirement_completion** | did it actually produce the N items (each with a reason) / N sentences / ≤ k words that were asked for | no (structural parse) |
| **length** | words, chars, sentences | no |
| **coherence & responsiveness** | 1–5, by **AI-generated proxy review (Claude)**; optional pluggable LLM judge | judge optional |

The deterministic axes need no judge, so the core result is cheap and fully
reproducible. The qualitative sample scores in `results/manual_review_scored.csv` were assigned
by Claude, not a human. They are provisional; independent human or judge review
is still needed. `manual_review.csv` is an unscored review worksheet.

## Headline result (Qwen3-1.7B, 41 prompts, greedy, CPU)

| Config | Letter % | Pure-English % | Requirement-met % | Usable (strict) % | Median words | Claude proxy coherence |
|---|---|---|---|---|---|---|
| Base + prompt only | 0.0 | 100 | 95.1 | 0.0 | 41 | — |
| Base + token mask | 100 | **61.0** | 85.4 | 82.9 | 33 | 3.33 |
| Adapter (SFT+RL) + mask | 100 | 100 | 87.8 | 87.8 | 27 | 4.89 |

Three findings the single usable-rate metric hides:

1. **Script-escape.** The token mask bans 'e'-tokens but not other scripts, so
   the mask-only model escapes into CJK/Cyrillic (`running鞋`, `帮助`, `вопросы`)
   in **~39% of answers**. A naive no-'e' check scores these as wins.
2. **Script compliance improves; qualitative fluency is provisional.** The
   measured pure-English rate rises 61% → 100%. Claude's unvalidated proxy
   coherence scores rise 3.33 → 4.89; they are not independent human evidence.
   Requirement-completion gains are small and their bootstrap CI includes zero.
3. **The adapter regresses on some requirements.** By requirement type, it
   improves `items_with_reason` (70.6% → 94.1%) but **drops on exact sentence
   counts** (90.9% → 63.6%): it compresses answers to a single fluent sentence
   when asked for exactly two. The shortest config (median 27 words) is also the
   one that most often under-delivers the requested structure.

Full table, per-type breakdown and failure examples: [`results/REPORT.md`](results/REPORT.md).

## Repo layout

```
prompts/build_prompts.py     authoring script for the prompt set
prompts/eval_prompts.jsonl   41 prompts, each with a machine-checkable requirement spec
src/masking.py               no-'e' LogitsProcessor (whitelists special tokens; decodes to catch 'e')
src/generate.py              resumable greedy generation for each config
src/metrics.py               deterministic axes (letter, script, requirement, length)
src/judge.py                 pluggable LLM judge (Anthropic/OpenAI) + manual-review CSV export
src/run_eval.py              score raw generations -> scored/ + summary.json
src/report.py                render results/REPORT.md
src/manual_scores.py         recorded Claude proxy scores for the fixed sample
demo.py                      one prompt, three configs, side by side
results/                     generated: raw/, scored/, summary.json, bootstrap.json, REPORT.md, manual_review*.csv
results/idea2_runB/          Run B training log + adapter config/card (weights not committed)
train/                       verifiable reward, train-prompt builder, GRPO-lite trainer
notebooks/                   Kaggle T4 notebook + the script that builds it
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. (re)build the prompt set
python prompts/build_prompts.py

# 2. generate answers for each config (resumable; --max-seconds to chunk on CPU)
python src/generate.py --config base_prompt  --max-new-tokens 160
python src/generate.py --config base_mask    --max-new-tokens 160
python src/generate.py --config adapter_mask --max-new-tokens 160   # pulls k33wu/qwen3-1.7b-no-e
# optional 4th config: --config sft_mask (k33wu/qwen3-1.7b-no-e-sft180)

# 3. score every deterministic axis
python src/run_eval.py --configs base_prompt base_mask adapter_mask
#   add --judge (needs ANTHROPIC_API_KEY or OPENAI_API_KEY) for the automated judge

# 4. re-export recorded Claude proxy scores (does NOT perform a new review) + report
python src/manual_scores.py
python src/report.py            # -> results/REPORT.md
```

On a GPU, drop `--max-seconds`, raise `--max-new-tokens`, and optionally sample
multiple answers per prompt. The existing bootstrap resamples the 41 prompts,
not repeated model generations; it does not estimate decoding variance.

## Configs

- `base_prompt` — Qwen3-1.7B, no mask. The "prompting alone" baseline (0% 'e'-free, as Kevin reports).
- `base_mask` — Qwen3-1.7B + no-'e' token mask.
- `adapter_mask` — base + Kevin's released SFT+RL adapter `k33wu/qwen3-1.7b-no-e` + mask.
- `sft_mask` — base + his SFT-only adapter `k33wu/qwen3-1.7b-no-e-sft180` + mask.

## Limitations / honesty notes

- CPU run: greedy, 1 sample/prompt, `max_new_tokens=160`. Deterministic, but no
  sampling variance; for CIs, sample 4–8/prompt and bootstrap over prompts (as Kevin does).
- The requirement checkers are structural, not semantic — they verify the shape
  of the answer (counts, substance length), not factual correctness. That is
  deliberate: structure is where the cheating shows up.
- `looks_english` uses a 0.9 Latin-letter threshold; `pure_latin` (≥0.999) is the
  strict escape count. Both are reported.
- 41 prompts is a screening-size set. The authoring script makes it trivial to extend.

## Part 2 — experimental verifiable-reward training kit

The `train/` workflow is designed to continue training Qwen3-1.7B from Kevin's SFT-only adapter with a
**fully verifiable reward (no LLM judge in the loop)** that targets requirement-completion,
then scores the result on this same benchmark.

```bash
python train/make_train_prompts.py                 # 72 train prompts, disjoint from eval
python train/reward.py                              # reward unit tests (8/8)
python train/grpo_train.py --device cuda --dtype float16 --steps 100 --micro-batch 2 --out results/idea2_runB
python src/generate.py --config trained_mask --adapter-path results/idea2_runB/adapter --device cuda
python src/generate.py --config sft_mask --device cuda      # SFT-180 start = controlled baseline
python src/run_eval.py  --configs base_mask sft_mask adapter_mask trained_mask
python src/bootstrap.py --configs base_mask sft_mask adapter_mask trained_mask
python src/report.py
```

- `train/reward.py` — verifiable reward (hard gates: no-'e' / English / not-degenerate;
  dense `0.15 + 0.70*completeness - repetition_penalty`), reusing `src/metrics.py` so reward
  and eval share one parser.
- `train/grpo_train.py` — a compact, self-contained GRPO loop (no `trl`): masked rollouts →
  group-normalised advantage → policy-gradient + KL-to-base → LoRA update → checkpoints.
- `notebooks/idea2_kaggle.ipynb` — Kaggle T4 workflow (smoke test → 100-step train → eval → zip); this is the notebook Run B came from.

### Execution caveats

- Run B ran on one Kaggle T4 (16 GB) in fp16 (`DTYPE="float16"`; the T4 has no
  bf16 hardware) with `--micro-batch 2`: ~100 steps in ~14 min, ~25 min end to end
  with evaluation. Other hardware is untested.
- Training saves adapter weights every 20 steps by default. It does **not**
  automatically resume training: each invocation starts a fresh loop from
  `--start-adapter` (Kevin's SFT adapter by default). Optimizer state and global
  step are not restored. Generation, separately, skips already-written prompt IDs.
- The Kaggle notebook pins `transformers==4.51.3`, `peft==0.15.2`,
  `accelerate==1.6.0`; `requirements.txt` still uses minimum versions, and model
  revisions are not pinned. Exact cross-version reproducibility is not guaranteed. The notebook
  also uses `matplotlib`, normally available on Kaggle.
- Optional judge API calls require separate SDK installation and credentials;
  no API cross-check is included in the committed results.

## Run B — 100-step verifiable-reward run (Kaggle T4)

**What changed vs. the first (40-step) attempt**

- **Micro-batching for T4 memory.** `train/grpo_train.py` now forwards/backwards the
  policy and reference passes in micro-batches (`--micro-batch 2`) and computes
  log-probs as `gathered logit − logsumexp` instead of materialising a full-vocab
  fp32 log-softmax, which was the T4 OOM.
- **fp16 on T4** and **pinned deps** in the notebook: `transformers==4.51.3`,
  `peft==0.15.2`, `accelerate==1.6.0`.
- **100 steps** (4 prompts × group 8, lr 1e-5, KL β 0.02, temp 0.7, 128 new tokens);
  `max_words` prompts excluded from training (72 → 66 train prompts).
- Reward: completeness is scaled by a length-adequacy floor so a too-short answer
  can't claim full credit (`train/reward.py`).
- Eval now includes the **SFT-180 start** (`sft_mask`) so `trained − SFT-180` is a
  controlled test of our reward.

**Results** (41 eval prompts, greedy, token mask on; full report: [`results/REPORT.md`](results/REPORT.md))

| Config | Requirement-met % | Pure-English % | Letter % | Median words |
|---|---|---|---|---|
| Base + token mask | 85.4 | 61.0 | 100 | 33 |
| SFT-180 start + mask | 97.6 | 100 | 100 | 45 |
| Kevin's adapter (SFT+RL) + mask | 87.8 | 100 | 100 | 27 |
| **Trained (Run B) + mask** | **95.1** | 100 | 100 | 31 |

| Paired difference (bootstrap over prompts, 2000 resamples) | Δ requirement (pts) | 95% CI |
|---|---|---|
| Trained − SFT-180 (controlled test) | −2.4 | [−7.3, 0.0] |
| Trained − Kevin's adapter (unmatched compute) | +7.3 | [−2.4, 19.5] |

**Honest takeaway**

- Run B did **not** beat its SFT-180 starting point: −2.4 pts (one prompt), CI
  [−7.3, 0], not significant. The +7.3 over Kevin's adapter is also not
  significant, and his adapter had far more RL, so it is not a fair comparison.
- Answers got **~30% shorter** (median 45 → 31 words) while staying 100% 'e'-free
  and 100% English. The one new miss (p26) is a single sentence where exactly two
  were asked for: the same failure Kevin's adapter shows (63.6% on sentence-count
  prompts vs. 90.9% for Run B).
- **KL rose from ~3 to ~8–12 while reward stayed roughly flat** (`results/idea2_runB/train_log.jsonl`),
  so the policy drifted from the start without being rewarded for it.
- **Next step:** a stronger KL penalty and an exact sentence-count reward (using
  the same checker as the eval), rather than more steps.

Qualitative coherence scores (Claude proxy review) cover base and Kevin's adapter
only; SFT-180 and Run B outputs have not been review-scored.

## Credit

Builds directly on Kevin Wu's experiment and released artifacts
(`k33wu/qwen3-1.7b-no-e`, `k33wu/qwen3-1.7b-no-e-sft180`, and his training/eval code).
This repo adds a separate requirement-level evaluation harness. Claude assisted
with development and assigned the provisional qualitative review scores.
Third-party model/adapter licenses remain separate from this repository's MIT
license; model weights are not bundled.
