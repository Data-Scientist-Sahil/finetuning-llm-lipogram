# Findings — a requirement-level benchmark for lipogram generation, and what RL does to it

This project builds on Kevin Wu's experiment ["Teaching Qwen to write without the letter e"](https://kevin-wu.ca/blog/qwn/).
Kevin trains Qwen3-1.7B to answer everyday prompts as lipograms (no letter 'e') and scores
them with a single **usable rate** — no-'e' AND coherent AND responsive, judged 1–5 by an LLM.
His own writeup flags that gains under the token mask came partly from *shorter* answers
(median length fell 52 → 18 tokens), because a summed judge reward lets a (coherence=5,
responsiveness=3) answer tie a (4,4) even though only the latter is usable.

The work here is in two parts: **(1)** a benchmark that decomposes that single usable rate into
separately-measured axes, and **(2)** a reinforcement-learning experiment that asks whether a
*fully verifiable* reward (no judge in the loop) can target completeness better than the
judge-based reward. Everything is measured with bootstrap confidence intervals, and the
negative/null results are reported as such.

## Part 1 — the benchmark

41 held-out prompts, each with an explicit, machine-checkable requirement ("give 3 suggestions,
each with a reason"; "exactly 2 sentences"; "≤ 8 words"). Five axes, scored **independently**:

- **letter compliance** — no 'e'/'E' (regex).
- **script purity** — fraction of letters that are Latin; catches a cheat the token mask allows.
- **requirement completion** — did the structural ask actually happen (parse the items/sentences).
- **length** — words / sentences.
- **coherence & responsiveness** — 1–5, by **AI-generated proxy review (Claude)** on a fixed
  9-prompt sample for base and Kevin's adapter only. These scores are not independent human
  review and have not been independently judge-validated. An LLM-judge backend is included but
  was not run for the shipped results.

### What the benchmark revealed (that a single usable rate hides)

**Script escape.** The token mask forbids 'e'-tokens but not other scripts, so the mask-only
base model escapes into CJK/Cyrillic (`running鞋`, `帮助`, `вопросы`) in **~39% of answers**
(pure-English 61.0%, CI [46.3, 75.6]). Each passes a naive no-'e' check but is not English.
Kevin's trained adapter fixes this completely (100% pure English), and the bootstrap shows this
is the adapter's **only statistically-robust edge** over a bare mask: +39 points, CI [24.4,
53.7], the one paired difference whose interval excludes zero. Its apparent completeness and
"usability" gains are within noise at n=41.

**Completeness is an SFT property, not an RL one.** Measured on the 41 prompts:

| Configuration | requirement-met % | pure-English % | median words |
|---|---|---|---|
| base + token mask | 85.4 | 61.0 | 33 |
| **SFT-180 + mask** (Kevin's SFT-only adapter) | **97.6** | 100 | 45 |
| Kevin's released adapter (SFT+RL) + mask | 87.8 | 100 | 27 |
| our RLVR Run B (SFT + 100-step verifiable RL) + mask | 95.1 | 100 | 31 |
| our RLVR Run C (KL to frozen SFT, beta 0.03) + mask | 92.7 | 100 | 20 |

The SFT-only checkpoint is the completeness champion. **Both** of Kevin's and our RL adapters
sit below it — Kevin's judge-reward adapter at −9.8 points, our Run B at −2.4, our Run C at
−4.9 — and all three are shorter. In other words, the RL stage trades completeness for
brevity; the SFT stage is where following the instruction actually lives. By requirement type,
SFT scores 100% on items, items-with-reason and sentences, and its only weak spot is
`max_words` (66.7% — it is too verbose when asked to be brief).

## Part 2 — RL with a verifiable reward (RLVR)

**Method (labelled honestly).** `train/grpo_train.py` is a compact, self-contained **GRPO-lite**
loop — REINFORCE with GRPO's group-normalised baseline plus a KL term (no PPO clipping). It is
legible and enough to test the reward; it is *not* a faithful GRPO reimplementation. Rollouts
are generated under the no-'e' mask (otherwise almost every sample fails the letter gate and
there is no learning signal), and the forward/backward is micro-batched with logsumexp
log-probs so it fits a 16 GB T4.

The KL reference changed between runs. In v1–v2 / Run B it was the base model with the adapter
disabled (`--kl-ref base`), so a KL of 8–12 mostly reflected the gap from base, not drift from
SFT. In Run C the trainer loads a **frozen copy of the SFT-init adapter** (`--kl-ref sft`, now
the default) and measures KL against that.

**Reward (`train/reward.py`, no judge).** Hard gates (letter / English / not-degenerate) plus a
dense term `0.15 + 0.70·completeness − repetition_penalty`, reusing the benchmark's own parser
so the reward and the evaluation share one definition of "requirement met".

**The iteration — and why it's the interesting part.** The reward was debugged *with the
benchmark*, which is the whole point of building Part 1 first:

- **v1** (40 steps, trained on all types, no length floor) *regressed*: it got better at
  `max_words` and worse at the completeness prompts — the reward had learned global brevity.
  Requirement-completion 92.7%, median 27 words.
- **v2 / Run B** (100 steps, `max_words` dropped from training, length-adequacy floor added so a
  too-short answer can't claim completeness credit) recovered: **95.1%**, the items-with-reason
  regression eliminated (100%), shrinkage eased (median 31). Paired vs SFT: −2.4 points, CI
  [−7.3, 0.0] — statistically indistinguishable from SFT (a one-prompt difference, 39 vs 40/41).
  Paired vs Kevin's released adapter: **+7.3 points** [−2.4, 19.5] — the point estimate favours
  the verifiable reward, but it is not significant. KL (to base) rose from ~3 to ~8–12 while reward
  stayed roughly flat.
- **Run C (final)** (100 steps, `max_words` re-included, **KL reference switched from base to a
  frozen SFT-init adapter**, beta 0.03): intended to target SFT's single weakness (brevity on
  `max_words`) while the SFT-anchored KL held completeness near the good start.

### Run C result

Run C did **not** improve on SFT, and it came out slightly worse than Run B.

- Requirement-met: base 85.4, SFT-180 97.6, Kevin's adapter 87.8, Run B 95.1, **Run C 92.7**.
- Paired bootstrap (n=41, 2000 resamples): Run C − SFT = **−4.9 [−12.2, 0]**, not significant;
  Run C − Kevin = **+4.9 [−4.9, 17.1]**, not significant. Kevin − base on `pure_english` remains
  the only significant paired gap: **+39 [24.4, 53.7]**.
- Median words: SFT 45, Run B 31, **Run C 20**, Kevin 27. Answers kept getting shorter.
- By type, Run C: items 100%, `items_with_reason` 88.2% (SFT 100%), sentences 100% (Kevin 63.6%),
  `max_words` 66.7% (same as SFT — the weak spot this run was meant to fix did not move).
- Failures: **p37** ("8 words or fewer") degenerated into "Kids grow with shut shut shut…" for
  160 tokens; **p09** and **p39** listed items without the required four-word reasons.

**KL and reward, from `results/idea2_runC/train_log.jsonl` (100 steps, 2893 s).** KL started at
0.0007 on step 1 — the SFT reference is working, and KL begins near 0 as designed. It did not
stay small: mean 5.04, median 5.32, first-10 mean 0.91, last-10 mean 6.76, max **8.6984** at
step 73, last step 8.0683. Mean reward started at 0.759, ended at 0.700 (a last-step dip),
first-10 mean 0.670, last-10 mean 0.775, overall mean 0.733, min 0.502 (step 9), max 0.824
(step 77). Reward drifted up slightly on average; it did not collapse. beta 0.03 was not strong
enough to hold the policy near SFT.

### Honest bottom line (as of Run C)

1. A verifiable, judge-free reward **did not beat** a strong SFT checkpoint on completeness
   (Run B −2.4, Run C −4.9; both CIs include 0) and is statistically indistinguishable from it
   at n=41. There is almost no completeness headroom above 97.6%. Versus Kevin's judge-reward
   adapter the point estimates favour the verifiable reward (Run B +7.3, Run C +4.9) but those
   gaps are also not significant, and the compute is unmatched.
2. The RL stage (either reward, either KL reference) trades completeness for brevity. Fixing
   the KL reference so it measures drift from SFT, not the gap to base, did not stop that
   length drift at beta 0.03. Putting `max_words` back in training did not raise `max_words`
   completion and produced a repetition collapse on one prompt.
3. The honest result is **no gain over SFT**. The benchmark is what makes that visible, and
   what let the reward be debugged (v1 → v2 → Run C) rather than guessed.
4. **Next ideas (not done):** a length/completeness floor for `items_with_reason`, a stronger
   repetition penalty or guard, and a stronger beta or fewer steps.

## Limitations (stated plainly)

- n = 41; most paired differences do **not** clear significance. Bootstrap CIs are over prompts,
  not over training seeds or repeated generations (single seed, greedy eval).
- GRPO-lite ≠ standard PPO-clipped GRPO. The KL-to-base weakness of runs v1–v2 was addressed in
  Run C (`--kl-ref sft`), but beta 0.03 still allowed KL to climb to ~8.
- "Pure Latin" is a character-set check, not proof of fluent English; "usable strict" is not an
  independently validated usability metric. The trained adapters have not had a human-coherence
  pass (the Claude proxy table currently covers only base and Kevin's adapter).
- Comparisons to Kevin's released adapter are unmatched in training budget — reference only.
- Run artifacts must be persisted via Kaggle **Save & Run All**; an interactive session wiped the
  first run's weights (numbers survived in logs). Adapter weights are not committed here.
