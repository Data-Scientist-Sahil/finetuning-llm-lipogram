# No-'e' lipogram benchmark — results

These results evaluate Kevin Wu's released `k33wu/qwen3-1.7b-no-e` adapter on `Qwen/Qwen3-1.7B`, not a model trained by Sahil. The experimental GRPO training kit has NOT RUN; no new trained-model results are included.

Each axis is measured **independently**. `letter_compliance` = answer has no 'e'. `requirement_completion` = the explicit structural ask (N items each with a reason, a sentence count, a word limit) was actually met. `usable` = both at once (this is the closest analogue to Kevin's single usable-rate gate, before coherence). Coherence/responsiveness sample scores were assigned by Claude as AI-generated proxy review, not independent human review, and are not independently judge-validated. No judge API cross-check ran. Legacy manual_* filenames are retained; manual_review.csv is an unscored worksheet.

## Headline: the axes come apart

| Config | n | Letter-compliant % | Pure-English % | Requirement-met % | Usable lenient % | Usable strict % | Mean words | Median words |
|---|---|---|---|---|---|---|---|---|
| Base + prompt only | 41 | 0.0 | 100.0 | 95.1 | 0.0 | 0.0 | 40.195 | 41 |
| Base + token mask | 41 | 100.0 | 61.0 | 85.4 | 85.4 | 82.9 | 39.537 | 33 |
| Adapter (SFT+RL) + mask | 41 | 100.0 | 100.0 | 87.8 | 87.8 | 87.8 | 34.707 | 27 |

Two gaps tell the story. **Letter-compliant % vs Requirement-met %**: the cheapest way to stay 'e'-free is to say less, so completeness lags letter-compliance. **Letter-compliant % vs Pure-English %**: the mask bans 'e'-tokens but not other scripts, so the mask-only base model escapes into CJK/Cyrillic characters (e.g. 'running鞋', '帮助', 'вопросы') in ~39% of answers — each passes a naive no-'e' check but is not English. Kevin's adapter fixes this completely (100% pure English), which is a large and concrete part of why it is 'better' — a single usable-rate number only captures it indirectly, through the coherence judge. `Usable strict %` requires no-'e' AND English (>=90% Latin) AND the requirement met.

## By requirement type

| Config | items (req-met %) | items_with_reason (req-met %) | max_words (req-met %) | sentences (req-met %) |
|---|---|---|---|---|
| Base + prompt only | 100.0 | 88.2 | 100.0 | 100.0 |
| Base + token mask | 100.0 | 70.6 | 100.0 | 90.9 |
| Adapter (SFT+RL) + mask | 100.0 | 94.1 | 100.0 | 63.6 |

`items_with_reason` is where shrinking bites hardest: it needs several substantive items, so a short answer cannot satisfy it even when it is perfectly 'e'-free.

## Uncertainty — bootstrap over prompts (n=41, 2000 resamples)

Point estimate with 95% CI, resampling the prompts with replacement (Kevin uses the same paired-bootstrap approach). This says whether a gap survives the small sample size.

| Config | letter | pure_english | requirement | usable_strict |
|---|---|---|---|---|
| Base + prompt only | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 95.1 [87.8, 100.0] | 0.0 [0.0, 0.0] |
| Base + token mask | 100.0 [100.0, 100.0] | 61.0 [46.3, 75.6] | 85.4 [73.2, 95.1] | 82.9 [70.7, 92.7] |
| Adapter (SFT+RL) + mask | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 87.8 [78.0, 97.6] | 87.8 [75.6, 97.6] |

**Paired difference: adapter (SFT+RL) − mask-only**, same resampled prompts:

| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |
|---|---|---|---|
| letter | 0.0 | [0.0, 0.0] | no |
| pure_english | 39.0 | [24.4, 53.7] | **yes** |
| requirement | 2.4 | [-9.8, 14.6] | no |
| usable_strict | 4.9 | [-9.8, 17.1] | no |

Takeaway: at n=41 the adapter's **only** statistically-robust edge over the plain mask is **English purity** (+39 pts, CI excludes 0). Its requirement-completion and usable-strict gains are within noise. The adapter does not reliably answer *more completely* than a bare mask here — it answers *in English* instead of escaping into another script. A single usable-rate number blends these and overstates the completeness story.

## Failure examples A — script escape (no 'e', but not English)

Letter-compliant answers that dodge the mask by switching scripts. A naive no-'e' gate scores these as wins:

### Base + token mask (1/41 answers)

- **p29** (english_ratio=0.891): Music can transform a mood by activating brain pathways that link rhythm to情绪. It can shift a mood by changing brain activity in parts of 大脑 that control情绪 and 专注. Music can shift a mood by 释放 情绪  and 促进  心情 变化.

## Failure examples B — incomplete (letter-compliant but requirement NOT met)

The shrinking/cheating mode: 'e'-free, but the requested items/sentences are not actually there:

### Base + token mask

- **p35** (sentences, 23 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "atleast"}`
  - prompt: Pick a book or a film for a long flight, say which, and in 2 sentences say why. Avoid the letter e.
  - answer: *Inspirational film: *Parasita*   ⏎ It's a gripping story of class division and survival, showing how far a family will go to climb social stairs.
- **p09** (items_with_reason, 55 words) — requirement detail: `{"satisfied": false, "found_items": 4, "items_with_reason": 3, "target": 4, "mode": "exactly", "min_reason_words": 4}`
  - prompt: Suggest 4 tips for a first job interview. For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e.
  - answer: 1. **Dont bring a laptop**   ⏎    It can distract from your focus and look unimportant. ⏎  ⏎ 2. **Arrival on时间**   ⏎    Shows punctuality and a good work habit. ⏎  ⏎ 3. **Dont talk about your past**   ⏎    It may not show your skills or fit for this position. ⏎  ⏎ 4. **Dont ask too many вопросы**   ⏎    It can show lack of自信 and unavailability to work.
- **p11** (items_with_reason, 55 words) — requirement detail: `{"satisfied": false, "found_items": 4, "items_with_reason": 1, "target": 4, "mode": "exactly", "min_reason_words": 4}`
  - prompt: Suggest 4 ways to study for a hard math class. For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e.
  - answer: 1. **Study in small groups**   ⏎    This allows for discussion and clarification of difficult topics, making it更容易 to grasp difficult概念. ⏎  ⏎ 2. **Utilizing flashcards**   ⏎    This aids in quick复习 and long-lasting记忆 of formulas and important概念. ⏎  ⏎ 3. **Solving similar past**   ⏎    This builds familiarity with common题型 and提高解决问题的能力. ⏎  ⏎ 4. **Taking short, daily复习**   ⏎    This maintains focus 

### Adapter (SFT+RL) + mask

- **p30** (sentences, 10 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "exactly"}`
  - prompt: In exactly 2 sentences, say why warm soup is good on a cold night. Avoid the letter e.
  - answer: Warm soup brings comfort and warmth on a chilly night.
- **p26** (sentences, 13 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "exactly"}`
  - prompt: In exactly 2 sentences, say what makes a city fun to visit. Avoid the letter e.
  - answer: A mix of sights and sounds, plus good food and a warm crowd.
- **p35** (sentences, 15 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "atleast"}`
  - prompt: Pick a book or a film for a long flight, say which, and in 2 sentences say why. Avoid the letter e.
  - answer: A Clockwork Orphan ⏎  ⏎ It is a story of loss and finding comfort in small things.

## AI-generated proxy review (Claude) — coherence & responsiveness (1-5)

Fixed 9-prompt sample per config, spanning all requirement types, scored by Claude, not a human. Lipogram spellings and factual errors were not penalised. These provisional scores are not independently judge-validated; independent review is still needed. See `results/manual_review_scored.csv`.

| Config | n | Mean coherence | Mean responsiveness |
|---|---|---|---|
| Base + token mask | 9 | 3.33 | 3.78 |
| Adapter (SFT+RL) + mask | 9 | 4.89 | 5.0 |

The provisional Claude scores suggest a qualitative gain (not independently validated): it comes from fluency and staying in English (coherence 3.33 -> 4.89), not from being more structurally complete. On the deterministic requirement axis the adapter actually *regresses* on exact-sentence-count prompts (see the by-type table) even as it improves on list/reason prompts.
