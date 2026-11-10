# No-'e' lipogram benchmark — results

Configs: `Qwen/Qwen3-1.7B` with a token mask (base), Kevin Wu's SFT-only start (`sft180`), Kevin Wu's released `k33wu/qwen3-1.7b-no-e` SFT+RL adapter, and `trained` = Run C, a 100-step GRPO-lite run with our verifiable reward, KL against a frozen SFT-init adapter, beta 0.03, started from `sft180` (results/idea2_runC/). Run B numbers are kept in the README.

Each axis is measured **independently**. `letter_compliance` = answer has no 'e'. `requirement_completion` = the explicit structural ask (N items each with a reason, a sentence count, a word limit) was actually met. `usable` = both at once (this is the closest analogue to Kevin's single usable-rate gate, before coherence). Coherence/responsiveness sample scores were assigned by Claude as AI-generated proxy review, not independent human review, and are not independently judge-validated. No judge API cross-check ran. Legacy manual_* filenames are retained; manual_review.csv is an unscored worksheet.

## Headline: the axes come apart

| Config | n | Letter-compliant % | Pure-English % | Requirement-met % | Usable lenient % | Usable strict % | Mean words | Median words |
|---|---|---|---|---|---|---|---|---|
| Base + token mask | 41 | 100.0 | 61.0 | 85.4 | 85.4 | 82.9 | 39.537 | 33 |
| SFT-180 start + mask | 41 | 100.0 | 100.0 | 97.6 | 97.6 | 97.6 | 52.366 | 45 |
| Kevin's adapter (SFT+RL) + mask | 41 | 100.0 | 100.0 | 87.8 | 87.8 | 87.8 | 34.707 | 27 |
| Trained (our RLVR) + mask | 41 | 100.0 | 100.0 | 92.7 | 92.7 | 92.7 | 26.0 | 20 |

Two gaps tell the story. **Letter-compliant % vs Requirement-met %**: the cheapest way to stay 'e'-free is to say less, so completeness lags letter-compliance. **Letter-compliant % vs Pure-English %**: the mask bans 'e'-tokens but not other scripts, so the mask-only base model escapes into CJK/Cyrillic characters (e.g. 'running鞋', '帮助', 'вопросы') in ~39% of answers — each passes a naive no-'e' check but is not English. Kevin's adapter fixes this completely (100% pure English), which is a large and concrete part of why it is 'better' — a single usable-rate number only captures it indirectly, through the coherence judge. `Usable strict %` requires no-'e' AND English (>=90% Latin) AND the requirement met.

## By requirement type

| Config | items (req-met %) | items_with_reason (req-met %) | max_words (req-met %) | sentences (req-met %) |
|---|---|---|---|---|
| Base + token mask | 100.0 | 70.6 | 100.0 | 90.9 |
| SFT-180 start + mask | 100.0 | 100.0 | 66.7 | 100.0 |
| Kevin's adapter (SFT+RL) + mask | 100.0 | 94.1 | 100.0 | 63.6 |
| Trained (our RLVR) + mask | 100.0 | 88.2 | 66.7 | 100.0 |

`items_with_reason` is where shrinking bites hardest: it needs several substantive items, so a short answer cannot satisfy it even when it is perfectly 'e'-free.

## Uncertainty — bootstrap over prompts (n=41, 2000 resamples)

Point estimate with 95% CI, resampling the prompts with replacement (Kevin uses the same paired-bootstrap approach). This says whether a gap survives the small sample size.

| Config | letter | pure_english | requirement | usable_strict |
|---|---|---|---|---|
| Base + token mask | 100.0 [100.0, 100.0] | 61.0 [46.3, 75.6] | 85.4 [73.2, 95.1] | 82.9 [70.7, 92.7] |
| SFT-180 start + mask | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 97.6 [92.7, 100.0] | 97.6 [92.7, 100.0] |
| Kevin's adapter (SFT+RL) + mask | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 87.8 [78.0, 97.6] | 87.8 [75.6, 97.6] |
| Trained (our RLVR) + mask | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 92.7 [85.4, 100.0] | 92.7 [82.9, 100.0] |

**Paired difference: trained (our RLVR) − SFT-180 start — **controlled test of our reward** (same start, only our RL differs)** (same resampled prompts):

| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |
|---|---|---|---|
| letter | 0.0 | [0.0, 0.0] | no |
| pure_english | 0.0 | [0.0, 0.0] | no |
| requirement | -4.9 | [-12.2, 0.0] | no |
| usable_strict | -4.9 | [-12.2, 0.0] | no |

**Paired difference: Kevin's adapter (SFT+RL) − SFT-180 start — reference: his full RL from the same start** (same resampled prompts):

| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |
|---|---|---|---|
| letter | 0.0 | [0.0, 0.0] | no |
| pure_english | 0.0 | [0.0, 0.0] | no |
| requirement | -9.8 | [-22.0, 0.0] | no |
| usable_strict | -9.8 | [-22.0, 0.0] | no |

**Paired difference: trained (our RLVR) − Kevin's released adapter — vs his full pipeline (unmatched compute, not a fair fight)** (same resampled prompts):

| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |
|---|---|---|---|
| letter | 0.0 | [0.0, 0.0] | no |
| pure_english | 0.0 | [0.0, 0.0] | no |
| requirement | 4.9 | [-4.9, 17.1] | no |
| usable_strict | 4.9 | [-7.3, 17.1] | no |

**Paired difference: Kevin's adapter − mask-only** (same resampled prompts):

| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |
|---|---|---|---|
| letter | 0.0 | [0.0, 0.0] | no |
| pure_english | 39.0 | [24.4, 53.7] | **yes** |
| requirement | 2.4 | [-9.8, 14.6] | no |
| usable_strict | 4.9 | [-7.3, 17.1] | no |

Read `trained − SFT-180` as the headline: it isolates whether OUR verifiable reward improved the model over its own starting point, with no confound from Kevin's extra RL. `trained − Kevin's adapter` is context only — his adapter has far more RL, so losing there is expected and not the claim. A completeness gain with a CI that excludes 0 on `requirement` / `items_with_reason` / `sentences` is the win; a CI straddling 0 at this scale (40-100 steps) means validate-and-scale, not failure.

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

### SFT-180 start + mask

- **p37** (max_words, 13 words) — requirement detail: `{"satisfied": false, "found_words": 13, "limit": 8}`
  - prompt: In 8 words or fewer, sum up why kids should nap. Avoid the letter e.
  - answer: Kids grow with naps. Naps aid growth, calm minds, and aid daily tasks.

### Kevin's adapter (SFT+RL) + mask

- **p30** (sentences, 10 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "exactly"}`
  - prompt: In exactly 2 sentences, say why warm soup is good on a cold night. Avoid the letter e.
  - answer: Warm soup brings comfort and warmth on a chilly night.
- **p26** (sentences, 13 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "exactly"}`
  - prompt: In exactly 2 sentences, say what makes a city fun to visit. Avoid the letter e.
  - answer: A mix of sights and sounds, plus good food and a warm crowd.
- **p35** (sentences, 15 words) — requirement detail: `{"satisfied": false, "found_sentences": 1, "target": 2, "mode": "atleast"}`
  - prompt: Pick a book or a film for a long flight, say which, and in 2 sentences say why. Avoid the letter e.
  - answer: A Clockwork Orphan ⏎  ⏎ It is a story of loss and finding comfort in small things.

### Trained (our RLVR) + mask

- **p09** (items_with_reason, 24 words) — requirement detail: `{"satisfied": false, "found_items": 4, "items_with_reason": 3, "target": 4, "mode": "exactly", "min_reason_words": 4}`
  - prompt: Suggest 4 tips for a first job interview. For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e.
  - answer: 1. Know your job and its limits. ⏎ 2. Ask about what you'll do. ⏎ 3. Say why you want this job. ⏎ 4. Finish with thanks.
- **p39** (items_with_reason, 41 words) — requirement detail: `{"satisfied": false, "found_items": 3, "items_with_reason": 2, "target": 3, "mode": "exactly", "min_reason_words": 4}`
  - prompt: Suggest 3 ways to stay warm on a cold walk. For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e.
  - answer: 1. Put on thick wool clothing. ⏎ Warm clothing stops cold air from blowing through gaps. ⏎  ⏎ 2. Walk slowly. ⏎ Slow motion allows body warmth to stay in motion. ⏎  ⏎ 3. Stop for a short drink. ⏎ A warm drink stops circulation and brings comfort.
- **p37** (max_words, 160 words) — requirement detail: `{"satisfied": false, "found_words": 160, "limit": 8}`
  - prompt: In 8 words or fewer, sum up why kids should nap. Avoid the letter e.
  - answer: Kids grow with shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut shut 

## AI-generated proxy review (Claude) — coherence & responsiveness (1-5)

Fixed 9-prompt sample per config, spanning all requirement types, scored by Claude, not a human. Lipogram spellings and factual errors were not penalised. These provisional scores are not independently judge-validated; independent review is still needed. See `results/manual_review_scored.csv`.

| Config | n | Mean coherence | Mean responsiveness |
|---|---|---|---|
| Base + token mask | 9 | 3.33 | 3.78 |
| Kevin's adapter (SFT+RL) + mask | 9 | 4.89 | 5.0 |

The provisional Claude scores suggest a qualitative gain (not independently validated): it comes from fluency and staying in English (coherence 3.33 -> 4.89), not from being more structurally complete. On the deterministic requirement axis the adapter actually *regresses* on exact-sentence-count prompts (see the by-type table) even as it improves on list/reason prompts.
