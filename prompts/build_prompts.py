"""Build the evaluation prompt set for the no-'e' (lipogram) benchmark.

Design goals
------------
Kevin Wu's released eval uses single-shot everyday prompts scored only by a
usable-rate metric (no-'e' AND coherent AND responsive, judged 1-5 by an LLM).
That metric can be satisfied by *shrinking* the answer: his median answer length
fell 52 -> 18 tokens, because a shorter answer is easier to keep coherent under
the token mask, and a (coherence=5, responsiveness=3) answer scores the same as
a (4,4) under his summed reward even though it drops half the question.

This benchmark is built to make that failure mode *visible and separable*. Every
prompt carries an explicit, programmatically-checkable requirement (e.g. "give
exactly 3 suggestions, each with a reason"). We then score four axes
INDEPENDENTLY rather than collapsing them into one pass/fail:
    1. letter_compliance    - contains no 'e'/'E' at all (deterministic)
    2. requirement_completion - did it actually satisfy the structural ask
                                (right number of substantive items/sentences)
    3. length               - words / chars / sentences produced
    4. coherence, responsiveness - 1-5, by human review and/or a pluggable judge

A model that "cheats" by answering short will pass letter_compliance and may
look locally coherent, but will FAIL requirement_completion. That is the whole
point of the harness.

Requirement spec types (checked by src/metrics.py):
    items_with_reason : {n, mode, min_reason_words}  numbered items each w/ a reason
    items             : {n, mode}                    numbered items, no reason needed
    sentences         : {n, mode}                    count of sentences
    max_words         : {k}                           answer must be <= k words

mode is "exactly" or "atleast".

Answers are instructed to number list items with digits (1, 2, 3) so that item
counting is robust under the lipogram constraint (spelled-out "three" contains
an 'e'; the numeral does not).

The prompts below were written for this benchmark and do not appear in Kevin's
released SFT data or eval set (which draw from Dolly / No Robots). Each is
answerable without the letter 'e' in principle.
"""
import json
import os

LIST_SUFFIX = " Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e."
REASON_SUFFIX = " For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e."

P = []

def add(cat, prompt, req):
    P.append({"id": f"p{len(P)+1:02d}", "category": cat, "prompt": prompt, "req": req})

# ----- items_with_reason: the headline category (shrinking is caught here) -----
add("suggestions", "I want to start running as a hobby. Suggest 3 things I should buy first." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 ways to stay calm during a long train journey." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 good habits for a tidy room." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 ways to cut down on plastic at a shop." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 fun things a child can do on a rainy day." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "I am hosting a small lunch. Suggest 3 dishes I could cook." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 ways to stay fit without a gym." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 low-cost gifts for a young kid." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 4 tips for a first job interview." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 4, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 4 ways to spend an afternoon in a big city." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 4, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 4 ways to study for a hard math class." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 4, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 4 small ways to cut monthly spending." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 4, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 5 topics for a short talk at school." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 5, "mode": "exactly", "min_reason_words": 3})
add("suggestions", "Suggest 5 ways to stay safe on a long road trip." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 5, "mode": "exactly", "min_reason_words": 3})

# ----- items: count only, no reason -----
add("list", "Name 3 sports that two kids can play in a park." + LIST_SUFFIX,
    {"type": "items", "n": 3, "mode": "exactly"})
add("list", "Name 3 colors that look good on a dark wall." + LIST_SUFFIX,
    {"type": "items", "n": 3, "mode": "exactly"})
add("list", "List 4 animals a kid might spot on a farm." + LIST_SUFFIX,
    {"type": "items", "n": 4, "mode": "exactly"})
add("list", "List 4 things to pack for a day at a hot, sandy spot by the ocean." + LIST_SUFFIX,
    {"type": "items", "n": 4, "mode": "exactly"})
add("list", "Name 5 jobs a town cannot run without." + LIST_SUFFIX,
    {"type": "items", "n": 5, "mode": "exactly"})
add("list", "List 5 countries with a long coast." + LIST_SUFFIX,
    {"type": "items", "n": 5, "mode": "exactly"})

# ----- steps (treated as ordered items) -----
add("steps", "Walk through how to wash a car, in 4 clear steps." + LIST_SUFFIX,
    {"type": "items", "n": 4, "mode": "exactly"})
add("steps", "Walk through how to plant a small shrub, in 4 clear steps." + LIST_SUFFIX,
    {"type": "items", "n": 4, "mode": "exactly"})
add("steps", "Walk through how to wrap a gift, in 3 clear steps." + LIST_SUFFIX,
    {"type": "items", "n": 3, "mode": "exactly"})
add("steps", "Walk through how to boil an egg, in 5 clear steps." + LIST_SUFFIX,
    {"type": "items", "n": 5, "mode": "exactly"})

# ----- sentences: ask for a specific count of full sentences -----
add("explain", "In exactly 2 sentences, say why a dog is a good pal for a child. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "exactly"})
add("explain", "In exactly 2 sentences, say what makes a city fun to visit. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "exactly"})
add("explain", "In exactly 3 sentences, talk about why books matter. Avoid the letter e.",
    {"type": "sentences", "n": 3, "mode": "exactly"})
add("explain", "In exactly 3 sentences, say what a good boss should do. Avoid the letter e.",
    {"type": "sentences", "n": 3, "mode": "exactly"})
add("explain", "In exactly 3 sentences, talk about how music can shift a mood. Avoid the letter e.",
    {"type": "sentences", "n": 3, "mode": "exactly"})
add("explain", "In exactly 2 sentences, say why warm soup is good on a cold night. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "exactly"})
add("explain", "In exactly 3 sentences, say what a first-grв class is for. Avoid the letter e.".replace("grв", "ade"),
    {"type": "sentences", "n": 3, "mode": "exactly"})
add("explain", "In exactly 4 sentences, tell a tiny story about a cat on a farm. Avoid the letter e.",
    {"type": "sentences", "n": 4, "mode": "exactly"})

# ----- two-part asks, counted as sentences -----
add("compare", "Pick a cat or a dog as a pal, say which, and in 2 sentences say why. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "atleast"})
add("compare", "Pick a bus or a car for a city trip, say which, and in 2 sentences say why. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "atleast"})
add("compare", "Pick a book or a film for a long flight, say which, and in 2 sentences say why. Avoid the letter e.",
    {"type": "sentences", "n": 2, "mode": "atleast"})

# ----- max_words: brevity requirement (opposite failure mode guard) -----
add("brief", "In 6 words or fewer, give a motto for a running club. Avoid the letter e.",
    {"type": "max_words", "k": 6})
add("brief", "In 8 words or fewer, sum up why kids should nap. Avoid the letter e.",
    {"type": "max_words", "k": 8})
add("brief", "In 5 words or fewer, name a good motto for a farm shop. Avoid the letter e.",
    {"type": "max_words", "k": 5})

# ----- a few more items_with_reason to round to 40 -----
add("suggestions", "Suggest 3 ways to stay warm on a cold walk." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 3 books a young adult might curl up with." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4})
add("suggestions", "Suggest 4 ways to turn a dull room into a cozy spot." + REASON_SUFFIX,
    {"type": "items_with_reason", "n": 4, "mode": "exactly", "min_reason_words": 4})

out = os.path.join(os.path.dirname(__file__), "eval_prompts.jsonl")
with open(out, "w") as f:
    for row in P:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

# sanity: no stray non-ascii crept into prompts
for row in P:
    row["prompt"].encode("ascii")

from collections import Counter
print(f"wrote {len(P)} prompts -> {out}")
print("by category:", dict(Counter(r["category"] for r in P)))
print("by req type:", dict(Counter(r["req"]["type"] for r in P)))
