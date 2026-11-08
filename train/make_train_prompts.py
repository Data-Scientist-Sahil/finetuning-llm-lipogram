"""Build a TRAINING prompt set, disjoint from the 41-prompt eval set.

Same structure and requirement-spec format as prompts/eval_prompts.jsonl, so the
reward (train/reward.py) and the benchmark (src/metrics.py) share one parser and
one notion of "requirement met" — no train/eval metric mismatch. Topics are a
different, larger pool; any prompt text colliding with the eval set is dropped.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIST_SUFFIX = " Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e."
REASON_SUFFIX = " For each one, add a short reason. Number each point with a digit (1, 2, 3, ...) and keep the whole answer free of the letter e."

SUGGEST_TOPICS = [
    "ways to wake up on time", "ways to drink more liquid in a day", "things to pack for a hike",
    "ways to calm a loud room", "snacks for a long study day", "ways to budget on a small salary",
    "ways to stay off your phone", "ways to look good on a photo", "gifts for a busy mom",
    "ways to tidy a messy car", "habits for a good night of rest", "ways to kick off a morning run",
    "ways to pick a gift for a boss", "ways to try a bold hair look", "small plants for a windowsill",
    "ways to build a daily habit", "ways to study a foraign tongue".replace("foraign","foreign"),
    "ways to spot a bad landlord", "tips for a smooth train ride", "ways to stay warm at night",
    "ways to pick a strong wifi plan", "things to do on a layover", "ways to grow a small savings pot",
    "ways to split a bill fairly", "tasks to do prior to a trip", "ways to pick a ripe mango",
]
NAME_TOPICS = [
    "fruits a kid might munch on", "animals you spot at a zoo", "tools in a toolbox",
    "drinks you buy at a cafv".replace("cafv","cafe"), "sports with a ball", "toys for a baby",
    "shops you find at a mall", "flowers in a spring garden", "parts of a car", "big rivers of Asia",
    "pizza toppings", "jobs on a farm", "birds in a city park", "things in a first-aid kit",
    "musical bands you know", "boats that sail an ocean", "mountains you could climb",
]
SENTENCE_TASKS = [
    "say why warm milk aids sound sleep", "say why walking is good for you", "talk about why art is fun",
    "say what a good pal brings to your day", "talk about how rain can calm a town",
    "say why a library is worth a visit", "say what a tidy room can do for focus",
    "talk about why a stray dog saw joy", "say why kids should run outdoors",
    "say what a long bath can do for you", "talk about why sunlight lifts a mood",
    "say why a strong soup warms a cold day", "say what maps did for old sailors",
]
STEP_TASKS = [
    "brush a dog's coat", "fold a cotton shirt", "cook plain rbut".replace("rut","rice"),
    "wrap a book as a gift", "pour a cup of tea without a spill".replace("tea","tai"),
    "paint a small wall", "pack a bag for school", "plant a row of pumpkin",
    "fix a flat bicycl tyr".replace("bicycl tyr","bicycle tyre"), "mop a dirty floor",
]
BRIEF_TASKS = [
    "sum up why books matter", "sum up why naps aid a child", "coin a motto for a gym",
    "sum up a good morning plan", "coin a slogan for a farm stall", "sum up why kids should play out",
]


def build():
    P = []
    def add(cat, prompt, req):
        P.append({"category": cat, "prompt": prompt, "req": req})

    for t in SUGGEST_TOPICS:
        n = random.choice([3, 3, 4, 5])
        add("suggestions", f"Suggest {n} {t}." + REASON_SUFFIX,
            {"type": "items_with_reason", "n": n, "mode": "exactly", "min_reason_words": 4})
    for t in NAME_TOPICS:
        n = random.choice([3, 4, 5])
        add("list", f"Name {n} {t}." + LIST_SUFFIX, {"type": "items", "n": n, "mode": "exactly"})
    for t in SENTENCE_TASKS:
        n = random.choice([2, 2, 3])
        add("explain", f"In exactly {n} sentences, {t}. Avoid the letter e.",
            {"type": "sentences", "n": n, "mode": "exactly"})
    for t in STEP_TASKS:
        n = random.choice([3, 4, 5])
        add("steps", f"Walk through how to {t}, in {n} clear steps." + LIST_SUFFIX,
            {"type": "items", "n": n, "mode": "exactly"})
    for t in BRIEF_TASKS:
        k = random.choice([5, 6, 8])
        add("brief", f"In {k} words or fewer, {t}. Avoid the letter e.", {"type": "max_words", "k": k})
    return P


def main(seed=0):
    random.seed(seed)
    P = build()
    # drop any that collide with the eval set
    eval_path = os.path.join(ROOT, "prompts", "eval_prompts.jsonl")
    eval_texts = set()
    if os.path.exists(eval_path):
        eval_texts = {json.loads(l)["prompt"] for l in open(eval_path) if l.strip()}
    seen = set()
    rows = []
    for p in P:
        if p["prompt"] in eval_texts or p["prompt"] in seen:
            continue
        seen.add(p["prompt"])
        p["id"] = f"t{len(rows)+1:03d}"
        rows.append(p)
    # sanity: ascii only
    for r in rows:
        r["prompt"].encode("ascii")
    out = os.path.join(ROOT, "prompts", "train_prompts.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "category": r["category"],
                                "prompt": r["prompt"], "req": r["req"]}, ensure_ascii=True) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} train prompts -> {out}")
    print("by req type:", dict(Counter(r["req"]["type"] for r in rows)))
    print("overlap with eval:", len({r['prompt'] for r in rows} & eval_texts))


if __name__ == "__main__":
    main()
