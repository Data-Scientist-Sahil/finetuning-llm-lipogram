"""Verifiable reward for GRPO — no LLM judge in the loop (RLVR).

This is the core of Idea 2. It turns the SAME axes the benchmark measures into a
dense, deterministic training signal, so the reward is free, reproducible, and
directly targets the thing Idea 1 found the judge-reward does NOT reliably move:
requirement-completion.

Shape (all in [0, 1]):
  hard gate   : not 'e'-free            -> 0.0   (the lipogram constraint)
  hard gate   : script-escape (CJK/etc) -> 0.0   (english_ratio < 0.85)
  hard gate   : degenerate/too short    -> 0.0   (< 3 words)
  otherwise   : 0.15 baseline
              + 0.70 * completeness_fraction   (continuous: partial credit)
              - repetition_penalty (up to 0.25)

Design choices that matter:
- **Continuous** completeness (not 0/1) so GRPO gets a gradient even when no
  rollout is fully correct — critical early in training.
- **Hard english gate** because Idea 1 showed the mask-only model escapes into
  CJK ~39% of the time; we refuse to reward that at all.
- **Repetition penalty** because Kevin's first RL reward-hacked into letter/word
  streams; this is our explicit anti-hacking guard (different from his failed
  min()/length-norm tweaks, which were still judge-based).

Reward fn signature matches trl GRPOTrainer AND our own minimal loop:
    rewards = reward_batch(completions, reqs)
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from metrics import (letter_compliance, english_ratio, parse_numbered_items,  # noqa: E402
                     split_sentences, word_count)

BASELINE = 0.15
W_COMPLETE = 0.70
W_REP_MAX = 0.25
MIN_WORDS = 3
ENGLISH_GATE = 0.85


def completeness_fraction(text: str, req: dict) -> float:
    """Continuous [0,1] version of requirement_completion: partial credit for
    getting part of the structural ask, penalty for overshooting an 'exactly' N."""
    t = req.get("type")
    if t in ("items", "items_with_reason"):
        n = req["n"]
        items = parse_numbered_items(text)
        if t == "items_with_reason":
            mrw = req.get("min_reason_words", 4)
            good = sum(1 for _, it in items if word_count(it) >= mrw)
        else:
            good = len(items)
        frac = min(good, n) / n
        # penalise overshoot for exactly-N (too many items is also wrong)
        if req.get("mode", "exactly") == "exactly" and len(items) > n:
            frac -= 0.25 * (len(items) - n)
        return max(0.0, min(1.0, frac))
    if t == "sentences":
        n = req["n"]
        c = len(split_sentences(text))
        if req.get("mode", "exactly") == "atleast":
            return 1.0 if c >= n else c / n
        return max(0.0, 1.0 - abs(c - n) / max(n, 1))  # closeness to exact count
    if t == "max_words":
        k = req["k"]
        w = word_count(text)
        if w == 0:
            return 0.0
        return 1.0 if w <= k else max(0.0, 1.0 - (w - k) / k)
    return 0.0


def expected_min_words(req: dict) -> int:
    """Rough floor on how many words a genuinely complete answer needs, so that a
    too-short answer cannot claim full completeness credit (the brevity trap seen in
    run 1: the model got shorter and lost completeness). No floor for max_words."""
    t = req.get("type")
    if t == "items_with_reason":
        return req["n"] * (req.get("min_reason_words", 4) + 1)
    if t == "items":
        return req["n"] * 2
    if t == "sentences":
        return req["n"] * 4
    return 0  # max_words / unknown: brevity is fine

def length_adequacy(text: str, req: dict) -> float:
    """[0,1] — scales down completeness when the answer is far shorter than a
    complete one would be. Capped at 1 so it never rewards padding; the repetition
    penalty guards the padding side."""
    em = expected_min_words(req)
    if em <= 0:
        return 1.0
    return min(1.0, word_count(text) / em)


_WORD = re.compile(r"[A-Za-z0-9']+")

def repetition_penalty(text: str) -> float:
    """0..W_REP_MAX. Penalise low distinct-token ratio and repeated trigrams —
    catches the 'stream of tokens' and 'say the same clause 3x' hacks."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 4:
        return 0.0
    distinct_ratio = len(set(words)) / len(words)
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    tri_rep = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
    pen = (1.0 - distinct_ratio) * 0.6 + tri_rep * 0.4
    return min(W_REP_MAX, pen * W_REP_MAX / 0.5)


def reward_one(text: str, req: dict) -> float:
    if not letter_compliance(text):
        return 0.0
    if english_ratio(text) < ENGLISH_GATE:
        return 0.0
    if word_count(text) < MIN_WORDS:
        return 0.0
    # completeness, scaled by length-adequacy so brevity is not a free win
    comp = completeness_fraction(text, req) * length_adequacy(text, req)
    r = BASELINE + W_COMPLETE * comp - repetition_penalty(text)
    return max(0.0, min(1.0, r))


def reward_batch(completions, reqs) -> list[float]:
    return [reward_one(c, q) for c, q in zip(completions, reqs)]


# trl GRPOTrainer adapter: reward_funcs=[trl_reward]; trl passes the dataset's
# 'req' column through **kwargs.
def trl_reward(prompts=None, completions=None, req=None, **kwargs) -> list[float]:
    comps = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
    return reward_batch(comps, req)


if __name__ == "__main__":
    tests = [
        # (text, req, expect_high?)
        ("1. A sun hat, so your scalp stays cool.\n2. A big jug, to carry your liquid.\n3. Good shos, to avoid any slip.",
         {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}, True),
        ("1. Shos. 2. Top. 3. Jug.",  # short, no reasons
         {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}, False),
        ("A dog brings joy. It shows a child what caring is.",
         {"type": "sentences", "n": 2, "mode": "exactly"}, True),
        ("A dog brings joy.",  # only 1 of 2 sentences
         {"type": "sentences", "n": 2, "mode": "exactly"}, False),
        ("running鞋 that fits your foot and帮助 you run",  # script escape
         {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}, False),
        ("joy joy joy joy joy joy joy joy joy joy",  # repetition hack
         {"type": "sentences", "n": 2, "mode": "exactly"}, False),
        ("This dog has an eye.",  # uses 'e'
         {"type": "sentences", "n": 1, "mode": "atleast"}, False),
        ("Joy. Fun. Good.",  # right sentence COUNT but trivially short -> length floor bites
         {"type": "sentences", "n": 3, "mode": "exactly"}, False),
    ]
    for text, req, hi in tests:
        r = reward_one(text, req)
        flag = "OK " if ((r > 0.6) == hi) else "!! "
        print(f"{flag} r={r:.3f} expect_high={hi}  {text[:55]!r}")
