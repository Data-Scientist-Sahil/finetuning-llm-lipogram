"""Score raw generations on every deterministic axis and aggregate.

Reads results/raw/<config>.jsonl (from generate.py), writes:
  results/scored/<config>.jsonl   per-answer scores
  results/summary.json            per-config and per-requirement-type aggregates
  results/manual_review.csv       sample for human coherence scoring

Run `python src/report.py` afterwards to render results/REPORT.md.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "results", "raw")
SCORED_DIR = os.path.join(ROOT, "results", "scored")

from metrics import score_answer  # noqa: E402
from judge import export_for_manual_review, judge_one, load_env  # noqa: E402


def load_raw(config: str) -> list[dict]:
    path = os.path.join(RAW_DIR, f"{config}.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def pct(xs):
    return round(100.0 * mean(xs), 1)


def aggregate(scored: list[dict]) -> dict:
    n = len(scored)
    letter = [r["scores"]["letter_compliant"] for r in scored]
    english = [r["scores"]["looks_english"] for r in scored]
    pure = [r["scores"].get("pure_latin", r["scores"]["looks_english"]) for r in scored]
    req = [r["scores"]["requirement"]["satisfied"] for r in scored]
    both = [r["scores"]["req_and_letter"] for r in scored]
    strict = [r["scores"]["usable_strict"] for r in scored]
    words = [r["scores"]["length"]["words"] for r in scored]
    gentok = [r.get("gen_tokens", 0) for r in scored]

    # median words
    sw = sorted(words)
    median_words = sw[len(sw) // 2] if sw else 0

    # automated judge means (only over answers that were judged)
    jc = [r["scores"]["judge"]["coherence"] for r in scored if "judge" in r["scores"]]
    jr = [r["scores"]["judge"]["responsiveness"] for r in scored if "judge" in r["scores"]]
    judge_block = {"n_judged": len(jc), "mean_coherence": mean(jc), "mean_responsiveness": mean(jr)} if jc else None

    by_type = defaultdict(lambda: {"n": 0, "req": [], "letter": [], "both": []})
    for r in scored:
        t = r["req"]["type"]
        by_type[t]["n"] += 1
        by_type[t]["req"].append(r["scores"]["requirement"]["satisfied"])
        by_type[t]["letter"].append(r["scores"]["letter_compliant"])
        by_type[t]["both"].append(r["scores"]["req_and_letter"])
    per_type = {t: {"n": d["n"], "letter_pct": pct(d["letter"]),
                    "requirement_pct": pct(d["req"]), "both_pct": pct(d["both"])}
                for t, d in by_type.items()}

    return {
        "n": n,
        "letter_compliance_pct": pct(letter),
        "english_pct": pct(english),
        "pure_latin_pct": pct(pure),
        "requirement_completion_pct": pct(req),
        "usable_req_and_letter_pct": pct(both),
        "usable_strict_pct": pct(strict),
        "mean_words": mean(words),
        "median_words": median_words,
        "mean_gen_tokens": mean(gentok),
        "judge": judge_block,
        "by_requirement_type": per_type,
    }


def main(configs, use_judge, judge_backend, manual_sample, judge_sample=None):
    os.makedirs(SCORED_DIR, exist_ok=True)
    if use_judge:
        p = load_env()
        print(f"[judge] backend={judge_backend} env={'loaded '+p if p else 'none'} "
              f"sample/config={judge_sample or 'all'}")
    summary = {}
    all_scored_for_review = []
    n_judged = 0
    for cfg in configs:
        raw = load_raw(cfg)
        if not raw:
            print(f"[skip] no raw results for {cfg}")
            continue
        scored = []
        judged_this_cfg = 0
        for r in raw:
            s = score_answer(r["answer"], r["req"])
            if use_judge and (judge_sample is None or judged_this_cfg < judge_sample):
                j = judge_one(r["prompt"], r["answer"], backend=judge_backend)
                if j:
                    s["judge"] = j
                    judged_this_cfg += 1
                    n_judged += 1
            r2 = dict(r)
            r2["scores"] = s
            scored.append(r2)
        with open(os.path.join(SCORED_DIR, f"{cfg}.jsonl"), "w") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        summary[cfg] = aggregate(scored)
        all_scored_for_review.extend(scored)
        print(f"[{cfg}] {summary[cfg]}")

    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if all_scored_for_review:
        csv_path = os.path.join(ROOT, "results", "manual_review.csv")
        export_for_manual_review(all_scored_for_review, csv_path, manual_sample)
        print("manual review csv ->", csv_path)
    print("summary ->", os.path.join(ROOT, "results", "summary.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["base_prompt", "base_mask", "adapter_mask"])
    ap.add_argument("--judge", action="store_true", help="also run the automated LLM judge (needs API key)")
    ap.add_argument("--judge-backend", default="groq", choices=["anthropic", "openai", "groq"])
    ap.add_argument("--judge-sample", type=int, default=None,
                    help="cap judged answers per config (controls API cost)")
    ap.add_argument("--manual-sample", type=int, default=None,
                    help="rows per config to export for human review (default: all)")
    a = ap.parse_args()
    main(a.configs, a.judge, a.judge_backend, a.manual_sample, a.judge_sample)
