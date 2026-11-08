"""Bootstrap confidence intervals over prompts, and paired config differences.

Mirrors the rigor in Kevin Wu's writeup (paired bootstrap over prompts, 1000
resamples) so reported gaps come with uncertainty, not just point estimates.

For each config we report mean +/- 95% CI for the key rates. We also report the
PAIRED difference adapter_mask - base_mask for each metric: resample the shared
set of prompt ids with replacement, recompute both configs on the same resample,
take the difference. A CI that excludes 0 means the gap is unlikely to be noise
at this sample size (n=41).
"""
from __future__ import annotations
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCORED = os.path.join(ROOT, "results", "scored")

METRICS = {
    "letter": lambda s: s["letter_compliant"],
    "pure_english": lambda s: s.get("pure_latin", s["looks_english"]),
    "requirement": lambda s: s["requirement"]["satisfied"],
    "usable_strict": lambda s: s["usable_strict"],
}


def load(cfg):
    path = os.path.join(SCORED, f"{cfg}.jsonl")
    rows = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                rows[r["id"]] = r["scores"]
    return rows


def ci(vals, lo=2.5, hi=97.5):
    vals = sorted(vals)
    n = len(vals)
    return (round(vals[int(n * lo / 100)], 1), round(vals[int(min(n - 1, n * hi / 100))], 1))


def main(configs, iters=2000, seed=0):
    random.seed(seed)
    data = {c: load(c) for c in configs}
    ids = sorted(set.intersection(*[set(d) for d in data.values()])) if data else []
    out = {"n_prompts": len(ids), "iters": iters, "per_config": {}, "paired_adapter_minus_mask": {}}

    # per-config CIs
    for c in configs:
        out["per_config"][c] = {}
        for mname, fn in METRICS.items():
            point = 100.0 * sum(fn(data[c][i]) for i in ids) / len(ids)
            boot = []
            for _ in range(iters):
                samp = [random.choice(ids) for _ in ids]
                boot.append(100.0 * sum(fn(data[c][i]) for i in samp) / len(samp))
            out["per_config"][c][mname] = {"mean": round(point, 1), "ci95": ci(boot)}

    # paired difference adapter_mask - base_mask (if both present)
    if "adapter_mask" in data and "base_mask" in data:
        a, b = data["adapter_mask"], data["base_mask"]
        for mname, fn in METRICS.items():
            point = 100.0 * (sum(fn(a[i]) for i in ids) - sum(fn(b[i]) for i in ids)) / len(ids)
            boot = []
            for _ in range(iters):
                samp = [random.choice(ids) for _ in ids]
                d = sum(fn(a[i]) - fn(b[i]) for i in samp) / len(samp)
                boot.append(100.0 * d)
            lo, hi = ci(boot)
            out["paired_adapter_minus_mask"][mname] = {
                "mean_diff": round(point, 1), "ci95": (lo, hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }

    with open(os.path.join(ROOT, "results", "bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["base_prompt", "base_mask", "adapter_mask"])
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    main(a.configs, a.iters)
