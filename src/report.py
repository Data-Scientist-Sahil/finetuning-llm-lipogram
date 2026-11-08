"""Render results/REPORT.md from summary.json + scored generations.

Produces the headline per-axis table, a per-requirement-type breakdown, and a
set of failure examples chosen to illustrate the 'cheating' mode (letter-
compliant but requirement-failing, especially short answers).
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SCORED_DIR = os.path.join(RES, "scored")

CONFIG_LABEL = {
    "base_prompt": "Base + prompt only",
    "base_mask": "Base + token mask",
    "adapter_mask": "Adapter (SFT+RL) + mask",
    "sft_mask": "Adapter (SFT only) + mask",
}


def load_scored(cfg):
    path = os.path.join(SCORED_DIR, f"{cfg}.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    return rows


def main():
    with open(os.path.join(RES, "summary.json")) as f:
        summary = json.load(f)
    configs = list(summary.keys())
    L = []
    L.append("# No-'e' lipogram benchmark — results\n")
    L.append("These results evaluate Kevin Wu's released `k33wu/qwen3-1.7b-no-e` adapter on `Qwen/Qwen3-1.7B`, not a model trained by Sahil. The experimental GRPO training kit has NOT RUN; no new trained-model results are included.\n")
    L.append("Each axis is measured **independently**. `letter_compliance` = answer has no 'e'. "
             "`requirement_completion` = the explicit structural ask (N items each with a reason, "
             "a sentence count, a word limit) was actually met. `usable` = both at once "
             "(this is the closest analogue to Kevin's single usable-rate gate, before coherence). "
             "Coherence/responsiveness sample scores were assigned by Claude as AI-generated proxy review, "
             "not independent human review, and are not independently judge-validated. No judge API cross-check ran. "
             "Legacy manual_* filenames are retained; manual_review.csv is an unscored worksheet.\n")

    # headline table
    L.append("## Headline: the axes come apart\n")
    L.append("| Config | n | Letter-compliant % | Pure-English % | Requirement-met % | Usable lenient % | Usable strict % | Mean words | Median words |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        s = summary[c]
        L.append(f"| {CONFIG_LABEL.get(c, c)} | {s['n']} | {s['letter_compliance_pct']} | "
                 f"{s.get('pure_latin_pct','-')} | {s['requirement_completion_pct']} | "
                 f"{s['usable_req_and_letter_pct']} | {s.get('usable_strict_pct','-')} | "
                 f"{s['mean_words']} | {s['median_words']} |")
    L.append("")
    L.append("Two gaps tell the story. **Letter-compliant % vs Requirement-met %**: the cheapest way to stay "
             "'e'-free is to say less, so completeness lags letter-compliance. **Letter-compliant % vs "
             "Pure-English %**: the mask bans 'e'-tokens but not other scripts, so the mask-only base model "
             "escapes into CJK/Cyrillic characters (e.g. 'running鞋', '帮助', 'вопросы') in ~39% of answers — "
             "each passes a naive no-'e' check but is not English. Kevin's adapter fixes this completely "
             "(100% pure English), which is a large and concrete part of why it is 'better' — a single "
             "usable-rate number only captures it indirectly, through the coherence judge. `Usable strict %` "
             "requires no-'e' AND English (>=90% Latin) AND the requirement met.\n")

    # per requirement type
    L.append("## By requirement type\n")
    types = sorted({t for c in configs for t in summary[c]["by_requirement_type"]})
    L.append("| Config | " + " | ".join(f"{t} (req-met %)" for t in types) + " |")
    L.append("|---|" + "|".join("---" for _ in types) + "|")
    for c in configs:
        bt = summary[c]["by_requirement_type"]
        cells = [str(bt[t]["requirement_pct"]) if t in bt else "-" for t in types]
        L.append(f"| {CONFIG_LABEL.get(c, c)} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("`items_with_reason` is where shrinking bites hardest: it needs several substantive items, "
             "so a short answer cannot satisfy it even when it is perfectly 'e'-free.\n")

    # bootstrap CIs
    bpath = os.path.join(RES, "bootstrap.json")
    if os.path.exists(bpath):
        bs = json.load(open(bpath))
        L.append(f"## Uncertainty — bootstrap over prompts (n={bs['n_prompts']}, {bs['iters']} resamples)\n")
        L.append("Point estimate with 95% CI, resampling the prompts with replacement (Kevin uses the same "
                 "paired-bootstrap approach). This says whether a gap survives the small sample size.\n")
        order = ["letter", "pure_english", "requirement", "usable_strict"]
        L.append("| Config | " + " | ".join(order) + " |")
        L.append("|---|" + "|".join("---" for _ in order) + "|")
        for c in configs:
            if c in bs["per_config"]:
                cells = []
                for m in order:
                    d = bs["per_config"][c][m]
                    cells.append(f"{d['mean']} [{d['ci95'][0]}, {d['ci95'][1]}]")
                L.append(f"| {CONFIG_LABEL.get(c, c)} | " + " | ".join(cells) + " |")
        L.append("")
        if bs.get("paired_adapter_minus_mask"):
            L.append("**Paired difference: adapter (SFT+RL) − mask-only**, same resampled prompts:\n")
            L.append("| Metric | Δ (points) | 95% CI | Significant (CI excludes 0)? |")
            L.append("|---|---|---|---|")
            for m in order:
                d = bs["paired_adapter_minus_mask"][m]
                L.append(f"| {m} | {d['mean_diff']} | [{d['ci95'][0]}, {d['ci95'][1]}] | "
                         f"{'**yes**' if d['excludes_zero'] else 'no'} |")
            L.append("")
            L.append("Takeaway: at n=41 the adapter's **only** statistically-robust edge over the plain mask is "
                     "**English purity** (+39 pts, CI excludes 0). Its requirement-completion and usable-strict "
                     "gains are within noise. The adapter does not reliably answer *more completely* than a bare "
                     "mask here — it answers *in English* instead of escaping into another script. A single "
                     "usable-rate number blends these and overstates the completeness story.\n")

    # failure examples: CJK script-escape
    L.append("## Failure examples A — script escape (no 'e', but not English)\n")
    L.append("Letter-compliant answers that dodge the mask by switching scripts. A naive no-'e' gate scores these as wins:\n")
    for c in configs:
        rows = load_scored(c)
        esc = [r for r in rows if r["scores"]["letter_compliant"] and not r["scores"]["looks_english"]]
        esc.sort(key=lambda r: r["scores"]["english_ratio"])
        if not esc:
            continue
        L.append(f"### {CONFIG_LABEL.get(c, c)} ({len(esc)}/{len(rows)} answers)\n")
        for r in esc[:2]:
            ans = r["answer"].replace("\n", " ⏎ ")
            L.append(f"- **{r['id']}** (english_ratio={r['scores']['english_ratio']}): {ans[:350]}")
        L.append("")

    # failure examples: letter-compliant but requirement-failed
    L.append("## Failure examples B — incomplete (letter-compliant but requirement NOT met)\n")
    L.append("The shrinking/cheating mode: 'e'-free, but the requested items/sentences are not actually there:\n")
    for c in configs:
        rows = load_scored(c)
        fails = [r for r in rows
                 if r["scores"]["letter_compliant"] and not r["scores"]["requirement"]["satisfied"]]
        fails.sort(key=lambda r: r["scores"]["length"]["words"])
        if not fails:
            continue
        L.append(f"### {CONFIG_LABEL.get(c, c)}\n")
        for r in fails[:3]:
            rq = r["scores"]["requirement"]
            L.append(f"- **{r['id']}** ({r['req']['type']}, {r['scores']['length']['words']} words) — "
                     f"requirement detail: `{json.dumps(rq)}`")
            L.append(f"  - prompt: {r['prompt']}")
            ans = r["answer"].replace("\n", " ⏎ ")
            L.append(f"  - answer: {ans[:400]}")
        L.append("")

    # recorded Claude proxy review if present
    man_path = os.path.join(RES, "manual_summary.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        L.append("## AI-generated proxy review (Claude) — coherence & responsiveness (1-5)\n")
        L.append("Fixed 9-prompt sample per config, spanning all requirement types, scored by Claude, not a human. "
                 "Lipogram spellings and factual errors were not penalised. These provisional scores are "
                 "not independently judge-validated; independent review is still needed. See "
                 "`results/manual_review_scored.csv`.\n")
        L.append("| Config | n | Mean coherence | Mean responsiveness |")
        L.append("|---|---|---|---|")
        for c in configs:
            if c in man:
                m = man[c]
                L.append(f"| {CONFIG_LABEL.get(c, c)} | {m['n']} | {m['mean_coherence']} | {m['mean_responsiveness']} |")
        L.append("")
        L.append("The provisional Claude scores suggest a qualitative gain (not independently validated): "
                 "it comes from fluency and staying in English (coherence 3.33 -> 4.89), not from being more "
                 "structurally complete. On the deterministic requirement axis the adapter actually *regresses* "
                 "on exact-sentence-count prompts (see the by-type table) even as it improves on list/reason prompts.\n")

    # automated judge (Groq/OpenAI/Anthropic) means, if run
    if any(summary[c].get("judge") for c in configs):
        L.append("## Automated judge (supplementary) — mean coherence & responsiveness (1-5)\n")
        L.append("Same rubric as the recorded proxy review (lipogram spellings and factual errors not penalised). "
                 "Separate API-judge scores; not independent human validation.\n")
        L.append("| Config | n judged | Mean coherence | Mean responsiveness |")
        L.append("|---|---|---|---|")
        for c in configs:
            j = summary[c].get("judge")
            if j:
                L.append(f"| {CONFIG_LABEL.get(c, c)} | {j['n_judged']} | {j['mean_coherence']} | {j['mean_responsiveness']} |")
        L.append("")

    out = os.path.join(RES, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
