"""Coherence / responsiveness judge — pluggable, and never the only signal.

The project brief is explicit: do not trust an AI judge alone. So this module
does two things:

  1. export_for_manual_review(): writes a CSV of (id, config, prompt, answer)
     with blank coherence/responsiveness columns for future independent review.
     The shipped sample scores were assigned by Claude, not a human; they have
     not been independently judge-validated.

  2. An optional automated judge (Anthropic or OpenAI) for scale. It needs an
     API key in the environment; if none is set it is simply skipped. The judge
     is calibrated the way Kevin's was: it is told NOT to penalise factual
     errors, only to rate (a) coherence as English and (b) whether the answer
     actually responds to the full request, each 1-5.

Both signals are kept SEPARATE from the deterministic axes and from each other.
"""
from __future__ import annotations
import csv
import json
import os
import re


def load_env(path: str | None = None):
    """Minimal .env loader (no dependency). Sets os.environ for KEY=VALUE lines.
    Looks at `path`, else ./.env, else repo-root/.env. Does not overwrite existing."""
    candidates = [path] if path else []
    candidates += [".env", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")]
    for p in candidates:
        if p and os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return p
    return None

JUDGE_SYSTEM = (
    "You rate short answers that are written WITHOUT the letter 'e' (a lipogram "
    "constraint). Odd word choices and missing-'e' spellings are expected and "
    "must NOT be penalised. Do not penalise factual errors. Rate only:\n"
    "coherence (1-5): does it read as fluent, grammatical English?\n"
    "responsiveness (1-5): does it fully do what the prompt asked, including any "
    "requested number of items or sentences?\n"
    "Reply with ONLY a JSON object: {\"coherence\": n, \"responsiveness\": n}."
)


def _judge_user(prompt: str, answer: str) -> str:
    return f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}\n\nScore it."


def export_for_manual_review(scored_rows: list[dict], out_csv: str, sample_per_config: int | None = None):
    """Write a CSV for human scoring. If sample_per_config is set, take the first
    N per config (ids are stable, so the sample is reproducible)."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    by_cfg: dict[str, list[dict]] = {}
    for r in scored_rows:
        by_cfg.setdefault(r["config"], []).append(r)
    rows = []
    for cfg, lst in by_cfg.items():
        lst = sorted(lst, key=lambda r: r["id"])
        rows.extend(lst[:sample_per_config] if sample_per_config else lst)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "config", "category", "prompt", "answer",
                    "letter_compliant", "requirement_satisfied",
                    "coherence_1to5", "responsiveness_1to5", "notes"])
        for r in rows:
            w.writerow([r["id"], r["config"], r["category"], r["prompt"], r["answer"],
                        r["scores"]["letter_compliant"],
                        r["scores"]["requirement"]["satisfied"], "", "", ""])
    return out_csv


def _parse_scores(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"coherence": int(d["coherence"]), "responsiveness": int(d["responsiveness"])}
    except Exception:
        return None


def judge_one(prompt: str, answer: str, backend: str = "anthropic", model: str | None = None) -> dict | None:
    """Return {'coherence','responsiveness'} or None if no key / failure."""
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic
            c = anthropic.Anthropic(api_key=key)
            r = c.messages.create(
                model=model or "claude-3-5-sonnet-latest", max_tokens=50,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": _judge_user(prompt, answer)}],
            )
            return _parse_scores(r.content[0].text)
        except Exception as e:
            print("judge error:", e)
            return None
    if backend in ("openai", "groq"):
        # Groq is OpenAI-compatible; just point the client at its base_url.
        if backend == "groq":
            key = os.environ.get("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
            default_model = "llama-3.3-70b-versatile"
        else:
            key = os.environ.get("OPENAI_API_KEY")
            base_url = None
            default_model = "gpt-4o-mini"
        if not key:
            return None
        try:
            from openai import OpenAI
            c = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
            r = c.chat.completions.create(
                model=model or default_model, max_tokens=50, temperature=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": _judge_user(prompt, answer)}],
            )
            return _parse_scores(r.choices[0].message.content)
        except Exception as e:
            print("judge error:", e)
            return None
    return None
