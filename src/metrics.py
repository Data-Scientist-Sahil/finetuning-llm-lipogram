"""Deterministic, per-axis scoring. No model required.

Four axes, scored INDEPENDENTLY (never collapsed into one number):

  letter_compliance   bool   answer contains no 'e'/'E'
  requirement_completion  dict  did it satisfy the explicit structural ask
  length              dict   words / chars / sentences
  (coherence/responsiveness are separate; shipped proxy scores are from Claude)

The requirement checkers are deliberately structural rather than semantic: a
model that "cheats" by answering short still passes letter_compliance and can
read as locally coherent, but it will FAIL requirement_completion because the
requested items / sentences are not actually there. That separation is the
whole contribution of this harness.
"""
from __future__ import annotations
import re

# ---------------------------------------------------------------- letter axis
_E = re.compile(r"[eE]")

def letter_compliance(text: str) -> bool:
    return _E.search(text or "") is None

# -------------------------------------------------------------- english axis
# The token mask bans 'e'-tokens but NOT non-Latin scripts, so a model can dodge
# the constraint by escaping into CJK (observed: "running鞋", "帮助", "进步").
# That passes a naive no-'e' check but is not English. We flag it deterministically.
_LATIN = re.compile(r"[A-Za-z]")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)  # any unicode letter

def english_ratio(text: str) -> float:
    letters = _ALPHA.findall(text or "")
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if _LATIN.match(c))
    return latin / len(letters)

def looks_english(text: str, threshold: float = 0.9) -> bool:
    return english_ratio(text) >= threshold

def pure_latin(text: str) -> bool:
    """Stricter: NO non-Latin letters at all. `looks_english` (ratio>=0.9) lets a
    few CJK/Cyrillic chars slip through; this flags any script-escape whatsoever."""
    return english_ratio(text) >= 0.999

# ---------------------------------------------------------------- parsing helpers
# A numbered item starts a line with a digit followed by . ) : or - , e.g.
#   "1. foo"   "2) bar"   "3 - baz"
_ITEM = re.compile(r"(?m)^\s*(\d+)\s*[\.\)\:\-]\s*(.*)$")

def parse_numbered_items(text: str) -> list[tuple[int, str]]:
    """Return [(number, item_text), ...] for digit-led list lines, de-duped by
    the leading number (keeps the first occurrence) so a stray repeat of '1.'
    does not inflate the count."""
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for m in _ITEM.finditer(text or ""):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        out.append((n, m.group(2).strip()))
    return out

_SENT_SPLIT = re.compile(r"[.!?]+")

def split_sentences(text: str) -> list[str]:
    """Count a segment as a sentence only if it has at least one letter, so that
    stray punctuation and list numerals do not count as sentences."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "")]
    return [p for p in parts if re.search(r"[A-Za-z]", p)]

def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))

# ---------------------------------------------------------------- length axis
def length(text: str) -> dict:
    return {
        "words": word_count(text),
        "chars": len(text or ""),
        "sentences": len(split_sentences(text)),
    }

# ---------------------------------------------------- requirement completion
def _check_items(text, n, mode, min_reason_words=0):
    items = parse_numbered_items(text)
    distinct = len(items)
    if min_reason_words > 0:
        with_reason = sum(1 for _, t in items if word_count(t) >= min_reason_words)
    else:
        with_reason = distinct
    if mode == "exactly":
        count_ok = distinct == n
    else:  # atleast
        count_ok = distinct >= n
    reason_ok = with_reason >= n
    return {
        "satisfied": bool(count_ok and reason_ok),
        "found_items": distinct,
        "items_with_reason": with_reason,
        "target": n,
        "mode": mode,
        "min_reason_words": min_reason_words,
    }

def _check_sentences(text, n, mode):
    c = len(split_sentences(text))
    ok = (c == n) if mode == "exactly" else (c >= n)
    return {"satisfied": bool(ok), "found_sentences": c, "target": n, "mode": mode}

def _check_max_words(text, k):
    w = word_count(text)
    return {"satisfied": bool(0 < w <= k), "found_words": w, "limit": k}

def requirement_completion(text: str, req: dict) -> dict:
    t = req.get("type")
    if t == "items_with_reason":
        return _check_items(text, req["n"], req.get("mode", "exactly"), req.get("min_reason_words", 4))
    if t == "items":
        return _check_items(text, req["n"], req.get("mode", "exactly"), 0)
    if t == "sentences":
        return _check_sentences(text, req["n"], req.get("mode", "exactly"))
    if t == "max_words":
        return _check_max_words(text, req["k"])
    return {"satisfied": False, "error": f"unknown req type {t}"}

# ---------------------------------------------------------------- bundle
def score_answer(text: str, req: dict) -> dict:
    """All deterministic axes for one answer."""
    lc = letter_compliance(text)
    eng = looks_english(text)
    pure = pure_latin(text)
    rc = requirement_completion(text, req)
    ln = length(text)
    return {
        "letter_compliant": lc,
        "english_ratio": round(english_ratio(text), 3),
        "looks_english": eng,
        "pure_latin": pure,
        "requirement": rc,
        "length": ln,
        # 'req_and_letter' is the lenient gate (no-e AND requirement met).
        # 'usable_strict' additionally demands the answer is actually English
        # (not a CJK-escape); coherence is added later from the judge/human axis.
        "req_and_letter": bool(lc and rc.get("satisfied")),
        "usable_strict": bool(lc and eng and rc.get("satisfied")),
    }


if __name__ == "__main__":
    # quick self-test
    good = "1. Run shoes, so your foot stays comfy.\n2. A sun hat, as it blocks rays.\n3. A drink jug, to stay hydrated... oops."  # has 'e'
    print("has e ->", not letter_compliance(good))
    clean = "1. Shos with good grip, so you do not slip.\n2. A warm top, to block cold air.\n3. A big jug, to stay full of liquid."
    print(score_answer(clean, {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}))
    short = "1. Shos.\n2. Top.\n3. Jug."
    print(score_answer(short, {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}))
