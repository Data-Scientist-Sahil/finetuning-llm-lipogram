"""Token mask that forbids the letter 'e' at decode time.

This mirrors the mechanism in Kevin Wu's lipogram experiment: before a token is
chosen, every token whose text contains 'e' or 'E' is set to -inf, so it can
never be sampled. The mask only *prevents* forbidden tokens; it does nothing to
keep the answer complete or on-topic. Catching that gap is the job of the eval,
not the mask.

Two correctness details that are easy to get wrong:

1. Special tokens are NEVER banned, even though several contain 'e'
   (e.g. '<|endoftext|>', '<|im_end|>'). If we banned the EOS token the masked
   model could never stop, so it would ramble to max_new_tokens every time and
   the "shrinking" comparison would be meaningless. We whitelist all special ids.

2. 'e' detection is done on the *decoded* text of each token (byte-level BPE is
   reversed via convert_tokens_to_string), so leading-space markers like 'Ġ' do
   not cause false matches and multi-byte tokens are handled correctly.

The banned-id list is cached to disk per tokenizer so the ~150k-token scan runs
only once.
"""
from __future__ import annotations
import hashlib
import json
import os
import torch
from transformers import LogitsProcessor

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", ".mask_cache")


def _cache_path(tokenizer) -> str:
    key = getattr(tokenizer, "name_or_path", "tok") + f"|v{tokenizer.vocab_size}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"banned_{h}.json")


def banned_token_ids(tokenizer, letter: str = "e") -> list[int]:
    """Return the ids of every token whose decoded text contains the letter
    (case-insensitive), excluding all special tokens."""
    path = _cache_path(tokenizer)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    letter = letter.lower()
    special = set(tokenizer.all_special_ids or [])
    vocab_size = len(tokenizer)
    ids = list(range(vocab_size))
    toks = tokenizer.convert_ids_to_tokens(ids)
    banned: list[int] = []
    for i, tok in zip(ids, toks):
        if i in special or tok is None:
            continue
        try:
            text = tokenizer.convert_tokens_to_string([tok])
        except Exception:
            text = tok
        if letter in text.lower():
            banned.append(i)
    with open(path, "w") as f:
        json.dump(banned, f)
    return banned


class ForbidLetterProcessor(LogitsProcessor):
    """Sets the logit of every banned token to -inf before sampling."""

    def __init__(self, banned_ids: list[int], device: str = "cpu"):
        self.banned = torch.tensor(sorted(set(banned_ids)), dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.banned.numel():
            scores = scores.clone()
            scores[:, self.banned] = float("-inf")
        return scores


def build_processor(tokenizer, device: str = "cpu", letter: str = "e") -> ForbidLetterProcessor:
    return ForbidLetterProcessor(banned_token_ids(tokenizer, letter), device=device)
