"""Tiny side-by-side demo: one prompt, three configs, scored live.

Usage:
    python demo.py                 # uses a built-in prompt
    python demo.py --prompt-id p01 # a prompt from the eval set
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import score_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--configs", nargs="+", default=["base_prompt", "base_mask", "adapter_mask"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()

    if a.prompt_id:
        with open(os.path.join(ROOT, "prompts", "eval_prompts.jsonl")) as f:
            row = next(json.loads(l) for l in f if json.loads(l)["id"] == a.prompt_id)
        prompt, req = row["prompt"], row["req"]
    else:
        prompt = ("Suggest 3 things I should buy first to start running. For each one, add a short "
                  "reason. Number each point with a digit and keep the whole answer free of the letter e.")
        req = {"type": "items_with_reason", "n": 3, "mode": "exactly", "min_reason_words": 4}

    from generate import build_model, make_inputs, CONFIGS
    from transformers import LogitsProcessorList
    from masking import build_processor
    import torch

    print("PROMPT:", prompt, "\nREQUIREMENT:", req, "\n" + "=" * 70)
    for cfg in a.configs:
        tok, model = build_model(cfg, "cpu", torch.bfloat16)
        proc = LogitsProcessorList([build_processor(tok)]) if CONFIGS[cfg]["mask"] else None
        inputs = make_inputs(tok, prompt, "cpu")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 logits_processor=proc, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        s = score_answer(ans, req)
        print(f"\n### {cfg}\n{ans}\n--> letter_ok={s['letter_compliant']} "
              f"req_ok={s['requirement']['satisfied']} words={s['length']['words']}")
        del model


if __name__ == "__main__":
    main()
