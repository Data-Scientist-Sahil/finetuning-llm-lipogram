"""Generate answers under each experimental config. CPU-friendly and resumable.

Configs
-------
  base_prompt   base Qwen3-1.7B, no mask         (prompting-alone baseline)
  base_mask     base Qwen3-1.7B + no-'e' mask    (mask-only)
  adapter_mask  base + k33wu/qwen3-1.7b-no-e (SFT+RL adapter) + mask
  sft_mask      base + k33wu/qwen3-1.7b-no-e-sft180 (SFT-only adapter) + mask

All runs are greedy (do_sample=False) so results are deterministic and
reproducible. Each completed answer is appended to results/raw/<config>.jsonl
immediately, so a run can be killed and resumed and already-done prompts are
skipped. Pass --max-seconds to stop cleanly before a wall-clock limit.
"""
from __future__ import annotations
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "results", "raw")

BASE_MODEL = "Qwen/Qwen3-1.7B"
CONFIGS = {
    "base_prompt":  {"adapter": None,                              "mask": False},
    "base_mask":    {"adapter": None,                              "mask": True},
    "adapter_mask": {"adapter": "k33wu/qwen3-1.7b-no-e",           "mask": True},
    "sft_mask":     {"adapter": "k33wu/qwen3-1.7b-no-e-sft180",    "mask": True},
    # our Idea-2 trained adapter; pass its local path with --adapter-path
    "trained_mask": {"adapter": None,                              "mask": True},
}


def load_prompts(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def done_ids(out_path: str) -> set[str]:
    ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def build_model(config_name: str, device: str, dtype, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = CONFIGS[config_name]
    adapter = adapter_path or cfg["adapter"]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return tok, model


def make_inputs(tok, prompt: str, device: str):
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(device)


def run(config_name: str, prompts_path: str, limit=None, max_new_tokens=200,
        device="cpu", max_seconds=None, dtype_str="bfloat16", adapter_path=None):
    import torch
    from transformers import LogitsProcessorList
    from masking import build_processor

    dtype = getattr(torch, dtype_str)
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"{config_name}.jsonl")
    prompts = load_prompts(prompts_path, limit)
    already = done_ids(out_path)
    todo = [p for p in prompts if p["id"] not in already]
    print(f"[{config_name}] {len(already)} done, {len(todo)} to go")
    if not todo:
        return

    tok, model = build_model(config_name, device, dtype, adapter_path)
    processors = None
    if CONFIGS[config_name]["mask"]:
        processors = LogitsProcessorList([build_processor(tok, device=device)])

    t0 = time.time()
    with open(out_path, "a") as f:
        for i, p in enumerate(todo):
            inputs = make_inputs(tok, p["prompt"], device)
            g0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    logits_processor=processors,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
            rec = {
                "id": p["id"], "config": config_name, "category": p["category"],
                "prompt": p["prompt"], "req": p["req"], "answer": answer,
                "gen_tokens": int(gen_ids.shape[0]),
                "gen_seconds": round(time.time() - g0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            f.flush()
            print(f"  [{config_name}] {p['id']} {rec['gen_tokens']}tok "
                  f"{rec['gen_seconds']}s  tot={round(time.time()-t0)}s")
            if max_seconds and (time.time() - t0) > max_seconds:
                print(f"[{config_name}] hit time budget, stopping (resumable)")
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--prompts", default=os.path.join(ROOT, "prompts", "eval_prompts.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seconds", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--adapter-path", default=None, help="local adapter dir (for --config trained_mask)")
    a = ap.parse_args()
    run(a.config, a.prompts, a.limit, a.max_new_tokens, a.device, a.max_seconds, a.dtype, a.adapter_path)
