"""Minimal, self-contained GRPO for the no-'e' task (RLVR, no judge in the loop).

Why hand-rolled instead of trl: our reward needs the no-'e' token mask applied
DURING rollout generation (otherwise nearly every sample fails the letter gate,
reward is ~0, and there is no learning signal). Injecting a custom LogitsProcessor
into trl's GRPO generation is version-fragile; a ~200-line loop that reuses our
own masked generation is simpler, dependency-light (just torch/transformers/peft),
and makes the algorithm legible.

Algorithm (GRPO-lite, on-policy):
  for each step:
    sample B prompts; for each, generate G completions under the no-'e' mask
    r = verifiable reward (train/reward.py) for each completion
    group-normalise: A_i = (r_i - mean_g) / (std_g + eps)      [GRPO advantage]
    policy loss = - mean_i( A_i * sum_t logp_policy(token_t) ) / tokens
    + beta * KL(policy || reference)   where reference = base w/ adapter disabled
    AdamW step on the LoRA params only; clip grad; checkpoint adapter

This is REINFORCE with GRPO's group baseline + a KL regulariser (single on-policy
update per batch, so no PPO ratio/clip). Enough to test the reward; swap in PPO
clipping later if needed.

Start point: Kevin's SFT-only adapter k33wu/qwen3-1.7b-no-e-sft180 (continue-train
its LoRA). Output: a new adapter dir, evaluated with the benchmark via
generate.py --adapter-path.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from masking import build_processor  # noqa: E402
from reward import reward_batch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_ADAPTER = "k33wu/qwen3-1.7b-no-e-sft180"


def load_prompts(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_policy(device, dtype, start_adapter=SFT_ADAPTER):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, start_adapter, is_trainable=True)
    model.to(device)
    return tok, model


def to_text(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def seqlogp_from_logits(logits, ids, attn, plens):
    """Per-sequence summed completion log-prob from logits, memory-efficiently.

    Avoids materialising a full [B, L, vocab] fp32 log-softmax (the T4 OOM): it
    gathers the target-token logit and subtracts a logsumexp over the vocab. Call
    on a MICRO-BATCH so the transient fp32 is small. Returns (sum_logp[B], ntok[B]).
    """
    logits = logits[:, :-1, :]
    targets = ids[:, 1:]
    tgt = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()   # [B, L-1]
    lse = torch.logsumexp(logits.float(), dim=-1)                        # [B, L-1]
    tok_logp = tgt - lse
    Lm1 = ids.shape[1] - 1
    idx = torch.arange(Lm1, device=ids.device).unsqueeze(0)
    plen_t = torch.tensor(plens, device=ids.device).unsqueeze(1)
    comp_mask = (idx >= (plen_t - 1)) & (attn[:, 1:] > 0)
    sum_logp = (tok_logp * comp_mask).sum(-1)
    ntok = comp_mask.sum(-1).clamp(min=1)
    return sum_logp, ntok


def train(args):
    device = args.device
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float32 if device == "cpu" else torch.bfloat16)
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok, model = build_policy(device, dtype, args.start_adapter)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mask_proc = LogitsProcessorList([build_processor(tok, device=device)])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    prompts = load_prompts(args.prompts)
    excl = set(args.exclude_req_types or [])
    if excl:
        before = len(prompts)
        prompts = [p for p in prompts if p["req"]["type"] not in excl]
        print(f"excluded req types {sorted(excl)}: {before} -> {len(prompts)} train prompts")

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = random.sample(prompts, min(args.batch_prompts, len(prompts)))
        all_ids, all_attn, plens, reqs, texts = [], [], [], [], []
        # ---- rollout (masked generation), no grad ----
        model.eval()
        with torch.no_grad():
            for p in batch:
                enc = tok(to_text(tok, p["prompt"]), return_tensors="pt").to(device)
                plen = enc["input_ids"].shape[1]
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature, top_p=0.95,
                    num_return_sequences=args.group, max_new_tokens=args.max_new_tokens,
                    logits_processor=mask_proc, pad_token_id=tok.pad_token_id,
                )
                for g in range(args.group):
                    ids = gen[g]
                    all_ids.append(ids)
                    plens.append(plen)
                    reqs.append(p["req"])
                    texts.append(tok.decode(ids[plen:], skip_special_tokens=True).strip())
        rewards = reward_batch(texts, reqs)

        # group-normalised advantages
        adv = []
        G = args.group
        for i in range(0, len(rewards), G):
            grp = rewards[i:i + G]
            m = sum(grp) / len(grp)
            var = sum((x - m) ** 2 for x in grp) / len(grp)
            sd = var ** 0.5
            adv.extend([(x - m) / (sd + 1e-4) for x in grp])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        # pad sequences to a batch
        maxlen = max(x.shape[0] for x in all_ids)
        pad_id = tok.pad_token_id
        ids_b = torch.full((len(all_ids), maxlen), pad_id, dtype=torch.long, device=device)
        attn_b = torch.zeros((len(all_ids), maxlen), dtype=torch.long, device=device)
        for j, x in enumerate(all_ids):
            ids_b[j, :x.shape[0]] = x
            attn_b[j, :x.shape[0]] = 1
        # ---- policy update, MICRO-BATCHED so vocab logits never blow up GPU mem ----
        # The GRPO-lite loss is separable per sequence (advantages are precomputed),
        # so we forward/​backward a few sequences at a time and accumulate gradients.
        model.train()
        opt.zero_grad()
        N = len(all_ids)
        MB = max(1, args.micro_batch)
        loss_val, kl_val = 0.0, 0.0
        for s in range(0, N, MB):
            sl = slice(s, s + MB)
            ids_mb, attn_mb, adv_mb = ids_b[sl], attn_b[sl], adv_t[sl]
            plens_mb = plens[s:s + MB]
            # reference (base, adapter disabled) first, then free it before the policy pass
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logits = model(input_ids=ids_mb, attention_mask=attn_mb).logits
                ref_sum, _ = seqlogp_from_logits(ref_logits, ids_mb, attn_mb, plens_mb)
                del ref_logits
            pol_logits = model(input_ids=ids_mb, attention_mask=attn_mb).logits
            sum_logp, ntok = seqlogp_from_logits(pol_logits, ids_mb, attn_mb, plens_mb)
            del pol_logits
            kl = (sum_logp - ref_sum) / ntok
            pg = -(adv_mb * (sum_logp / ntok))
            loss = (pg + args.beta * kl).sum() / N          # 1/N so the whole step is a mean
            loss.backward()
            loss_val += loss.item()
            kl_val += kl.sum().item() / N
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        rec = {"step": step, "mean_reward": round(sum(rewards) / len(rewards), 3),
               "max_reward": round(max(rewards), 3), "loss": round(loss_val, 4),
               "kl": round(kl_val, 4), "secs": round(time.time() - t0)}
        print(rec)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(os.path.join(args.out, "adapter"))
    log.close()
    print("saved adapter ->", os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(ROOT, "prompts", "train_prompts.jsonl"))
    ap.add_argument("--start-adapter", default=SFT_ADAPTER)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "idea2_runB"))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-req-types", nargs="*", default=["max_words"],
                    help="req types to drop from TRAINING (default: max_words, which teaches brevity)")
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="sequences per forward/backward chunk (lower = less GPU mem). T4-safe default 2.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="float32|bfloat16|float16 (default: fp32 cpu / bf16 gpu)")
    train(ap.parse_args())
