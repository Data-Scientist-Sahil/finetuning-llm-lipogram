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

Status: experimental training kit, not yet run or GPU-validated. Adapter saves
do not implement full training resume: no optimizer/global-step restoration.

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


def seq_logprobs(model, input_ids, attn, prompt_len):
    """Sum log p(token) over the completion tokens of each sequence. Returns
    (sum_logp [B], n_tokens [B]). Uses current grad mode of the caller."""
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    # mask: only completion tokens (positions >= prompt_len-1 in the shifted view) and real tokens
    L = input_ids.shape[1]
    idx = torch.arange(L - 1, device=input_ids.device).unsqueeze(0)
    comp_mask = (idx >= (prompt_len - 1)) & (attn[:, 1:] > 0)
    summed = (tok_logp * comp_mask).sum(-1)
    ntok = comp_mask.sum(-1).clamp(min=1)
    return summed, ntok


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
        # per-sample prompt_len differ; compute logprobs per sample-group using min plen trick:
        # we stored plens per sample — handle via a loop-safe constant since all in a group share plen
        # (left-padding would shift; we right-pad and pass plen per sample through comp_mask)
        # Build a per-row prompt_len mask instead of a scalar:
        model.train()
        out = model(input_ids=ids_b, attention_mask=attn_b)
        logits = out.logits[:, :-1, :]
        targets = ids_b[:, 1:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        Lm1 = ids_b.shape[1] - 1
        idx = torch.arange(Lm1, device=device).unsqueeze(0)
        plen_t = torch.tensor(plens, device=device).unsqueeze(1)
        comp_mask = (idx >= (plen_t - 1)) & (attn_b[:, 1:] > 0)
        sum_logp = (tok_logp * comp_mask).sum(-1)
        ntok = comp_mask.sum(-1).clamp(min=1)

        # KL to reference (base, adapter disabled), no grad
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(input_ids=ids_b, attention_mask=attn_b)
            ref_logp = torch.log_softmax(ref_out.logits[:, :-1, :].float(), -1)
            ref_tok = ref_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            ref_sum = (ref_tok * comp_mask).sum(-1)
        kl = ((sum_logp - ref_sum) / ntok).mean()

        pg = -(adv_t * (sum_logp / ntok)).mean()
        loss = pg + args.beta * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        rec = {"step": step, "mean_reward": round(sum(rewards) / len(rewards), 3),
               "max_reward": round(max(rewards), 3), "loss": round(loss.item(), 4),
               "kl": round(kl.item(), 4), "secs": round(time.time() - t0)}
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="float32|bfloat16|float16 (default: fp32 cpu / bf16 gpu)")
    train(ap.parse_args())
