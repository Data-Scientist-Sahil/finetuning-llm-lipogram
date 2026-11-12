---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: apache-2.0
---

# Run B — no-'e' lipogram LoRA adapter (config only)

LoRA adapter (r=16, alpha=16, all attention and MLP projections) on
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), continued from Kevin Wu's
SFT-only adapter [`k33wu/qwen3-1.7b-no-e-sft180`](https://huggingface.co/k33wu/qwen3-1.7b-no-e-sft180)
(Apache-2.0) with 100 steps of GRPO-lite and a verifiable, judge-free reward
(`train/reward.py`), on a Kaggle T4 in fp16.

KL reference: base model with the adapter disabled (`--kl-ref base`), beta 0.02;
`max_words` prompts excluded from training. Requirement-met 95.1% on the 41-prompt eval
(SFT-180 start 97.6%).

**Weights are not included** in this repository (size); only `adapter_config.json`
and this card are committed. Training log: `../train_log.jsonl`.

Results, method and limitations: see the repository [README](../../../README.md) and
[FINDINGS](../../../FINDINGS.md). Intended for research on constrained generation only.
