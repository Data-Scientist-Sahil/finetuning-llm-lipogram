"""Emit idea2_kaggle.ipynb — an experimental Kaggle training notebook (GPU execution unverified).

Run `python notebooks/build_notebook.py` to (re)generate the .ipynb.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src):return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

cells = [
md("""# Experimental verifiable-reward training for no-'e' generation

**Status: NOT RUN.** No new trained-model results are included. The committed
benchmark evaluates Kevin Wu's released adapter, not a model trained here.

This workflow is intended to continue training Qwen3-1.7B from Kevin Wu's
SFT-only adapter with a verifiable structural reward, then evaluate it.

**GPU caveat:** execution and memory requirements are unverified. Code currently
requests bfloat16; use hardware with native BF16 support. T4 has no native BF16
support, so this is not a verified T4-ready workflow. Changing dtype requires
numerical/memory validation before training.

Adapter weights are saved every 20 steps. This is not full training resume:
each invocation starts a new loop from the configured adapter; optimizer state
and global step are not restored. No runtime estimate is validated."""),

md("## 0. GPU check"),
code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout or "no nvidia-smi")
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Enable the GPU accelerator in Kaggle settings."
"""),

md("""## 1. Get the repo

Two options — use whichever you set up:
- **Git:** push `no-e-eval` to a GitHub repo and set `REPO_URL` below.
- **Kaggle dataset:** upload the `no-e-eval` folder as a dataset; it mounts under
  `/kaggle/input/...`. Set `INPUT_DIR` to that path.

The cell copies the repo to `/kaggle/working/no-e-eval` (writable)."""),
code("""
import os, shutil, subprocess
REPO_URL = ""                       # e.g. "https://github.com/<you>/no-e-eval.git"
INPUT_DIR = ""                      # e.g. "/kaggle/input/no-e-eval/no-e-eval"
DST = "/kaggle/working/no-e-eval"
if REPO_URL:
    if not os.path.exists(DST): subprocess.run(["git","clone",REPO_URL,DST], check=True)
elif INPUT_DIR:
    if os.path.exists(DST): shutil.rmtree(DST)
    shutil.copytree(INPUT_DIR, DST)
else:
    raise SystemExit("Set REPO_URL or INPUT_DIR to point at the repo.")
os.chdir(DST); print("cwd:", os.getcwd()); print(sorted(os.listdir(".")))
"""),

md("## 2. Deps (Kaggle already has torch; refresh transformers/peft). No trl needed."),
code("""
import sys, subprocess
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.51","peft>=0.11","accelerate>=0.30"], check=True)
import transformers, peft; print("transformers", transformers.__version__, "peft", peft.__version__)
"""),

md("## 3. Smoke test — 2 tiny steps, confirm the full loop runs on the GPU"),
code("""
import subprocess, sys
r = subprocess.run([sys.executable,"train/grpo_train.py","--device","cuda","--dtype","bfloat16",
                    "--steps","2","--batch-prompts","2","--group","4","--max-new-tokens","24",
                    "--save-every","2","--out","/kaggle/working/smoke"], capture_output=True, text=True)
print(r.stdout[-2000:]); print("STDERR tail:", r.stderr[-800:])
assert "saved adapter" in r.stdout, "smoke test failed — check STDERR above"
"""),

md("""## 4. Train Run B (verifiable reward)

Defaults mirror Kevin's scale loosely (LoRA already in the SFT adapter; group of 8,
4 prompts/step). Tune `STEPS` down first if you want a faster first look. The run
logs mean/max reward per step to `results/idea2_runB/train_log.jsonl`."""),
code("""
import subprocess, sys
STEPS = 120
r = subprocess.run([sys.executable,"train/grpo_train.py","--device","cuda","--dtype","bfloat16",
                    "--steps",str(STEPS),"--batch-prompts","4","--group","8","--max-new-tokens","128",
                    "--lr","1e-5","--beta","0.02","--temperature","0.7","--save-every","20",
                    "--out","/kaggle/working/no-e-eval/results/idea2_runB"], text=True)
print("return code", r.returncode)
"""),

md("### reward curve"),
code("""
import json, matplotlib.pyplot as plt
log = [json.loads(l) for l in open("results/idea2_runB/train_log.jsonl")]
plt.figure(figsize=(7,3))
plt.plot([x["step"] for x in log], [x["mean_reward"] for x in log], label="mean reward")
plt.plot([x["step"] for x in log], [x["max_reward"] for x in log], label="max reward", alpha=.5)
plt.xlabel("step"); plt.ylabel("reward"); plt.legend(); plt.title("Run B — verifiable reward"); plt.show()
"""),

md("""## 5. Evaluate on the benchmark & compare

Generate answers on the 41 held-out eval prompts for the **trained adapter** and
(for the baseline) **Kevin's released adapter**, score every axis, bootstrap, and
print the comparison. GPU makes this quick."""),
code("""
import subprocess, sys, os
env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
def gen(cfg, extra=[]):
    subprocess.run([sys.executable,"src/generate.py","--config",cfg,"--device","cuda",
                    "--dtype","bfloat16","--max-new-tokens","160"]+extra, check=True, env=env)
gen("trained_mask", ["--adapter-path","results/idea2_runB/adapter"])
gen("adapter_mask")          # Kevin's released SFT+RL adapter (baseline)
gen("base_mask")             # mask-only, for context
"""),
code("""
import subprocess, sys
subprocess.run([sys.executable,"src/run_eval.py","--configs","base_mask","adapter_mask","trained_mask"], check=True)
subprocess.run([sys.executable,"src/bootstrap.py","--configs","base_mask","adapter_mask","trained_mask"], check=True)
subprocess.run([sys.executable,"src/report.py"], check=True)
print(open("results/REPORT.md").read())
"""),

md("""## 6. Package results for download

Grab the adapter + results from the Kaggle output panel (or the zip below) and
download them to your chosen local output directory."""),
code("""
import shutil
shutil.make_archive("/kaggle/working/idea2_runB_out","zip","/kaggle/working/no-e-eval/results")
print("zip at /kaggle/working/idea2_runB_out.zip")
"""),

md("""## Interpreting future results

Compare held-out requirement completion and script compliance, with paired
bootstrap intervals over prompts. Report inconclusive intervals honestly.
Obtain independent human review for qualitative coherence: existing proxy
scores were assigned by Claude and are not independently judge-validated.
Do not treat the existing released-adapter results as evidence this training
workflow has succeeded."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "idea2_kaggle.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(cells))
