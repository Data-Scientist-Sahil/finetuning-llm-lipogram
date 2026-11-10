"""Emit idea2_kaggle.ipynb — a push-button Kaggle (T4) notebook for Idea 2 Step 1.

Run `python notebooks/build_notebook.py` to (re)generate the .ipynb.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src):return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

cells = [
md("""# Idea 2 — Step 1: verifiable-reward GRPO (RLVR) for the no-'e' task

**Goal:** train Qwen3-1.7B (from Kevin Wu's SFT-only adapter, `sft180`) with a
*fully verifiable* reward — no LLM judge in the loop — that targets
requirement-completion, then score it on our benchmark.

**This is a 40-step *validation* run, not an attempt to beat Kevin's number.**
Kevin's 60.5% usable rate came from ~300 RL steps; 40 steps of this loop will not
match that, and that is not the claim. The claim this run tests is narrow and fair:
does our reward improve the model **over its own SFT-180 starting point**?

**The honest comparison is `trained` vs `SFT-180` (same start, only our RL differs).**
Kevin's released SFT+RL adapter is included only as a *reference* line — it has far
more RL, so it is not a matched baseline.

**Method label:** this is **GRPO-lite** — REINFORCE with GRPO's group-normalised
baseline + a KL term (no PPO clipping, KL reference = base w/ adapter disabled). It
is legible and enough to test the reward; it is *not* a faithful GRPO reimplementation.

**Escalation ladder (stop at the first win):**
1. **this notebook** — 40 steps; validate the pipeline + a direction signal vs SFT-180.
2. if promising: scale steps (120+); if inconclusive: matched judge-reward Run A.
3. if still unclear: reward engineering (perplexity/fluency term, weight tuning).

**Hardware / run settings (learned the hard way on run 1):**
- *Settings → Accelerator → GPU T4*, and **Internet ON** (model downloads from HF).
- **Use fp16, not bf16.** The T4 is Turing and has no bf16 hardware; `DTYPE="float16"`
  is set in the GPU cell and threaded through training + eval.
- **Persist your artifacts:** run via **Save Version → Save & Run All (Commit)**, not just
  interactive cells — an interactive `/kaggle/working` can be wiped on a reset and the
  trained adapter is lost. The batch run keeps `/kaggle/working/` as the version Output.
- **Memory:** the trainer forwards sequences in **micro-batches** (`--micro-batch 2`) and
  computes log-probs without a full-vocab fp32 upcast, so the reference/policy passes fit a
  T4. If you still OOM, lower `--micro-batch` to 1 (then `--group` 6, then `--max-new-tokens` 96).
- The trainer checkpoints every 20 steps but does **not auto-resume** (a restart begins
  again from SFT-180)."""),

md("## 0. GPU check"),
code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout or "no nvidia-smi")
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Enable the GPU accelerator in Kaggle settings."
# T4 (Turing) has NO bf16 hardware -> force fp16. Only switch to bfloat16 on Ampere+ (A100/L4).
DTYPE = "float16"
print("dtype:", DTYPE)
"""),

md("""## 1. Get the repo

Two options — use whichever you set up:
- **Git:** push `no-e-eval` to a GitHub repo and set `REPO_URL` below.
- **Kaggle dataset:** upload the `no-e-eval` folder as a dataset; it mounts under
  `/kaggle/input/...`. Set `INPUT_DIR` to that path.

The cell copies the repo to `/kaggle/working/no-e-eval` (writable)."""),
code("""
import os, shutil, subprocess, glob
REPO_URL  = ""   # optional: a git URL. Leave blank when using a Kaggle Dataset.
INPUT_DIR = ""   # optional: path to the no-e-eval folder; blank = auto-detect under /kaggle/input
DST = "/kaggle/working/no-e-eval"

def _autodetect():
    hits = glob.glob("/kaggle/input/**/src/generate.py", recursive=True)
    return os.path.dirname(os.path.dirname(hits[0])) if hits else ""

if not REPO_URL and not INPUT_DIR:
    INPUT_DIR = _autodetect()
    print("auto-detected INPUT_DIR:", INPUT_DIR or "(none - did you add the dataset?)")

if REPO_URL:
    if not os.path.exists(DST): subprocess.run(["git","clone",REPO_URL,DST], check=True)
elif INPUT_DIR:
    if os.path.exists(DST): shutil.rmtree(DST)
    shutil.copytree(INPUT_DIR, DST)
else:
    raise SystemExit("Add the no-e-eval dataset (Add Input, right panel), or set REPO_URL/INPUT_DIR.")
os.chdir(DST); print("cwd:", os.getcwd()); print(sorted(os.listdir(".")))
"""),

md("## 2. Deps (Kaggle already has torch; refresh transformers/peft). No trl needed."),
code("""
import sys, subprocess
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers==4.51.3","peft==0.15.2","accelerate==1.6.0"], check=True)
import transformers, peft; print("transformers", transformers.__version__, "peft", peft.__version__)
"""),

md("## 3. Smoke test — 2 tiny steps, confirm the full loop runs on the GPU"),
code("""
import subprocess, sys, os
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
r = subprocess.run([sys.executable,"train/grpo_train.py","--device","cuda","--dtype",DTYPE,
                    "--steps","2","--batch-prompts","2","--group","4","--micro-batch","2",
                    "--max-new-tokens","24","--save-every","2","--out","/kaggle/working/smoke"],
                   capture_output=True, text=True, env=env)
print(r.stdout[-2000:]); print("STDERR tail:", r.stderr[-800:])
assert "saved adapter" in r.stdout, "smoke test failed — check STDERR above"
"""),

md("""## 4. Train Run B (verifiable reward v2)

Run 1 (40 steps) regressed on completeness and shrank outputs — it got *better* at the
`max_words` prompts and *worse* at the completeness prompts, i.e. the reward optimized
**brevity**. Two targeted fixes for this run:
- **`max_words` dropped from training** (`grpo_train.py` defaults `--exclude-req-types max_words`);
  still scored at eval. It was teaching "shorter is safer".
- **Length-adequacy floor in the reward** (`train/reward.py`): a too-short answer can no
  longer claim completeness credit.

100 steps, group 8. Reward unit tests (`python train/reward.py`) cover the new floor.
If a T4 OOMs, drop `--group` 8→6 or `--max-new-tokens` 128→96."""),
code("""
import subprocess, sys, os
STEPS = 100
OUT = "/kaggle/working/no-e-eval/results/idea2_runB"
# micro-batch=2 keeps vocab logits small (fits T4 16 GB); expandable_segments avoids fragmentation.
# If you still OOM: --micro-batch 1, then --group 6, then --max-new-tokens 96.
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
r = subprocess.run([sys.executable,"train/grpo_train.py","--device","cuda","--dtype",DTYPE,
                    "--steps",str(STEPS),"--batch-prompts","4","--group","8","--micro-batch","2",
                    "--max-new-tokens","128","--lr","1e-5","--beta","0.02","--temperature","0.7",
                    "--save-every","20","--out",OUT], text=True, env=env)  # drops max_words by default
print("return code", r.returncode)
assert os.path.isdir(os.path.join(OUT,"adapter")), "adapter not saved — do not proceed; re-run as Save & Run All"
print("adapter saved:", os.listdir(os.path.join(OUT,"adapter")))
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

Generate answers on the 41 held-out eval prompts for four configs, score every axis,
bootstrap, and print the report. The **primary** comparison is `trained_mask` vs
`sft_mask` (same SFT-180 start — isolates our reward). `adapter_mask` (Kevin's
released SFT+RL) and `base_mask` are reference lines, not the yardstick."""),
code("""
import subprocess, sys, os
env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
def gen(cfg, extra=[]):
    subprocess.run([sys.executable,"src/generate.py","--config",cfg,"--device","cuda",
                    "--dtype",DTYPE,"--max-new-tokens","160"]+extra, check=True, env=env)
gen("trained_mask", ["--adapter-path","results/idea2_runB/adapter"])  # OUR run
gen("sft_mask")              # SFT-180 start = the controlled baseline
gen("adapter_mask")          # Kevin's released SFT+RL adapter (reference)
gen("base_mask")             # mask-only (context)
"""),
code("""
import subprocess, sys
CFGS = ["base_mask","sft_mask","adapter_mask","trained_mask"]
subprocess.run([sys.executable,"src/run_eval.py","--configs",*CFGS], check=True)
subprocess.run([sys.executable,"src/bootstrap.py","--configs",*CFGS], check=True)
subprocess.run([sys.executable,"src/report.py"], check=True)
print(open("results/REPORT.md").read())
"""),

md("""## 6. Package results for download — and PERSIST them

**Important:** in an interactive session, `/kaggle/working` can be wiped on a reset,
which loses the trained adapter. To keep the artifacts, run the whole notebook via
**Save Version → Save & Run All (Commit)** — that executes it as a batch job and
persists `/kaggle/working/` as the version's Output, where you download the zip.

The cell is self-contained and asserts the adapter exists before zipping, so a lost
artifact fails loudly instead of producing an empty zip."""),
code("""
import os, shutil
RES = "/kaggle/working/no-e-eval/results"
ADA = os.path.join(RES, "idea2_runB", "adapter")
assert os.path.isdir(ADA), f"no adapter at {ADA}: training did not persist. Re-run via Save & Run All."
print("adapter files:", os.listdir(ADA))
# copy adapter + results to the notebook working root so they land in the version Output
shutil.copytree(os.path.join(RES, "idea2_runB"), "/kaggle/working/idea2_runB", dirs_exist_ok=True)
zip_path = shutil.make_archive("/kaggle/working/idea2_runB_out", "zip", RES)
print("zip:", zip_path, os.path.getsize(zip_path), "bytes")
"""),

md("""## What counts as a Step-1 result

Read the **`trained − SFT-180`** paired-bootstrap table in the report — that is the
controlled test of our reward (ignore `trained − Kevin's adapter`; his has far more RL).

- **Win:** `requirement` and/or `items_with_reason` / `sentences` rise with a 95% CI
  that excludes 0, while `pure_english` holds. → scale steps (120+).
- **Inconclusive at 40 steps** (CI straddles 0) but the reward curve climbed: pipeline
  works, signal is directional → scale steps, not a failure.
- **Flat reward curve / no movement:** debug the loop/reward before spending more GPU.

Either way it is a legitimate, honest result. Escalation ladder: `planning/05-idea2-plan.md`."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "idea2_kaggle.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(cells))
