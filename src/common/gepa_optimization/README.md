# GEPA Prompt Optimization — Student Handoff Guide

This guide explains everything you need to run the GEPA prompt optimization pipeline from scratch. It assumes you know Python and basic ML, but have never used **vLLM** or **GEPA/DSPy** before.

---

## Table of Contents

1. [What Does This Code Do?](#1-what-does-this-code-do)
2. [Two Tools You Need to Understand](#2-two-tools-you-need-to-understand)
   - [vLLM — running models locally](#vllm--running-models-locally)
   - [GEPA — automatic prompt optimization](#gepa--automatic-prompt-optimization)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Directory Structure](#4-directory-structure)
5. [Step-by-Step Setup](#5-step-by-step-setup)
   - [Option A: Docker (recommended)](#option-a-docker-recommended)
   - [Option B: Bare metal](#option-b-bare-metal)
6. [Configuration Reference](#6-configuration-reference)
7. [Running the Optimization](#7-running-the-optimization)
8. [Understanding the Output](#8-understanding-the-output)
9. [Monitoring a Run](#9-monitoring-a-run)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick-Reference Cheat Sheet](#11-quick-reference-cheat-sheet)

---

## 1. What Does This Code Do?

The goal is to teach a small language model (e.g., Qwen3-4B) to answer multi-hop reasoning questions by generating **First-Order Logic (FOL) proof chains**.

A proof chain looks like this:

```
<logic>
BornIn(1924, LouisCha)
PenName(LouisCha, JinYong)
BornIn(1924, LouisCha) ∧ PenName(LouisCha, JinYong)
∃x (PenName(x, JinYong) ∧ BornIn(1924, x))
</logic>
Final Answer: 1924
```

The model's quality depends heavily on the **system prompt** (the instructions we give it). Writing a good prompt by hand is hard. GEPA automates this: it tries hundreds of prompt variants and keeps the best ones.

The optimization evaluates each prompt variant using two scoring signals:

| Signal | What it checks |
|---|---|
| **Standard metric** | Can we parse the `<logic>` block? Are the FOL formulas syntactically valid? (checked by Prover9 theorem prover) |
| **LLM-as-judge metric** | A second LLM rates the output's format quality and logical reasoning quality (1–5 scale) |

The two scores are combined with configurable weights (`standard_weight`, `judge_weight`) and GEPA maximizes the combined score.

---

## 2. Two Tools You Need to Understand

### vLLM — running models locally

**What it is:** vLLM is a Python library and server that runs open-source LLMs locally on your GPU. Instead of sending requests to the OpenAI API, you:

1. Download a model from HuggingFace (e.g., `Qwen/Qwen3-4B-AWQ`)
2. Start a vLLM server: `vllm serve Qwen/Qwen3-4B-AWQ --port 8000`
3. vLLM exposes an **OpenAI-compatible HTTP API** at `http://localhost:8000/v1`

You can then call it exactly like you would call OpenAI's API — same request format, same response format. DSPy (the library GEPA lives in) connects to it this way.

**Why we need two servers:** We run two models simultaneously on the same GPU:
- **FOL model** (port 8000) — the model we are *optimizing the prompt for*
- **Judge model** (port 8001) — a separate model that *scores* the FOL model's outputs

**How memory is split:** The config controls how much GPU memory each server gets (`fol_gpu_mem: 0.30`, `judge_gpu_mem: 0.60`). On a 24 GB GPU, 30% ≈ 7 GB for the FOL model, 60% ≈ 14 GB for the judge.

**Key vLLM concepts:**

| Concept | Explanation |
|---|---|
| `--max-model-len` | Maximum context window (input + output tokens). Larger = more memory. |
| `--gpu-memory-utilization` | Fraction of GPU VRAM to reserve (0.0–1.0). The two models must sum to < 1.0. |
| `--max-num-seqs` | How many requests to process in parallel. Keep low (4) when two models share the GPU. |
| `--dtype auto` | Let vLLM pick the best dtype (float16/bfloat16) automatically. |
| `--trust-remote-code` | Required for models with custom tokenizer code (Qwen3, etc.). |

**Health check:** Once started, check if a server is ready:
```bash
curl http://localhost:8000/health
# Returns: {"status":"healthy"}
```

---

### GEPA — automatic prompt optimization

**What it is:** GEPA (Genetic-Pareto Automatic) is an optimizer inside [DSPy](https://dspy.ai/). DSPy is a framework for programming language models — instead of writing prompts by hand you write Python code that describes *what* the model should do, and DSPy handles prompt formatting.

**How GEPA works (simplified):**

```
Start with a seed prompt (your initial instructions)
    ↓
Evaluate it on a validation set → get a score
    ↓
Generate MANY candidate mutations of the prompt
(GEPA asks a "reflection LM" to suggest improvements
 based on which examples succeeded and which failed)
    ↓
Evaluate each candidate → scores
    ↓
Keep the best candidates (Pareto selection)
    ↓
Repeat until budget exhausted
    ↓
Return the best prompt found
```

**Budget options (pick exactly one):**

| Option | Meaning | Typical wall-clock time |
|---|---|---|
| `gepa_auto: light` | A few dozen evaluations | ~30–60 min |
| `gepa_auto: medium` | ~100 evaluations | ~2–4 hours |
| `gepa_auto: heavy` | ~300 evaluations | ~8–12 hours |
| `max_full_evals: N` | Exactly N full evaluation rounds | Proportional to N |
| `max_metric_calls: N` | Stop after N total metric calls | More fine-grained control |

**DSPy terms you will see in logs:**

| Term | Meaning |
|---|---|
| `seed program` | Your initial prompt + DSPy module |
| `trainset` | Examples GEPA uses to *generate* prompt candidates |
| `valset` | Examples GEPA uses to *score* prompt candidates (held-out) |
| `ChainOfThought` | A DSPy module that instructs the LM to think step-by-step |
| `Prediction` | DSPy's output object (contains the score and feedback) |

---

## 3. Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB (requires small quantized models) | 24 GB |
| System RAM | 16 GB | 32 GB |
| Disk space | ~20 GB (model weights) | ~50 GB |
| CUDA | 11.8+ | 12.4 |

The default config uses:
- FOL model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` — a 4-bit quantized 4B model (~5 GB)
- Judge model: `Qwen/Qwen3-8B-AWQ` — a 4-bit quantized 8B model (~8 GB)

These fit together on a single 24 GB GPU (30% + 60% = 90% utilization).

---

## 4. Directory Structure

```
gepa_optimization/
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── run.sh                           ← one-command launch script
├── gepa_optimize_prompt.py          ← main Python script
│
├── config/
│   └── gepa_config.yaml             ← all settings (models, budget, weights, ...)
│
├── templates/                       ← prompt templates (YAML)
│   ├── default.yaml                 ← minimal example template
│   └── qwen3_4B/
│       ├── uni_logic_answer.yaml    ← unified logic+answer format
│       └── gepa_uni_logic_answer.yaml  ← alternative template used in production runs
│
├── core/                            ← Python package: dataset loading, metrics, templates
│   ├── dataset_handler.py           ← loads HotpotQA, 2WikiMultihop, FOLIO, ProntoQA
│   ├── template_handler.py          ← parses template YAML, formats prompts, parses model output
│   ├── metrics.py                   ← compute_single_completion_metrics (parse + FOL + ROUGE)
│   ├── le_evaluator.py              ← logical-equivalence metric (truth-table comparison)
│   └── utils.py                     ← ConfigArgumentParser (YAML + argparse merger), set_seed
│
├── solver/                          ← Python package: FOL syntax checking
│   ├── fol_parser.py                ← NLTK-based CFG parser for FOL strings
│   ├── fol_formula.py               ← validates a single FOL formula (with subprocess timeout)
│   ├── P9_formula.py                ← converts FOL to Prover9 input format
│   └── logic_evaluator.py           ← runs Prover9 to check consistency + entailment
│
└── output/                          ← created at runtime
    └── gepa/
        └── <output_dir>/            ← one folder per run (named by output_dir in config)
            ├── optimized_instructions_<timestamp>.txt
            ├── optimized_program_<timestamp>.json
            ├── optimization_results_<timestamp>.json
            ├── gepa_optimization_stats.csv
            └── logs/                ← GEPA internal logs
```

---

## 5. Step-by-Step Setup

### Option A: Docker (recommended)

Docker gives you an isolated environment with all system dependencies (CUDA, Prover9, Python 3.11) pre-installed. This is the safest way to reproduce results.

**Step 1 — Build the Docker image**

From the *repository root* (one level up from this directory):

```bash
cd /path/to/synfol      # repository root
docker build -f docker/Dockerfile -t synfol:latest .
```

This takes 10–20 minutes the first time (downloads PyTorch, vLLM, etc.).

**Step 2 — Copy or mount this directory inside the container**

There are two approaches:

*Approach 2a: mount at runtime (easiest for development)*
```bash
docker run --rm -it \
    --gpus all \
    -v $(pwd)/gepa_optimization:/workdir/gepa_optimization \
    -v ~/.cache/huggingface:/hf_home \
    synfol:latest bash
```

*Approach 2b: copy into image (for a clean handoff)*
The Dockerfile already copies the full repo; you can just use the image as-is.

**Step 3 — Inside the container, navigate to the optimization directory**

```bash
cd /workdir/gepa_optimization
```

**Step 4 — Download NLTK data (first time only)**

```bash
python3 -c "import nltk; nltk.download('punkt')"
```

**Step 5 — Run the optimization**

```bash
./run.sh config/gepa_config.yaml
```

That's it. The script starts both vLLM servers and runs the optimization.

---

### Option B: Bare metal

Use this if you can't use Docker.

**Step 1 — Install Python 3.11**

```bash
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-dev python3-pip
```

**Step 2 — Install vLLM**

vLLM must be installed before the other requirements (it pins torch):

```bash
pip install vllm==0.11.0
```

> Note: vLLM will install PyTorch automatically. Do NOT install torch separately before this step.

**Step 3 — Install Prover9**

Prover9 is a first-order theorem prover used to check FOL validity. You need to build it from source.

```bash
# Clone the bundled copy from the repository root
cd /path/to/synfol/Prover9
make all

# Set the environment variable so Python's NLTK can find it
export PROVER9=/path/to/synfol/Prover9/bin
# Add this line to your ~/.bashrc to make it permanent
echo 'export PROVER9=/path/to/synfol/Prover9/bin' >> ~/.bashrc
```

Verify it works:
```bash
echo "formulas(assumptions). P(a). end_of_list.
formulas(goals). P(a). end_of_list." | $PROVER9/prover9
# Should print: -------- PROOF --------
```

**Step 4 — Install Python requirements**

```bash
cd gepa_optimization
pip install -r requirements.txt
```

**Step 5 — Download NLTK data**

```bash
python3 -c "import nltk; nltk.download('punkt')"
```

**Step 6 — Run the optimization**

Start vLLM servers manually (in two separate terminals or use `&`), then run the script.

```bash
# Terminal 1: FOL server
vllm serve unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit \
    --port 8000 --gpu-memory-utilization 0.30 --max-model-len 4096

# Terminal 2: Judge server
vllm serve Qwen/Qwen3-8B-AWQ \
    --port 8001 --gpu-memory-utilization 0.60 --max-model-len 5120

# Terminal 3: Run optimization (once both servers are healthy)
cd gepa_optimization
python3 gepa_optimize_prompt.py --config config/gepa_config.yaml
```

Or just use `./run.sh` which does all three automatically.

---

## 6. Configuration Reference

All settings live in `config/gepa_config.yaml`. Here is a full annotated version:

```yaml
# ── MODELS ─────────────────────────────────────────────────────────────────
fol_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
# The model whose prompt you are optimizing.
# Must be a HuggingFace model ID. Will be downloaded automatically.
# Use a small quantized model to fit alongside the judge.

judge_model: Qwen/Qwen3-8B-AWQ
# The model that scores each output (LLM-as-judge).
# Should be capable enough to evaluate FOL quality.
# Qwen3-8B-AWQ is a good balance of speed and quality.

# ── TEMPLATE ───────────────────────────────────────────────────────────────
initial_template: qwen3_4B/uni_logic_answer
# Which prompt template to use as the starting point.
# Path is relative to the templates/ directory (omit .yaml extension).
# Available: default, qwen3_4B/uni_logic_answer, qwen3_4B/gepa_uni_logic_answer

# ── DATASETS ───────────────────────────────────────────────────────────────
dataset:
  - hotpotqa       # Multi-hop QA from Wikipedia
  - 2wiki           # 2WikiMultihop QA
# You can list one or multiple datasets; they will be merged.
# Supported: hotpotqa, 2wiki, folio, prontoqa, logic_translations_q14B

split: train        # Which split to sample from: train or test
downsample: 0.5     # Use only 50% of the dataset (faster iteration)

# ── SAMPLES FOR GEPA ───────────────────────────────────────────────────────
num_train_samples: 8000
# Number of examples GEPA uses to *generate* prompt candidates.
# Higher = more diverse training signal, slower per iteration.

num_val_samples: 2000
# Number of examples GEPA uses to *evaluate* each candidate prompt.
# This is the number of model calls per evaluation round.
# Reduce to 50–100 for quick tests.

# ── SERVER SETTINGS ────────────────────────────────────────────────────────
vllm_url: http://localhost
fol_port: 8000        # Port for the FOL model server
judge_port: 8001      # Port for the judge model server

fol_max_len: 4096     # Max token context for FOL model
fol_gpu_mem: 0.30     # GPU memory fraction for FOL model (~7 GB on 24 GB GPU)

judge_max_len: 5120   # Max token context for judge model
judge_gpu_mem: 0.60   # GPU memory fraction for judge model (~14 GB on 24 GB GPU)
# IMPORTANT: fol_gpu_mem + judge_gpu_mem should be < 0.95

# ── GENERATION SETTINGS ────────────────────────────────────────────────────
temperature: 0.8      # Sampling temperature for FOL model (higher = more diverse)
max_tokens: 1024      # Max new tokens the FOL model generates

judge_temperature: 0.6
judge_max_tokens: 2048      # Longer for thinking models (includes <think> tokens)
judge_enable_thinking: true  # Enable Qwen3 "thinking" mode (better reasoning, slower)

# ── METRIC WEIGHTS ─────────────────────────────────────────────────────────
standard_weight: 0.5  # Weight for FOL parsing + syntax validity score
judge_weight: 0.5     # Weight for LLM judge score
# Combined score = standard_weight * standard_score + judge_weight * judge_score

# ── GEPA BUDGET (choose exactly one) ───────────────────────────────────────
gepa_auto: light
# Presets: light (~30 min), medium (~2-4 h), heavy (~8-12 h)

# Alternatively, specify exact budgets:
# max_full_evals: 5       # Number of full evaluation rounds
# max_metric_calls: 500   # Total number of metric function calls

# ── FORMAT SETTINGS ────────────────────────────────────────────────────────
facts_proof_divided: false
# false = unified format: all FOL in one <logic> block
# true  = separated format: FACTS: ... PROOF: ... sections

# ── PERFORMANCE ────────────────────────────────────────────────────────────
num_threads: 4  # Parallel evaluation threads (keep low to avoid GPU OOM)

# ── OUTPUT & LOGGING ───────────────────────────────────────────────────────
output_dir: gepa_uni_logic_long
# Results go to: output/gepa/<output_dir>/

use_wandb: false
# Set to true and configure wandb_project/wandb_name to track runs in W&B

log_level: INFO
seed: 42
```

### Common config adjustments

**To do a quick smoke test (< 5 minutes):**
```yaml
num_train_samples: 20
num_val_samples: 10
downsample: 0.01
max_full_evals: 2
num_threads: 2
judge_enable_thinking: false
output_dir: quick_test
```

**To run a full production run:**
```yaml
num_train_samples: 8000
num_val_samples: 2000
downsample: 0.5
gepa_auto: medium
num_threads: 8
output_dir: production_run_v1
```

---

## 7. Running the Optimization

**Always run from the `gepa_optimization/` directory:**

```bash
cd gepa_optimization

# Recommended: let run.sh handle vLLM servers automatically
./run.sh config/gepa_config.yaml

# Or run the Python script directly (assumes vLLM servers are already up)
python3 gepa_optimize_prompt.py --config config/gepa_config.yaml
```

**Overriding config values from the command line:**

Any config key can be overridden on the command line:

```bash
python3 gepa_optimize_prompt.py \
    --config config/gepa_config.yaml \
    --num_val_samples 50 \
    --max_full_evals 3 \
    --output_dir my_test_run
```

---

## 8. Understanding the Output

After a run, look in `output/gepa/<output_dir>/`:

```
output/gepa/gepa_uni_logic_long/
├── optimized_instructions_20250514_143022.txt   ← THE MAIN RESULT
├── optimized_program_20250514_143022.json       ← Full DSPy program state
├── optimization_results_20250514_143022.json    ← Summary JSON
├── gepa_optimization_stats.csv                  ← Scores appended per run
└── logs/                                        ← GEPA internal logs
```

**`optimized_instructions_*.txt`** — This is the most important file. It contains:
- The best system prompt GEPA found
- The original seed score and the optimized score
- Metadata (model, template, datasets used)

Example content:
```
OPTIMIZED INSTRUCTIONS
================================================================================

You are solving a multi-hop reasoning question using First-Order Logic (FOL).
[...GEPA-optimized instructions here...]

================================================================================
OPTIMIZATION INFO
================================================================================
FOL Model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
Seed Score: 0.3124
Optimized Score: 0.5801
Improvement: 0.2677
```

**`gepa_optimization_stats.csv`** — One row per run, useful for tracking experiments over time.

**Interpreting scores:**

The combined score is between 0 and 1:
- `0.0` — complete failure (no valid FOL, wrong format)
- `0.3–0.5` — mediocre (some valid FOL, poor reasoning)
- `0.6–0.8` — good (mostly valid FOL, decent logical structure)
- `0.8+` — excellent

An improvement of 0.1+ from seed to optimized is a meaningful gain. GEPA typically delivers 0.05–0.3 improvement depending on the initial template quality and budget.

---

## 9. Monitoring a Run

**Watch the logs in real time:**

```bash
# Main optimization log (printed to stdout when using ./run.sh)
# To follow vLLM server logs:
tail -f output/vllm_fol.log
tail -f output/vllm_judge.log
```

**What you'll see in stdout:**

```
2025-05-14 14:30:22 - INFO - Loading template: qwen3_4B/uni_logic_answer
2025-05-14 14:30:23 - INFO - Loading datasets...
2025-05-14 14:30:45 - INFO - Total merged examples: 12430
2025-05-14 14:30:45 - INFO - Train examples: 8000
2025-05-14 14:30:45 - INFO - Val examples: 2000
2025-05-14 14:30:46 - INFO - Initializing DSPy language models...
2025-05-14 14:30:47 - INFO - Evaluating seed program...
  [Evaluate] 100%|████████| 2000/2000 [12:34<00:00]
2025-05-14 14:43:21 - INFO - Seed program score: 31.24%
2025-05-14 14:43:21 - INFO - STARTING GEPA OPTIMIZATION
  [GEPA] Iteration 1/N ...
  ...
2025-05-14 16:10:05 - INFO - OPTIMIZATION COMPLETE
2025-05-14 16:10:05 - INFO - Optimized program score: 58.01%
2025-05-14 16:10:05 - INFO - Improvement: 26.77%
```

**Checking GPU usage:**

```bash
watch -n 2 nvidia-smi
```

You should see two vLLM processes sharing the GPU.

---

## 10. Troubleshooting

### vLLM server won't start

**Symptom:** `ERROR: FOL server failed to become healthy after 5 minutes.`

**Check the log:**
```bash
tail -50 output/vllm_fol.log
```

Common causes:
- **Out of GPU memory:** Reduce `fol_gpu_mem` and `judge_gpu_mem`. Make sure they sum to < 0.95.
- **Model not found:** Check the model name is a valid HuggingFace ID. First download takes time.
- **Port already in use:** Kill old vLLM processes: `pkill -f "vllm serve"`
- **CUDA version mismatch:** Ensure vLLM version matches your CUDA version.

### CUDA out of memory during optimization

**Symptom:** `torch.cuda.OutOfMemoryError` in the logs.

**Fix:** Reduce `num_threads` (fewer parallel API calls) and reduce `num_val_samples`.

### `ModuleNotFoundError: No module named 'solver'`

**Symptom:** Python can't find the `solver` or `core` packages.

**Fix:** Make sure you are running from inside the `gepa_optimization/` directory:
```bash
cd gepa_optimization
python3 gepa_optimize_prompt.py ...    # correct
python3 gepa_optimization/gepa_optimize_prompt.py ...  # wrong
```

### `PROVER9` not set / Prover9 not found

**Symptom:** `OSError` or `RuntimeError` mentioning `prover9` when evaluating logic.

**Fix:** Set the environment variable:
```bash
export PROVER9=/path/to/Prover9/bin
# Verify: ls $PROVER9/prover9
```

In Docker, Prover9 is pre-installed at `/Prover9/bin` and `PROVER9` is set automatically.

### `GEPA not available in dspy`

**Symptom:** `ImportError: cannot import name 'GEPA' from 'dspy'`

**Fix:** You need exactly `dspy==3.0.3`. Check and reinstall:
```bash
pip show dspy
pip install dspy==3.0.3
```

### Judge model returns unparseable output

**Symptom:** Many `WARNING - Failed to parse judge response:` lines, judge score always 0.

**Causes and fixes:**
- The judge model is too small to follow the scoring format. Switch to a larger judge model.
- `judge_enable_thinking: true` but the model doesn't support thinking. Set it to `false`.
- `judge_max_tokens` is too low — the model truncates before writing the scores. Increase to 4096.

### Dataset download fails

**Symptom:** `ConnectionError` when loading datasets.

**Fix:** Make sure you have internet access and a HuggingFace token if needed:
```bash
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

---

## 11. Quick-Reference Cheat Sheet

```bash
# Build Docker image (from repo root)
docker build -f docker/Dockerfile -t synfol:latest .

# Start container with GPU
docker run --rm -it --gpus all \
    -v $(pwd)/gepa_optimization:/workdir/gepa_optimization \
    -v ~/.cache/huggingface:/hf_home \
    synfol:latest bash

# Inside container: run optimization
cd /workdir/gepa_optimization
./run.sh config/gepa_config.yaml

# Check GPU memory during run
watch -n 2 nvidia-smi

# Check if a vLLM server is healthy
curl http://localhost:8000/health
curl http://localhost:8001/health

# Kill all vLLM servers
pkill -f "vllm serve"

# Quick smoke test (modify config inline)
python3 gepa_optimize_prompt.py \
    --config config/gepa_config.yaml \
    --num_train_samples 10 \
    --num_val_samples 5 \
    --max_full_evals 1 \
    --output_dir smoke_test

# Read the best optimized prompt
cat output/gepa/gepa_uni_logic_long/optimized_instructions_*.txt | head -50

# Check scores across runs
cat output/gepa/gepa_uni_logic_long/gepa_optimization_stats.csv
```

---

## Next Steps

Once you have a good optimized prompt (score > 0.6), you can:

1. **Copy the prompt into a template file** (`templates/qwen3_4B/your_new_template.yaml`) and use it for inference or fine-tuning.
2. **Run GEPA again** using the optimized prompt as the new seed (update `initial_template` in config), for a second round of refinement.
3. **Try a heavier budget** (`gepa_auto: medium` or `gepa_auto: heavy`) to push the score further.
4. **Evaluate on the test set** using `src/evaluate.py` with the optimized template to get final benchmark numbers.
