# MoE Routing Dynamics During Chain-of-Thought Reasoning

## Scope

This project analyzes how Mixture-of-Experts (MoE) models route tokens through experts during mathematical reasoning, focusing on traces that contain **backtracking**, **self-correction**, **contradiction**, or **failure**. The goal is to connect three levels of analysis at token/step granularity:

1. Visible CoT text (reasoning steps, errors, corrections)
2. Hidden-state geometry (representation trajectories)
3. MoE router behavior (expert selection, entropy, stability)

The central hypothesis: successful reasoning traces exhibit stable routing trajectories, while failed traces show characteristic routing disruptions around first-error, contradiction, and backtracking events.

## Setup

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install the project
pip install -e .
```

Requires Python ≥ 3.11.

## Experiments

### Experiment 1 — Failure Taxonomy

**Objective:** Produce reasoning traces and classify reasoning events (backtracking, contradiction, self-correction, final-answer reversal). Produces the descriptive foundation table.

Two kinds of datasets are handled:
- **Generation datasets** (GSM8K, MATH): the model generates its own CoT, which is then classified.
- **Given-solution datasets** (ProcessBench, PRM800K): no generation. The pre-written, gold-labeled solution *is* the reasoning chain we analyze, so `first_error_step` indexes the same chain whose router logits are later extracted. For PRM800K the chain is reconstructed from the per-step ratings (good prefix + first `-1`-rated step). A run over only given-solution datasets never loads the model.

```bash
# Standard prompt
python -m moe_exp.experiment1.run \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --datasets gsm8k math prm800k \
    --output-dir results/exp1

# Self-checking prompt (elicits more backtracking/self-correction)
python -m moe_exp.experiment1.run \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --datasets gsm8k math \
    --self-check \
    --output-dir results/exp1
```

**Output:** `results/exp1/<model>/<dataset>/traces.jsonl`, `results/exp1/summary.json`

### Experiment 2 — Router Logit Extraction

**Objective:** Run a forward pass over Exp1 traces to extract per-token router logits, selected experts, and expert weights for every MoE layer.

```bash
python -m moe_exp.experiment2.run \
    --input results/exp1/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces.jsonl \
    --output results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl
```

**Output:** `traces_with_routing.jsonl` + `tensors/` directory with `.pt` files per trace.

### Event-Centered Routing Analysis

**Objective:** Compute routing metrics (entropy, expert-switch rate, top-k overlap, router margin) in a ±window around reasoning events. Produces the paper's central "before/at/after" table.

```bash
python -m moe_exp.analysis.event_routing \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
    --output results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/event_routing.json \
    --window 5
```

**Output:** JSON with per-event-type aggregated metrics and per-trace breakdowns.

### Experiment 3 — Geometry-Routing Correlation

**Objective:** Test whether hidden-state similarity predicts routing similarity (the "Myth of Expert Specialization" hypothesis). Compare correlation strength across correct vs. failed vs. backtracking tokens.

```bash
python -m moe_exp.experiment3.run \
    --input results/exp1/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces.jsonl \
    --output results/exp3/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/geometry_correlation.json \
    --samples 2000
```

**Output:** Per-layer correlation values with Mantel test p-values.

### Recompute Taxonomy Metrics

**Objective:** Re-run the classifier on existing traces (useful after updating classification logic).

```bash
python -m moe_exp.analysis.recompute_metrics --results-dir results/exp1
```

## Dependency Graph

```
Experiment 1 (trace generation)
│
├──► Experiment 2 (router extraction)     [requires: exp1 traces]
│    │
│    └──► Event Routing Analysis           [requires: exp2 tensors]
│
└──► Experiment 3 (geometry correlation)   [requires: exp1 traces, needs GPU]

Recompute Metrics ◄── exp1 traces (offline, no GPU)
```

```
exp1 ─────► exp2 ─────► event_routing
  │
  └───────► exp3
```

- **exp1 → exp2**: Exp2 needs the `traces.jsonl` from Exp1 as input.
- **exp2 → event_routing**: Event routing analysis loads the `.pt` tensor files saved by Exp2 (no GPU needed).
- **exp1 → exp3**: Exp3 re-runs forward passes itself, only needs the trace text from Exp1.

## Project Structure

```
src/moe_exp/
├── schemas.py              # TraceRecord, StepLabels, ModelLogs
├── utils.py                # Answer extraction, I/O helpers
├── models/
│   ├── loader.py           # Model/tokenizer loading with quantization
│   └── inference.py        # CoT generation, forward-pass log extraction
├── datasets/
│   └── loaders.py          # GSM8K, MATH, PRM800K, ProcessBench
├── analysis/
│   ├── classifier.py       # Backtracking/contradiction/reversal detection
│   ├── step_splitter.py    # CoT text → reasoning steps
│   ├── event_routing.py    # Event-centered routing metrics
│   └── recompute_metrics.py
├── experiment1/
│   ├── run.py              # Trace generation + taxonomy
│   └── taxonomy.py         # Summary table builder
├── experiment2/
│   └── run.py              # Router logit extraction
└── experiment3/
    └── run.py              # Geometry-routing correlation
```

## Models

| Model | Status |
|-------|--------|
| allenai/OLMoE-1B-7B-0924-Instruct | Primary (exp1-3 done) |
| Qwen/Qwen1.5-MoE-A2.7B-Chat | Planned |

## Key Flags

| Flag | Effect |
|------|--------|
| `--self-check` | Uses a self-checking prompt that encourages step verification and error correction |
| `--max-items N` | Limit examples per dataset (for smoke tests) |
| `--quantization bnb-4bit` | Load model in 4-bit quantization |
| `--reasoning-only` | event_routing: skip any traces tagged `task_type="meta_reasoning"` (none are produced by the current loaders; kept for forward compatibility) |
