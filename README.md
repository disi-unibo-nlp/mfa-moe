# MoE Routing Dynamics During Chain-of-Thought Reasoning

## Scope

This project analyzes how Mixture-of-Experts (MoE) models route tokens through experts during mathematical reasoning, focusing on traces that contain **backtracking**, **self-correction**, **contradiction**, or **failure**. The goal is to connect three levels of analysis at token/step granularity:

1. Visible CoT text (reasoning steps, errors, corrections)
2. Hidden-state geometry (representation trajectories)
3. MoE router behavior (expert selection, entropy, stability)

The central hypothesis: successful reasoning traces exhibit stable routing trajectories, while failed traces show characteristic routing disruptions around first-error, contradiction, and backtracking events.

## Setup

```bash
# Install PyTorch built for CUDA 12.8 first.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install the project
pip install -e .
```

Requires Python ≥ 3.11.

The editable install includes the CPU analysis dependencies used by Experiments
4 and the linear probes, including scikit-learn. PyTorch remains the only
dependency that must be installed separately so that its CUDA build matches the
host GPU.

## Running the Full Pipeline

`run_pipeline.sh` chains all stages (Exp1 → Exp2 → Event Routing → Exp3 → Exp4 → Exp5) with resume behavior: a stage is skipped when its output file already exists, so re-running after an interruption picks up at the first missing output. Exp2 saves hidden states by default for the prospective Exp4 probes; pass `--skip-exp4` to omit both hidden-state storage and the probe stage.

```bash
# On the SLURM cluster (stages run inside Docker):
sbatch run_pipeline.sh --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k

# Natively, without Docker — e.g. on a vast.ai instance (auto-enabled when
# docker is not installed):
./run_pipeline.sh --local --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k

# Smoke test / self-check variant:
./run_pipeline.sh --local --model ... --dataset gsm8k --max-items 50 --self-check
```

The Docker image defaults to CUDA 12.8 with cu128 PyTorch wheels. The
`CUDA_BASE` and `TORCH_INDEX` build arguments can still be overridden for a
different deployment target.

### Running on vast.ai (RTX 5090)

The 5090 is Blackwell (`sm_120`): it needs CUDA 12.8+ and PyTorch ≥ 2.7 built as cu128. Older wheels (cu121/cu124) fail with "no kernel image is available for execution on the device".

1. **Template:** the official *PyTorch (Vast)* template with the newest CUDA 12.8+ image tag (or a custom template using `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`). When browsing offers, filter **CUDA Version ≥ 12.8**.
2. **Disk:** allocate ~80–100 GB — OLMoE-1B-7B is ~14 GB of bf16 weights, and Exp2 tensors were ~1.5 GB for the full GSM8K set. 32 GB VRAM is comfortable at batch size 1.
3. **Setup on the instance:**

```bash
cd /workspace && git clone <repo-url> moe-mfaExperiments && cd moe-mfaExperiments
pip install -e .
# Keep the HF cache on the persistent volume, or the ~14 GB of weights
# re-download after every instance restart:
export HF_HOME=/workspace/hf_cache
# Verify the GPU is usable — should print (12, 0) without errors:
python -c "import torch; print(torch.cuda.get_device_capability())"
```

4. **Run** (`--local` is auto-enabled when docker is absent, so it can be omitted):

```bash
./run_pipeline.sh --local --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k
```

If the instance is interrupted, re-run the same command — completed stages are skipped.

## Experiments

### Experiment 0a — GEPA-Optimized Schoenfeld Episode Judge

**Objective:** Optimize a local llama.cpp-served judge to assign the seven
sentence-level reasoning episodes from the gold corpus of Li et al. The judge
now receives the SAT problem plus previous/current/next response units so it
can distinguish given facts from deductions and checks. GEPA uses
class-balanced exact-match feedback by default, and final selection is based on
validation balanced accuracy with a per-class recall safety gate.

The default few-shot prompt contains 21 audited synthetic contrastive examples
(three per class). Nested response-grouped cross-validation is available for
configuration selection and excludes six locked-test responses entirely. A
normal final-fit run no longer evaluates the locked test unless
`--evaluate-locked-test` is explicitly supplied.

```bash
# Select a configuration without touching the locked test.
sbatch run_experiment0a.sh \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward balanced \
  --selection-metric balanced_accuracy \
  --cv-folds 5 \
  --gepa-auto light \
  --output-dir results/exp0a/context-balanced-cv-s42

# Fit the selected configuration, still without test evaluation.
sbatch run_experiment0a.sh \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward balanced \
  --gepa-auto light \
  --output-dir results/exp0a/context-balanced-final-s42
```

Every run writes an annotation audit for multi-line, multi-sentence, and mixed
structural/substantive units. Reports include accuracy, balanced accuracy,
macro-F1, per-class metrics, Cohen's kappa, and Kendall's tau-b.

The following are legacy single-sentence results, retained for provenance. The
first completed run used the base prompt, seed 42, GEPA's `light` budget,
and a Q4_K_XL quantization of Qwen3.6-27B on one RTX 3090. The 38 documents
were split before sentence flattening into 26 train, 6 validation, and 6 test
documents (2,382/407/336 sentences). All 407 validation and 336 test requests
returned a valid class.

| Evaluation | Accuracy | Cohen's kappa | Kendall's tau-b | Composite score |
|---|---:|---:|---:|---:|
| Seed validation prompt | 63.88% | 0.558 | 0.481 | 0.760 |
| GEPA-optimized validation prompt | 68.55% | 0.607 | 0.585 | 0.798 |
| Held-out test, optimized prompt | 66.07% | 0.571 | 0.607 | 0.794 |
| Few-shot validation (seed and optimized) | 68.06% | 0.608 | 0.583 | 0.798 |
| Held-out test, few-shot prompt | **69.94%** | **0.618** | **0.667** | **0.821** |

GEPA improved validation accuracy by 4.67 percentage points. The close
validation and held-out composite scores are encouraging, but this is a
single seed with only six held-out documents; repeated response-level splits
are needed before treating the difference as stable. On the test set, the
weakest classes were `Plan` (8/20, 40%) and `Explore` (7/15, 46.7%). The
largest confusions were `Analyze`→`Verify` (16), `Verify`→`Analyze` (14), and
`Read`→`Analyze` (12).

On the same seed and response split, the seven-example few-shot condition
performed better on the held-out test set than the optimized base condition:
69.94% versus 66.07% accuracy, and 0.821 versus 0.794 composite agreement.
However, GEPA did not improve the few-shot validation accuracy: both its seed
and selected optimized prompt scored 68.06%. Thus the current result favors
few-shot prompting for held-out generalization, while the measurable GEPA gain
is confined to the base condition.

See [`src/moe_exp/experiment0a/README.md`](src/moe_exp/experiment0a/README.md)
for setup, metric conventions, and run commands.

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

### Experiment 4 — Prospective Prefix-Only Failure Prediction

**Objective:** Test whether failure is decodable before it occurs. At 25%, 50%,
and 75% of each trace, pool only the hidden/router states already observed and
compare structure-only, router-only, hidden-only, and hidden+router linear
probes. Targets are final incorrectness and whether a gold first-error step is
still in the future. Traces whose first error is already inside the observed
prefix are excluded from the future-error target.

Experiment 4 requires Exp2 output produced with `--extract-hidden-states` (the
full pipeline enables this automatically):

```bash
python -m moe_exp.experiment4.run \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/processbench/traces_with_routing.jsonl \
    --output results/exp4/allenai--OLMoE-1B-7B-0924-Instruct/processbench/prospective_probes.json \
    --model-id allenai/OLMoE-1B-7B-0924-Instruct
```

**Output:** `prospective_probes.json`, containing out-of-fold AUROC, balanced
accuracy, F1, and trace-bootstrap confidence intervals for every prefix,
target, source, and layer. This is prospective with respect to the prefix, but
remains a correlational decoding analysis rather than a causal intervention.

### Experiment 5 — Expert Behavior Around Reasoning Events

**Objective:** For each expert (per layer), measure over/under-use around reasoning events: activation frequency and weight mass per reasoning phase (normal, backtracking, contradiction, self-correction, first-error, final-answer), expert usage before vs. after the first error and before vs. after successful/failed self-corrections, plus global co-activation and top-1 expert transition matrices. Feeds the expert-usage heatmaps and transition-matrix figures. We deliberately describe experts as associated with reasoning-state regions/transitions, not as "math experts" or "logic experts".

```bash
python -m moe_exp.experiment5.run \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
    --output results/exp5/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/expert_events.json \
    --window 5
```

**Output:** `expert_events.json` (per-phase and before/after usage statistics) + `expert_arrays.npz` (per-layer co-activation and transition matrices). Offline over Exp2 tensors, no GPU needed.

### Recompute Taxonomy Metrics

**Objective:** Re-run the classifier on existing traces (useful after updating classification logic).

```bash
python -m moe_exp.analysis.recompute_metrics --results-dir results/exp1
```

### Relabel Existing Traces and Rebuild the Taxonomy

Reapply the current high-precision event classifier without regenerating traces
or router tensors. Gold first-error annotations are preserved.

```bash
python -m moe_exp.analysis.relabel_traces \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
    --output results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing_relabelled.jsonl

python -m moe_exp.analysis.summarize_taxonomy \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing_relabelled.jsonl \
            results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/processbench/traces_with_routing_relabelled.jsonl \
    --output results/exp1/summary_relabelled.json
```

### Full-Trace Linear Probes and Structure Baseline

These offline analyses train trace-grouped logistic probes over pooled router,
hidden-state, or combined features and compare them with a three-feature trace
length/structure baseline. Unlike Experiment 4, they pool the complete trace
and therefore measure decodability rather than prospective prediction.

```bash
python -m moe_exp.analysis.linear_probe \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
    --output results/probes/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/router_probe.json \
    --feature-source router \
    --targets correctness first_error contradiction

python -m moe_exp.analysis.structure_probe \
    --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
    --output results/probes/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/structure_probe.json \
    --targets correctness first_error contradiction
```

Hidden and combined full-trace probes require Exp2 tensors generated with
`--extract-hidden-states`, just like Experiment 4.

## Dependency Graph

```
Experiment 1 (trace generation)
│
├──► Experiment 2 (router extraction)     [requires: exp1 traces]
│    │
│    ├──► Event Routing Analysis           [requires: exp2 tensors]
│    ├──► Experiment 4 (prefix probes)     [requires: exp2 hidden + router tensors]
│    └──► Experiment 5 (expert behavior)   [requires: exp2 tensors]
│
└──► Experiment 3 (geometry correlation)   [requires: exp1 traces, needs GPU]

Recompute Metrics ◄── exp1 traces (offline, no GPU)
```

```
exp1 ─────► exp2 ─────► event_routing
  │           │
  │           ├───────► exp4
  │           └───────► exp5
  │
  └───────► exp3
```

- **exp1 → exp2**: Exp2 needs the `traces.jsonl` from Exp1 as input.
- **exp2 → event_routing**: Event routing analysis loads the `.pt` tensor files saved by Exp2 (no GPU needed).
- **exp2 → exp4**: Exp4 pools prefixes from aligned hidden/router tensors and trains trace-grouped probes (CPU after extraction).
- **exp2 → exp5**: Exp5 loads the same `.pt` tensor files (no GPU needed).
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
│   ├── linear_probe.py      # Full-trace router/hidden/combined probes
│   ├── structure_probe.py   # Non-routing trace-structure baseline
│   ├── relabel_traces.py    # Refresh labels on existing trace files
│   ├── summarize_taxonomy.py # Summarize labelled trace files
│   └── recompute_metrics.py # Refresh Exp1 taxonomy summaries in place
├── experiment1/
│   ├── run.py              # Trace generation + taxonomy
│   └── taxonomy.py         # Summary table builder
├── experiment2/
│   └── run.py              # Router logit extraction
├── experiment3/
│   └── run.py              # Geometry-routing correlation
├── experiment4/
│   └── run.py              # Prospective prefix-only linear probes
└── experiment5/
    └── run.py              # Expert behavior around reasoning events
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
