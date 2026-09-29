# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research code studying MoE router dynamics during chain-of-thought reasoning. The host is
CINECA LEONARDO; the account/cluster rules in `/leonardo_work/IscrC_MIOSR/lmolfett/AGENTS.md`
(write locations, no GPU work on login nodes, Slurm only after the user authorizes the
workload and budget, no commit/push without approval) apply to everything here.

## Environments and tests

The repo `.venv` is empty; do not use it. The working environments live outside the repo in
`/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/`:

- `correlation-client-3.11`: pydantic/numpy/pytest, **no torch**. Use for CPU-only logic tests.
- `vllm-cu129`: vLLM 0.29 + torch cu129. Used by sbatch jobs; tensor tests run in Slurm.

The package is not installed; always set `PYTHONPATH=src`.

```bash
E=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $E -m pytest -q -p no:cacheprovider tests/test_sample_stratified.py
PYTHONPATH=src $E -m pytest -q -p no:cacheprovider tests/test_routing_dynamics.py::test_name
```

Only light, torch-free tests belong on the login node (600 s CPU limit). Tests that need
torch skip or fail in the client env. The tensor test gate is
`sbatch/native_forward_cpu_tests.sbatch` (serial partition, runs `scripts/validate_forward_cpu.py`,
which selects specific tests by name). CUDA loader tests (`tests/test_qwen_quantized.py -k cuda`)
run inside `sbatch/native_forward_debug.sbatch`. Tests never download weights or call a live judge.
Lint config: ruff, line length 100, py311 (ruff is not installed on the host).

The Docker launchers (`run_docker.sh`, `run_all.sh`, `tag_slurm.sh`, `run_gepaLLMAsJudge.sh`)
target an RTX 5090 workstation with Docker and do not run on LEONARDO. On LEONARDO use the
native `sbatch/native_*.sbatch` launchers, which `module load python/3.11.7 cuda/12.6`, use
`vllm-cu129`, redirect all caches into the run directory, run offline (`HF_HUB_OFFLINE=1`),
and refuse to run outside a matching Slurm allocation. Several support `--plan-only` / `--dry-run`
to print the argv without submitting.

## Architecture

`src/moe_exp/` has four active workflows; `old/` and `src/moe_exp/old/` are archived experiments.

- `gepaLLMAsJudge/`: GEPA/DSPy optimization of the prompt that labels reasoning sentences with the
  seven Schoenfeld episode classes. Its frozen program JSON
  (`results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_*.json`) is the judge
  used everywhere downstream.
- `correlation_pipeline/`: the main line of work (see below).
- `probeTest/`: layer-wise hidden-state probes; its `best_by_target` layers define the default
  layer subset retained by forward extraction.
- `moe_guiding/`: a vLLM plugin (registered as a `vllm.general_plugins` entry point) that alters
  top-2 routing at inference time. Exploratory.

Shared: `schemas.py` (`TraceRecord` is the unit record passed between every stage), `models/`
(loading, per-family router adapters, streaming extraction), `datasets/loaders.py`.

### Correlation pipeline stages

1. **Generate** (`generate.py`, `benchmarks.py`, `scoring.py`): vLLM OpenAI server, six default math
   benchmarks, 32,768-token completion budget. Generation saves exact prompt/completion token IDs,
   a tokenizer fingerprint and character offsets (`metadata.token_replay`). Per-model server
   settings (reasoning parser, MTP, context) are in `model_profiles.py`.
2. **Sample** (`sample_stratified.py`): 100k sentence units per source model, one completion per
   `source_problem_id`, stratified by dataset x correctness x reasoning-length quartile, split into
   four 25k parts. Deterministic and plan-hashed; re-running a different plan into an existing
   directory is refused.
3. **Annotate** (`annotation_batch.py`, `batch_judge.py`, `batch_predictor.py`, `annotate.py`):
   the Qwen3.8-27B judge labels sentences in batches with checkpoint resume.
   `annotation_merge.py` / `labels_export.py` verify the parts and publish merged labels.
4. **Forward replay** (`extract.py` -> `models/routing_extraction.py`, `models/router_adapters.py`):
   teacher-forces the saved token IDs through the HF checkpoint. It never re-renders chat templates.
   Streaming hooks reduce router and hidden tensors to compact per-trace features on the fly;
   only top-k expert IDs are persisted unless `--save-raw-tensors` is passed.
   `forward_adapter.py` converts verified batch labels into forward annotation JSONL.
5. **Analyze** (`analyze.py`, `features.py`, `expert_analysis.py`, `views.py`, `view_analysis.py`):
   correctness/episode associations with problem-clustered bootstrap. Repeated attempts are
   never treated as independent. Length-truncated generations are excluded.
6. **Dynamics** (`dynamics/`, CLI `python -m moe_exp.correlation_pipeline.dynamics`): class
   transitions, trailing-window routing overlap/entropy/JSD, and sentence-level expert counts.
   Enabled in extraction with `--dynamics --all-router-layers --router-only`. See
   `report/ROUTING_DYNAMICS.md`.

`spans.py` is the single source of truth for sentence segmentation, trace digests and offset
alignment. Labels, views and dynamics all key off `trace_digest` plus sentence spans, so a
change there invalidates existing annotations.

### Cross-cutting contracts

- **Router semantics differ by family.** Qwen3.5/3.6 and Gemma expose probabilities, which are
  stored as log-probs. Qwen3 and GPT-OSS expose logits. GLM and Nemotron use sigmoid affinities
  with selection-correction biases. Always use the model's actual selected experts rather than
  raw top-k (see `router_adapters.py`). Boundary gaps can therefore be negative.
- **Checkpoints and invalidation.** Every stage resumes from checkpoints keyed by content hashes
  (traces, program, judge settings, replay version, router-capture version, feature schema,
  code). If you change semantics, bump the relevant version so stale outputs are recomputed
  or refused rather than silently reused. `provenance.code_provenance()` hashes all of
  `src/moe_exp/**/*.py`, so any source edit invalidates CPU-gate files that GPU jobs check.
- **Reasoning length** comes from server `reasoning_tokens` with a fallback to counting via
  `token_replay` offsets. vLLM reports 0 for gpt-oss.
- **Outputs** go to `results/` and `data/labels/` (both gitignored) and to scratch paths named
  in the sbatch files. Published trace roots and label sets are provenance; do not overwrite them.

## Docs to read before changing a stage

- `src/moe_exp/correlation_pipeline/README.md`: stage commands, 100k sampling/labeling
  procedure, port allocation per source/part, LEONARDO forward validation.
- `src/moe_exp/correlation_pipeline/MODEL_SUPPORT.md`: per-model profiles, quantization,
  replay and routing contracts.
- `src/moe_exp/correlation_pipeline/REASONING_VIEWS.md`: class/position views and resume rules.
- `src/moe_exp/gepaLLMAsJudge/README.md`: split isolation and the locked test policy.
- `NEXT_STEPS.md`: agreed research priorities (storage audit, prespecified statistics).
- `report/`: LaTeX report and `ROUTING_DYNAMICS.md` (current pilot status and job IDs).
