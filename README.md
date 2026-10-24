# MoE routers during chain-of-thought reasoning

This repository holds the code behind a journal paper on how the routers of mixture-of-experts (MoE) language models behave while the model reasons step by step.
Eight models solve competition-math problems. A judge model labels each reasoning sentence with one of seven states (Read, Analyze, Plan, Implement, Explore, Verify, Monitor).
We replay the saved text through each model and record which experts the routers pick, ask whether routing says anything about the state of a sentence beyond its text, and then edit the routing during generation to see what changes.
The manuscript is in `paper/`.

The eight models, with the short names used in every command: `gpt` (GPT-OSS-20B), `nemotron` (Nemotron-3.5-Lightning-30B-A3B), `qwen35` (Qwen3.5-35B-A3B), `qwen36` (Qwen3.6-35B-A3B), `gemma` (Gemma-4-26B-A4B), `glm` (GLM-4.7-Flash), `qwen330b` (Qwen3-30B-A3B) and `lfm25` (LFM2.5-24B-A2B). Their checkpoints and router types are in [src/models](src/models/README.md).

## Layout

| Folder | What it holds |
| --- | --- |
| `scripts/` | Bash only. One launcher per experiment, the model servers, the shared Slurm setup. |
| `src/` | Python only, one folder per topic (generation, labeling, capture, routing, steering, paper). |
| `tests/` | Unit tests, one folder per `src/` topic. |
| `results/` | Git-ignored outputs of generation and labeling runs. |
| `report/` | Receipts: small JSON files that record what a finished run read and produced, each with a hash of its own content. Whether a stage passed is decided by its receipt, not by a log or a job state. |
| `paper/` | The manuscript (an Overleaf checkout, sources only), the research wiki and the support files around it. |
| `configs/`, `data/`, `requirements/` | The frozen design of the eight-model study, small input data, the lock file of the judge-client environment. |
| `docs/ledger/` | Decisions and plans. `AGENTS.md` is the dated log of what is running now. |

Every folder under `scripts/`, `src/` and `tests/` has its own README. Start there when you work on one part.

## Setup

- Everything runs on the CINECA LEONARDO cluster. Run GPU and long CPU work through Slurm only. The login node is for short checks (600 s of CPU, no GPU).
- Two environments live in `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/`. `vllm-cu129` (vLLM 0.29, torch with CUDA 12.9, scipy) is the one every Slurm job uses. `correlation-client-3.11` (numpy, pandas, pydantic, pytest; no torch, no scipy) is for light tests.
- The judge client needs a third environment with DSPy, built once by `scripts/setup/dspy_env.sh`.
- The package is not installed. Run from the repository root with `PYTHONPATH=src`.
- Compute nodes are offline. Model weights are read from `mfa-moe/cache/hf`.
- `scripts/slurm/env.sh` sets up each job: modules, offline mode, per-job cache folders, and a port derived from the job id, so two jobs on one node do not collide. Launchers source it.

```bash
PYTHONPATH=src /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python \
  -m pytest -q -p no:cacheprovider tests/capture
```

## The experiments

Each paper experiment has one launcher. The launcher submits the job, which calls one module in `src/`. An `OUT_DIR` argument must name a new folder.

### Data

| Experiment | Launcher (`scripts/`) | Module (`src/`) | Output or receipt | Used in the paper |
| --- | --- | --- | --- | --- |
| Trace generation, 8 models | `generation/generate.sbatch MODEL DATASET SHARD NSHARDS`; `lfm25`: `sbatch steering/gpu.sh steering.pulse native --shard I/N` | `generation.generate`; `steering.pulse` | `results/correlation_pipeline/card-v3/MODEL/generation/<checkpoint>/<dataset>/traces.jsonl` | Benchmark tables, setup |
| Answer scoring (J1) | `generation/score.sbatch ITEMS OUT` | `generation.scoring` (inside generation), `generation.answer_equivalence` | judge verdicts in `OUT` | Accuracy in the benchmark tables |
| Benchmark accuracy tables | `routing/benchmarks.sh OUT_DIR` | `routing.benchmark_accuracy`, `routing.benchmark_inventory` | `OUT_DIR/accuracy/*.csv`, `OUT_DIR/inventory` | Tables `benchmarks`, `benchmark_details` |
| Judge optimisation (GEPA) | `labeling/optimize_judge.sh API_BASE OUT_DIR` | `labeling.gepa.run` | `selected_program_*.json` (the frozen judge) | Appendix, external annotation development |
| Sentence sampling | `labeling/sample.sh MODEL` | `labeling.sample_cohorts` | `cohorts.json`, `parts.json`, four label part roots | Appendix, sentence sampling |
| Sentence labeling | `labeling/label.sbatch TRACE_ROOT OUT_DIR` | `labeling.annotation_batch` | `annotations.json`, `summary.json` | Every exhibit that uses states |
| Router capture | `capture/capture.sbatch MODEL MANIFEST OUT_DIR [verify]`, then `capture/gate.sh MANIFEST_DIR OUT.json` | `capture.run`, `capture.gate` | per-trace packets and tensors, `result.json`, the gate's acceptance JSON | Input of all routing results |
| LFM2.5 replay | `sbatch steering/gpu.sh steering.pulse lfm-replay --shard I/N` | `steering.pulse`, `capture.lfm_replay` | `lfm25/replay/<attempt>.npz` under the study scratch folder | Input of the LFM2.5 rows |

### Analyses of native routing

| Experiment | Launcher (`scripts/routing/`) | Module (`src/routing/`) | Output or receipt | Used in the paper |
| --- | --- | --- | --- | --- |
| Present-state readout, 7 models | `readout.sh OUT_DIR` | `readout` | `readout-results.v3.json` | Figure `readout`, table `readout` |
| Present-state readout, LFM2.5 | `lfm.sh OUT_DIR` | `lfm` (`derive`, `readout`, `dynamics`) | `report/final-routing-20261015-r1/lfm25/READOUT_RESULTS.json` (a rerun writes `OUT_DIR/READOUT_RESULTS.json`) | Same figure and table |
| Anticipation (GPT-OSS, Qwen3.6) | `anticipation.sh OUT_DIR` | `anticipation` | `anticipation-results.json` | Figure `next_state`, table `next_state` |
| Next state, 8 models | `next_state.sh` | `next_state` | `report/final-routing-20261015-r1/driver/NEXT_STATE.json` | Not in the paper yet (it will replace the anticipation figure) |
| Expert selection coverage | `selection.sh OUT_DIR` (LFM2.5: `lfm.sh OUT_DIR`) | `selection`, `lfm` | `selection_coverage.csv` | Figure `selection`, tables `selection`, `selection_folds` |
| Occupancy and transitions | `dynamics.sh OUT_DIR` (LFM2.5: `lfm.sh OUT_DIR`) | `dynamics`, `lfm` | `NATIVE_DYNAMICS_SUMMARY.json` per model | Figures `occupancy`, `state_transitions`, `transition_summary`; tables `occupancy`, `state_transitions` |
| Next-sentence labels for the transitions, 8 models | `transition_pairs.sh [SOURCE_CLASSES [SHARDS]]` (judge shards on GPU) | `sentence_table`, `transition_pairs` | `report/final-routing-20261015-r1/driver/TRANSITION_PAIRS.<variant>.json`, `.../transition-pairs/<model>/<variant>/NATIVE_DYNAMICS_SUMMARY.json` | Figure `state_transitions`, table `state_transitions`: fills the Read and Verify rows that have no labelled successor (the exporter still reads `dynamics`; switching it is open) |
| Efficiency link | `efficiency_link.sh` | `efficiency_link` | `report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json` | Table `efficiency_link`; picks the experts of whole-reasoning steering |
| Lexical classifier | `lexical.sh OUT_DIR` | `lexical` | five fold classifiers in `OUT_DIR` | Appendix, lexical reading of continuations |

### Routing interventions

Launchers are in `scripts/steering/`, modules in `src/steering/`.

| Experiment | Launcher | Module | Output or receipt | Used in the paper |
| --- | --- | --- | --- | --- |
| Pulsed steering, main and strong doses | `pulse.sh SHARDS HOURS MODEL...`, then `pulse_analyze.sh OUT` | `pulse`, `pulse_outcomes`, `pulse_effects` | `report/final-routing-20261015-r1/driver/analysis-final/EFFECTS.json` and `.../analysis-strong/EFFECTS.json` | Figure `routing`, tables `routing`, `routing_levels`, `routing_strong` |
| Whole-reasoning steering and dose sweep | `whole.sh KIND NAME TESTS [OPTION...]` (calls `launch.sh`) | `release`, `whole`, `whole_effects` | `report/whole-reasoning-20261021-r20/analysis-combined/EFFECTS_WHOLE.json` | Figures `whole_states`, `whole_dose`; tables `whole`, `whole_cells`, `whole_occupancy`, `whole_levels` |
| Outcome-selected nudge | `outcome_usage.sh`, `whole.sh outcome ...`, `outcome_next.sh` | `outcome` | `analysis/EFFECTS_OUTCOME.json` in the release | Not in the paper. This is the next line of work. |

### Paper

| Step | Launcher (`scripts/paper/`) | Module (`src/paper/`) | Output |
| --- | --- | --- | --- |
| Tables and figure data from receipts | `export.sh [EFFECTS_WHOLE.json]` | `export_routing_results`, `export_whole_results`, `export_cl_exhibits` | table `.tex` files and plotting data in `paper/` |
| Build the figures and the PDF | `build.sh` | `build_cl_figures`, `build_cl_manuscript` | PDF under `paper/manuscript-support/build` |
| Check the writing rules | `check.sh` | paper-writer rules | rule report |

Unit tests of the live code: `tests/` (see its README). Every steering release first runs the steering and routing tests in a Slurm job (`python -m steering.whole validate`).

## Old code

The code of the finished campaigns (earlier pipelines, Docker launchers, the rescue study, the old job graph) was removed from the tree on 2026-10-23. It is kept in the git tag `archive/pre-cleanup-20261023`. Every receipt in `report/` names the commit it was made with, so check out that commit to rerun an old result.
