# MoE routers during chain-of-thought reasoning

This repository holds the code behind a journal paper on how the routers of mixture-of-experts (MoE) language models behave while the model reasons step by step.
Eight models solve competition-math problems. A judge model labels each reasoning sentence with one of seven states (Read, Analyze, Plan, Implement, Explore, Verify, Monitor).
We replay the saved text through each model and record which experts the routers pick, ask whether routing says anything about the state of a sentence beyond its text, and then edit the routing during generation to see what changes.
The manuscript lives in `paper/`, which this repository does not track (it is an Overleaf checkout kept next to the code; see `.gitignore`).

The eight models, with the short names used in every command: `gpt` (GPT-OSS-20B), `nemotron` (Nemotron-3.5-Lightning-30B-A3B), `qwen35` (Qwen3.5-35B-A3B), `qwen36` (Qwen3.6-35B-A3B), `gemma` (Gemma-4-26B-A4B), `glm` (GLM-4.7-Flash), `qwen330b` (Qwen3-30B-A3B) and `lfm25` (LFM2.5-24B-A2B). Their checkpoints and router types are in [src/models](src/models/README.md).

## Layout

| Folder | What it holds |
| --- | --- |
| `scripts/` | Bash only. One launcher per experiment, the model servers, the shared Slurm setup. |
| `src/` | Python only, one folder per topic (generation, labeling, capture, routing, steering, paper). |
| `tests/` | Unit tests, one folder per `src/` topic. |
| `results/` | Git-ignored outputs of generation and labeling runs. |
| `report/` | Receipts: small JSON files that record what a finished run read and produced, each with a hash of its own content. Whether a stage passed is decided by its receipt, not by a log or a job state. |
| `paper/` | Not tracked here: the manuscript (an Overleaf checkout), and the support files around it sit in this folder on disk. The research wiki is in `docs/wiki/`. The code in `src/paper` reads and writes it. |
| `configs/`, `data/`, `requirements/` | The frozen design of the eight-model study, small input data, the lock file of the judge-client environment. |
| `TODO.md`, `scripts/<topic>/{TODO,DECISIONS,WALKTHROUGH}.md` | Plans, decisions and walkthroughs, each under a plain label; the root `TODO.md` lists the open work. `AGENTS.md` is the dated log of what is running now. |

Every folder under `scripts/`, `src/` and `tests/` has its own README. Start there when you work on one part.

## Setup

- Everything runs on the CINECA LEONARDO cluster. Run GPU and long CPU work through Slurm only. The login node is for short checks (600 s of CPU, no GPU).
- Two environments live in `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/`. `vllm-cu129` (vLLM 0.29, torch with CUDA 12.9, scipy) is the one every Slurm job uses. `correlation-client-3.11` (numpy, pandas, pydantic, pytest; no torch, no scipy) is for light tests.
- The judge client needs a third environment with DSPy, built once by `scripts/setup/build_judge_client_env.sh`.
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
| Trace generation, 8 models | `generation/generate_answers.sbatch HF_MODEL DATASET SHARD NSHARDS OUT_DIR`; `lfm25`: `sbatch slurm/run_module_gpu.sbatch steering.lfm25_generation native --shard I/N` | `generation.generate_answers`; `steering.lfm25_generation` | `results/correlation_pipeline/card-v3/MODEL/generation/<checkpoint>/<dataset>/traces.jsonl` | Benchmark tables, setup |
| Answer scoring (J1) | `generation/list_rejected_answers.sh TRACES_DIR ITEMS_JSONL`, then `generation/judge_answers.sbatch ITEMS_JSONL VERDICTS_JSONL` | `generation.scoring` (inside generation), `generation.judge_answers` | judge verdicts in `VERDICTS_JSONL` | Accuracy in the benchmark tables |
| Benchmark accuracy tables | `routing/summarize_benchmark_accuracy.sh OUTCOMES_JSON OUT_DIR`, `routing/inventory_generated_corpora.sh OUT_DIR` | `routing.benchmark_accuracy`, `routing.benchmark_inventory` | `OUT_DIR/*.csv`, the inventory's CSV tables and receipts | Tables `benchmarks`, `benchmark_details` |
| Judge optimisation (GEPA) | `labeling/optimize_judge_prompt.sh SERVER_URL DATASET_DIR SEED_PROMPT OUT_DIR` | `labeling.gepa.optimize_judge` | `selected_program_*.json` (the frozen judge) | Appendix, external annotation development |
| Sentence sampling | `labeling/choose_sentences.sh TRACES_DIR OUT_DIR` | `labeling.choose_sentences` | `cohorts.json`, `parts.json`, four label part roots | Appendix, sentence sampling |
| Sentence labeling | `labeling/label_sentences.sbatch TRACE_ROOT OUT_DIR [JUDGE_PROGRAM]` | `labeling.label_sentences` | `annotations.json`, `summary.json` | Every exhibit that uses states |
| Router capture | `capture/capture_routers.sbatch MANIFEST OUT_DIR`, then `capture/check_capture_complete.sh MANIFEST_DIR OUT_JSON [CAPTURE_ROOT]` | `capture.capture_routers`, `capture.check_complete` | per-trace packets and tensors, `result.json`, the completeness check's acceptance JSON | Input of all routing results |
| LFM2.5 replay | `sbatch slurm/run_module_gpu.sbatch steering.lfm25_generation replay --shard I/N` | `steering.lfm25_generation`, `capture.lfm_replay` | `lfm25/replay/<attempt>.npz` under the study scratch folder | Input of the LFM2.5 rows |

### Analyses of native routing

| Experiment | Launcher (`scripts/routing/`) | Module (`src/routing/`) | Output or receipt | Used in the paper |
| --- | --- | --- | --- | --- |
| Present-state readout, 7 models | `readout_present_state.sh ANALYSIS_DIR OUT_DIR` | `present_state_readout` | `readout-results.v3.json` | Figure `readout`, table `readout` |
| Present-state readout, LFM2.5 | `run_lfm25_analyses.sh STUDY_DIR OUT_DIR` | `lfm25_analyses` (`derive`, `readout`, `dynamics`) | `report/final-routing-20261015-r1/lfm25/READOUT_RESULTS.json` (a rerun writes `OUT_DIR/READOUT_RESULTS.json`) | Same figure and table |
| Anticipation (GPT-OSS, Qwen3.6) | `anticipate_next_state.sh ANALYSIS_DIR OUT_DIR` | `next_state_anticipation` | `anticipation-results.json` | Not in the paper (replaced by the eight-model next state) |
| Next state, 8 models | `predict_next_state.sh STUDY_DIR [HF_MODEL...]` | `next_state` | `report/final-routing-20261015-r1/driver/NEXT_STATE.json` | Figure `next_state`, table `next_state` |
| Expert selection coverage | `export_selection_coverage.sh CAMPAIGN_DIR OUT_DIR` (LFM2.5: `run_lfm25_analyses.sh STUDY_DIR OUT_DIR`) | `expert_selection`, `lfm25_analyses` | `selection_coverage.csv` | Figure `selection`, tables `selection`, `selection_folds` |
| Occupancy and transitions | `compute_state_dynamics.sh CAMPAIGN_DIR OUT_DIR [HF_MODEL...]` (LFM2.5: `run_lfm25_analyses.sh STUDY_DIR OUT_DIR`) | `state_dynamics`, `lfm25_analyses` | `NATIVE_DYNAMICS_SUMMARY.json` per model (Qwen3.6: `report/final-routing-20261015-r1/transition-pairs/qwen36/NATIVE_DYNAMICS_SUMMARY.original-sample.json`, all 1,486 families) | Figures `occupancy`, `state_transitions`, `transition_summary`; tables `occupancy`, `state_transitions` |
| Next-sentence labels for the transitions, 8 models | `label_next_sentences.sh STUDY_DIR [SOURCE_CLASSES [SHARDS]]` (judge shards on GPU) | `labelled_sentence_table`, `next_sentence_labels` | `report/final-routing-20261015-r1/driver/TRANSITION_PAIRS.<variant>.json`, `.../transition-pairs/<model>/<variant>/NATIVE_DYNAMICS_SUMMARY.json` | Figure `state_transitions`, table `state_transitions`: fills the Read and Verify rows that have no labelled successor (the exporter still reads `state_dynamics`; switching it is open) |
| Efficiency link | `estimate_efficiency_link.sh STUDY_DIR` | `efficiency_link` | `report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json` | Table `efficiency_link`; picks the experts of whole-reasoning steering |
| Lexical classifier | `fit_lexical_classifier.sh LEXICAL_DIR OUT_DIR` | `lexical` | five fold classifiers in `OUT_DIR` | Appendix, lexical reading of continuations |

### Routing interventions

Launchers are in `scripts/steering/`, modules in `src/steering/`.

| Experiment | Launcher | Module | Output or receipt | Used in the paper |
| --- | --- | --- | --- | --- |
| Pulsed steering, main and strong doses | `run_pulsed_steering.sh STUDY_DIR HF_MODEL SHARDS HOURS [main\|strong]` (once per model), then `analyze_pulsed_steering.sh STUDY_DIR OUT_DIR` | `pulsed_steering`, `pulsed_outcomes`, `pulsed_effects` | `report/final-routing-20261015-r1/driver/analysis-final/EFFECTS.json` and `.../analysis-strong/EFFECTS.json` | Figure `routing`, tables `routing`, `routing_levels`, `routing_strong` |
| Whole-reasoning steering and dose sweep | `seal_release.sh KIND NAME TESTS [OPTION...]`, then `submit_release_jobs.sh RELEASE_DIR [JOB...]`; KIND `continuation` runs what a complement left unfinished | `release`, `whole_reasoning`, `whole_reasoning_effects`, `plan_continuation` (the continuation's plan) | `report/whole-reasoning-20261021-r20/analysis-combined/EFFECTS_WHOLE.json` | Figures `whole_states`, `whole_dose`; tables `whole`, `whole_cells`, `whole_occupancy`, `whole_levels` |
| Outcome-selected nudge | `measure_expert_usage.sh REPORTS_DIR OUT_DIR HF_MODEL...`, `seal_release.sh outcome ...`, `decide_and_launch_next.sbatch EFFECTS NAME TESTS ...` | `outcome` | `analysis/EFFECTS_OUTCOME.json` in the release | Not in the paper. This is the next line of work. |
| Outcome steering r2: native attempts with routing, 8 models | `seal_release.sh capture NAME TESTS ...` and `submit_release_jobs.sh`, then `derive_outcome_sets.sh RELEASE_DIR HF_MODEL...` | `outcome` (`capture`, `classes`, `select`, `evidence`, `attempts`, `derive`) | `report/outcome-capture-20261023-r2/MODEL/SELECTION_R2.json`, `SIGNAL.json`, `ATTEMPT_SELECTION.json` | Not in the paper. |
| Outcome steering r2: class-conditioned sweep, then the confirmed cell | `seal_release.sh sweep NAME TESTS --half A ...` and `submit_release_jobs.sh`, then `pick_and_launch_half_b.sbatch HALF_A_DIR HF_MODEL NAME TESTS ...` | `outcome` (`manifest`, `analysis`, `pick`) | `analysis/EFFECTS_OUTCOME.json` and `PICK.json` in the release (half A, `report/outcome-steering-20261023-r6`, was stopped before its analysis) | Not in the paper. |

### Paper

| Step | Launcher (`scripts/paper/`) | Module (`src/paper/`) | Output |
| --- | --- | --- | --- |
| Tables and figure data from receipts | `export_paper_tables.sh STUDY_DIR [EFFECTS_WHOLE_JSON]` | `export_routing_results`, `export_whole_results`, `export_cl_exhibits` | table `.tex` files and plotting data in `paper/` |
| Build the figures and the PDF | `build_paper_pdf.sh` | `build_cl_figures`, `build_cl_manuscript` | PDF under `paper/manuscript-support/build` |
| Check the writing rules | `check_writing_rules.sh` | paper-writer rules | rule report |

Unit tests of the live code: `tests/` (see its README). Every steering release first runs the steering and routing tests in a Slurm job (`python -m steering.whole_reasoning validate`).

## Old code

The code of the finished campaigns (earlier pipelines, Docker launchers, the rescue study, the old job graph) was removed from the tree on 2026-10-23. It is kept in the git tag `archive/pre-cleanup-20261023`. Every receipt in `report/` names the commit it was made with, so check out that commit to rerun an old result.
