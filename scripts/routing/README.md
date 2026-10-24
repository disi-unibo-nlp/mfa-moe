# scripts/routing

The analyses of native routing, one launcher each. They read finished captures, labels and receipts and submit one or more CPU jobs (`scripts/slurm/run_module_cpu.sbatch`). Only `label_next_sentences.sh` also submits GPU jobs, for the judge. An `OUT_DIR` must be a folder that does not exist yet. `STUDY_DIR` is the study folder (`report/final-routing-20261015-r1`); its receipts are written there. A model is a Hugging Face id (for example `openai/gpt-oss-20b`).

| Script | What it does |
| --- | --- |
| `readout_present_state.sh ANALYSIS_DIR OUT_DIR` | Present-state readout of the seven captured models: does routing tell a sentence's state beyond its text? About 5 hours on 4 CPUs. |
| `anticipate_next_state.sh ANALYSIS_DIR OUT_DIR` | Does routing predict the next state beyond the past text? GPT-OSS and Qwen3.6. About 21 hours on 4 CPUs. |
| `predict_next_state.sh STUDY_DIR [HF_MODEL...]` | Next-state prediction: one 3-hour job per model (all eight when none is named), then a job that joins the receipts. |
| `export_selection_coverage.sh CAMPAIGN_DIR OUT_DIR` | The held-out coverage table of the expert-selection figure (`selection_coverage.csv`), for the models `CAMPAIGN.json` lists. |
| `compute_state_dynamics.sh CAMPAIGN_DIR OUT_DIR [HF_MODEL...]` | State occupancy, dwell and transitions, and expert nominations, one job per model (a few minutes each); the models `CAMPAIGN.json` lists when none is named. |
| `label_next_sentences.sh STUDY_DIR [SOURCE_CLASSES [SHARDS]]` | Next-sentence labels for the state-transition figure on the eight models of the study: rebuilds Qwen3.6's sentence table, then per model a plan, SHARDS judge jobs (`judge_next_sentences.sbatch`, 2 GPUs, 4 hours each, about 120,000 contexts for Read and Verify) and a summary, and one job that joins the eight panels. A relaunch continues the chain: it skips the stages whose receipts exist (a model whose panel exists is done; one whose plan exists goes to its judges, which resume). `DRY_RUN=1` prints the sbatch lines. |
| `judge_next_sentences.sbatch STUDY_DIR HF_MODEL I/N [SOURCE_CLASSES]` | One judge shard of `label_next_sentences.sh`; a rerun resumes. It sets up like `scripts/slurm/run_module_gpu.sbatch` (the LFM2.5 labels); the DSPy packages come from the pinned copy the campaign names, not from `PYTHONPATH`. |
| `rebuild_sentence_table.sh STUDY_DIR HF_MODEL` | Rebuilds a model's labelled-sentence table on all of its question families (about 1 minute on 8 CPUs) and prints the job ID. Only Qwen3.6 needs it; `label_next_sentences.sh` submits it first. |
| `estimate_efficiency_link.sh STUDY_DIR` | The efficiency link for all eight models: its table, the published receipt and the link test on the pulsed-steering cells (1 hour). |
| `fit_lexical_classifier.sh LEXICAL_DIR OUT_DIR` | Fits the lexical classifier that labels steered continuations, on the `native_<model>.parquet` tables in `LEXICAL_DIR`. About 2 hours on 4 CPUs. |
| `run_lfm25_analyses.sh STUDY_DIR OUT_DIR` | The LFM2.5 analyses: expert selection and nominations first (`derive`), then the readout (about 1.5 hours) and the dynamics, both after `derive` succeeds. The readout writes into the new `OUT_DIR` and checks its inputs against the published receipt, which it never rewrites. |
| `summarize_benchmark_accuracy.sh OUTCOMES_JSON OUT_DIR` | Benchmark accuracy tables from the outcome sidecar. |
| `inventory_generated_corpora.sh OUT_DIR` | The per-model inventory of the generated corpora. |

Run them from the repository root. Those that take no model list read the models from their input (`CAMPAIGN.json`, the study's `BINDING.json`); `predict_next_state.sh` asks the registry (`python -m models.roster`, with Python 3.11 and `PYTHONPATH=src`). To run a single model, name it:

```bash
scripts/routing/readout_present_state.sh /leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/readout-new
scripts/routing/predict_next_state.sh report/final-routing-20261015-r1                       # all eight
scripts/routing/predict_next_state.sh report/final-routing-20261015-r1 openai/gpt-oss-20b    # one model, then the collection
scripts/routing/compute_state_dynamics.sh report/research-campaign-v3-20261014-r5-r2 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/dynamics-new
```

- Inputs are saved folders you name: the study folder for the next-state, LFM2.5, next-sentence and efficiency-link jobs, the capture record `report/research-campaign-v3-20261014-r5-r2` for `compute_state_dynamics.sh`, `report/expert-memory-study-v1` for `export_selection_coverage.sh`, and the saved analysis caches (`ANALYSIS` in `src/store.py`) for the readout and anticipation. They are only read.
- `readout_present_state.sh` and `anticipate_next_state.sh` redo expensive published analyses (19 and 84 core-hours). They write a new folder and compare their inputs with the frozen specification of the first run before fitting. That specification names its inputs by absolute path, so the published run is reproduced only with the analysis folder it used (`ANALYSIS` in `src/store.py`); any other `ANALYSIS_DIR` stops at that comparison.
- `label_next_sentences.sh` names the receipt folder of the rule with `python -m routing.next_sentence_labels variant`, so it needs Python 3.11 and `PYTHONPATH=src` too.
- Which module runs, what it writes and which paper exhibit uses it: [`src/routing`](../../src/routing/README.md).
