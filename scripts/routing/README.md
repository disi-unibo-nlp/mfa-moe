# scripts/routing

The analyses of native routing, one launcher each. They read finished captures, labels and receipts and submit one or more CPU jobs (`scripts/slurm/cpu.sh`). Only `transition_pairs.sh` also submits GPU jobs, for the judge. An `OUT_DIR` must be a folder that does not exist yet.

| Script | What it does |
| --- | --- |
| `readout.sh OUT_DIR` | Present-state readout of the seven captured models: does routing tell a sentence's state beyond its text? About 5 hours on 4 CPUs. |
| `anticipation.sh OUT_DIR` | Does routing predict the next state beyond the past text? GPT-OSS and Qwen3.6. About 21 hours on 4 CPUs. |
| `next_state.sh` | Next-state prediction for all eight models: one 3-hour job per model, then a job that joins them. |
| `selection.sh OUT_DIR` | The held-out coverage table of the expert-selection figure (`selection_coverage.csv`). |
| `dynamics.sh OUT_DIR` | State occupancy, dwell and transitions, and expert nominations, of the seven models, one job per model (a few minutes each). |
| `transition_pairs.sh [SOURCE_CLASSES [SHARDS]]` | Next-sentence labels for the state-transition figure on all eight models: rebuilds Qwen3.6's sentence table, then per model a plan, SHARDS judge jobs (`pairs_judge.sbatch`, 2 GPUs, 4 hours each, about 120,000 contexts for Read and Verify) and a summary, and one job that joins the eight panels. A relaunch continues the chain: it skips the stages whose receipts exist (a model whose panel exists is done; one whose plan exists goes to its judges, which resume). `DRY_RUN=1` prints the sbatch lines. |
| `pairs_judge.sbatch MODEL I/N [SOURCE_CLASSES]` | One judge shard of `transition_pairs.sh`; a rerun resumes. It sets up like `scripts/steering/gpu.sh` (the LFM2.5 labels); the DSPy packages come from the pinned copy the campaign names, not from `PYTHONPATH`. |
| `efficiency_link.sh` | The efficiency link for all eight models: its table, the published receipt and the link test on the pulsed-steering cells (1 hour). |
| `lexical.sh OUT_DIR` | Fits the lexical classifier that labels steered continuations. About 2 hours on 4 CPUs. |
| `lfm.sh OUT_DIR` | The LFM2.5 analyses: expert selection and nominations first (`derive`), then the readout (about 1.5 hours) and the dynamics, both after `derive` succeeds. The readout writes into the new `OUT_DIR` and checks its inputs against the published receipt, which it never rewrites. |
| `benchmarks.sh OUT_DIR` | Benchmark accuracy tables and the per-model inventory of the generated corpora. |

```bash
scripts/routing/readout.sh /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/readout-new
scripts/routing/next_state.sh                 # writes report/final-routing-20261015-r1/<model>/NEXT_STATE.json
```

- Inputs are saved folders named in each module: `report/research-campaign-v3-20261014-r5-r2` for `dynamics.sh`, `report/expert-memory-study-v1` for `selection.sh`, `report/final-routing-20261015-r1` for the LFM2.5, next-state and efficiency-link jobs, and the saved analysis caches that `ANALYSIS` in `src/store.py` points to for the readout and next-state jobs. They are only read.
- `readout.sh` and `anticipation.sh` redo expensive published analyses (19 and 84 core-hours). They write a new folder and compare their inputs with the frozen specification of the first run before fitting.
- Which module runs, what it writes and which paper exhibit uses it: [`src/routing`](../../src/routing/README.md).
