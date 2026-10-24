# scripts/routing

The CPU analyses of native routing, one launcher each. They read finished captures, labels and receipts and submit one or more CPU jobs (`scripts/slurm/cpu.sh`). None of them uses a GPU. An `OUT_DIR` must be a folder that does not exist yet.

| Script | What it does |
| --- | --- |
| `readout.sh OUT_DIR` | Present-state readout of the seven captured models: does routing tell a sentence's state beyond its text? About 5 hours on 4 CPUs. |
| `anticipation.sh OUT_DIR` | Does routing predict the next state beyond the past text? GPT-OSS and Qwen3.6. About 21 hours on 4 CPUs. |
| `next_state.sh` | Next-state prediction for all eight models: one 3-hour job per model, then a job that joins them. |
| `selection.sh OUT_DIR` | The held-out coverage table of the expert-selection figure (`selection_coverage.csv`). |
| `dynamics.sh OUT_DIR` | State occupancy, dwell and transitions, and expert nominations, of the seven models, one job per model (a few minutes each). |
| `efficiency_link.sh` | The efficiency link for all eight models: its table, the published receipt and the link test on the pulsed-steering cells (1 hour). |
| `lexical.sh OUT_DIR` | Fits the lexical classifier that labels steered continuations. About 2 hours on 4 CPUs. |
| `lfm.sh` | The LFM2.5 analyses: expert selection and nominations first (`derive`), then the readout (about 1.5 hours) and the dynamics, both after `derive` succeeds. |
| `benchmarks.sh OUT_DIR` | Benchmark accuracy tables and the per-model inventory of the generated corpora. |

```bash
scripts/routing/readout.sh /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/readout-new
scripts/routing/next_state.sh                 # writes report/final-routing-20261015-r1/<model>/NEXT_STATE.json
```

- Inputs are saved folders named in each module: `report/research-campaign-v3-20261014-r5-r2` for `dynamics.sh`, `report/expert-memory-study-v1` for `selection.sh`, `report/final-routing-20261015-r1` for the LFM2.5, next-state and efficiency-link jobs, and the saved analysis caches that `ANALYSIS` in `src/store.py` points to for the readout and next-state jobs. They are only read.
- `readout.sh` and `anticipation.sh` redo expensive published analyses (19 and 84 core-hours). They write a new folder and compare their inputs with the frozen specification of the first run before fitting.
- Which module runs, what it writes and which paper exhibit uses it: [`src/routing`](../../src/routing/README.md).
