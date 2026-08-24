# Archived numbered experiments

This package preserves Experiments 1-5 and their Python entry points. They are
kept for reproducibility and are no longer the primary workflow.

| Historical experiment | Module |
|---|---|
| Experiment 1: trace generation and taxonomy | `moe_exp.old.experiment1` |
| Experiment 2: router extraction wrapper | `moe_exp.old.experiment2` |
| Experiment 3: geometry-routing correlation | `moe_exp.old.experiment3` |
| Experiment 4: prefix probes | `moe_exp.old.experiment4` |
| Experiment 5: expert-event analysis | `moe_exp.old.experiment5` |

The Experiment 2 implementation now lives in
`moe_exp.models.routing_extraction`, because the active correlation pipeline
also uses it. The archived module is a compatibility wrapper.

Run the complete historical pipeline from the repository root:

```bash
old/scripts/run_pipeline.sh --local \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --dataset gsm8k
```

It writes to `results/old/exp1` through `results/old/exp5`. The original full
README, including methodology and historical commands, is retained at
`old/README.md`; the full report is at `old/report/legacy_full_report.tex`.
