# scripts/labeling

Three steps that turn finished traces into labelled sentences, and the one-off optimisation of the judge.

| Script | What it does |
| --- | --- |
| `optimize_judge_prompt.sh SERVER_URL DATASET_DIR SEED_PROMPT OUT_DIR` | Optimises the judge prompt with GEPA against a running server of Qwen3.8-27B (`SERVER_URL` such as `http://127.0.0.1:8080/v1`) on the labelled Schoenfeld responses in `DATASET_DIR`, starting from the prompt in `SEED_PROMPT`. Seed 42, a 21-example few-shot prompt, an LLM-judge reward, balanced-accuracy selection. It ran once on a workstation with llama.cpp and produced the frozen judge, with `DATASET_DIR=data/Schoenfeld_Reasoning` and `SEED_PROMPT=results/exp0a/context-llmjudge-20k-s42/gepa_optimized_prompt_20260811_135905.txt`. |
| `choose_sentences.sh TRACES_DIR OUT_DIR` | Picks the sentences to label from the finished traces in `TRACES_DIR` (cohorts A and B, four parts), as a CPU job. The model is read from the traces. `OUT_DIR` must not exist yet: cohorts are never overwritten. |
| `label_sentences.sbatch TRACE_ROOT OUT_DIR [JUDGE_PROGRAM]` | Starts the judge and labels every selected sentence under `TRACE_ROOT` (with the frozen judge program unless `JUDGE_PROGRAM` is given). 2 GPUs, 12 hours. A rerun resumes from `OUT_DIR/checkpoints`; pass `TRACE_ROOT` spelled exactly as before, because the spelling is part of the checkpoint key. |

```bash
RES=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/card-v3/gpt
scripts/labeling/choose_sentences.sh $RES/generation/openai--gpt-oss-20b $RES/cohorts
R=$RES/cohorts/labels/part-00/generation                       # one of four parts
sbatch scripts/labeling/label_sentences.sbatch $R/$(ls $R) $RES/annotations/part-00
```

- `choose_sentences.sh` reads the folder `TRACES_DIR` that holds `<dataset>/traces.jsonl` (the output of `scripts/generation`) and writes `OUT_DIR` with `cohorts.json`, `parts.json`, and four label part roots `labels/part-0X/generation/<checkpoint>/<dataset>/traces.jsonl`.
- Submit `label_sentences.sbatch` once per part. The judge program is the frozen `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`, loaded by default.
- Details, and what the labels contain: [`src/labeling`](../../src/labeling/README.md) and [`src/labeling/gepa`](../../src/labeling/gepa/README.md).

Paper use: the labels define the states in every exhibit except the benchmark tables. The judge optimisation is the appendix on external annotation development.
