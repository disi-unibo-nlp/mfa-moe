# scripts/labeling

Three steps that turn finished traces into labelled sentences, and the one-off optimisation of the judge.

| Script | What it does |
| --- | --- |
| `optimize_judge.sh API_BASE OUT_DIR` | Optimises the judge prompt with GEPA against a running server of Qwen3.8-27B (`API_BASE` such as `http://127.0.0.1:8080/v1`). Seed 42, a 21-example few-shot prompt, an LLM-judge reward, balanced-accuracy selection. It ran once on a workstation with llama.cpp and produced the frozen judge. |
| `sample.sh MODEL` | Picks the sentences to label from the finished traces of `MODEL` (cohorts A and B, four parts), as a CPU job. |
| `label.sbatch TRACE_ROOT OUT_DIR` | Starts the judge and labels every selected sentence under `TRACE_ROOT`. 2 GPUs, 12 hours. A rerun resumes from `OUT_DIR/checkpoints`. |

```bash
scripts/labeling/sample.sh gpt
RES=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/card-v3/gpt
R=$RES/cohorts/labels/part-00/generation                       # one of four parts
sbatch scripts/labeling/label.sbatch $R/$(ls $R) $RES/annotations/part-00
```

- `sample.sh` reads `results/correlation_pipeline/card-v3/MODEL/generation/<checkpoint>/` (in the main checkout, `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo`) and writes `.../MODEL/cohorts/` (`cohorts.json`, `parts.json`, and four label part roots `labels/part-0X/generation/<checkpoint>/<dataset>/traces.jsonl`).
- Submit `label.sbatch` once per part. The judge program is the frozen `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`, loaded by default.
- Details, and what the labels contain: [`src/labeling`](../../src/labeling/README.md) and [`src/labeling/gepa`](../../src/labeling/gepa/README.md).

Paper use: the labels define the states in every exhibit except the benchmark tables. The judge optimisation is the appendix on external annotation development.
