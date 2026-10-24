# scripts/capture

Record which experts the routers pick on the saved traces of one model, and accept a finished capture.

| Script | What it does |
| --- | --- |
| `capture.sbatch MODEL MANIFEST OUTPUT_DIR [verify]` | One GPU job (2 GPUs, 240 GB, 16 hours) that replays the traces of one shard from a frozen manifest. With the word `verify` it runs the 2-trace check that must pass before any shard. |
| `gate.sh MANIFEST_DIR OUTPUT_JSON [--captures ROOT]` | CPU job that re-hashes every captured file of one model and cohort and writes the acceptance file. |

Every capture needs the frozen manifests of the paper (`/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/{A,B}/SHARD.json` and `MODEL/verify.json`). The paper's own capture folders are read only. A repeat capture goes to a new folder under `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/`, with the check in `ROOT/verify` and the shards in `ROOT/0`, `ROOT/1`, and so on:

```bash
M=/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/capture-gpt-A
sbatch scripts/capture/capture.sbatch gpt $M/manifests/gpt/verify.json $ROOT/verify verify   # once per model and tree
sbatch scripts/capture/capture.sbatch gpt $M/manifests/gpt/A/0.json $ROOT/0                  # once per shard
scripts/capture/gate.sh $M/manifests/gpt/A acceptance/capture-gpt-A.json --captures $ROOT    # when all shards are done
```

- The job stops starting traces in the last 30 minutes of its allocation (it writes `drained.json` and exits with code 75). Submit the same command again to resume. It resumes only with unchanged code.
- `MODEL` is one of `gpt gemma glm nemotron qwen330b qwen35 qwen36`. `gpt` also loads a newer `kernels` package (set in the script) so that the MXFP4 weights are not dequantised.
- Run from a clean checkout: outputs record the commit and the hash of `src/`.
- The accepted result of the paper is in `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/acceptance/capture-<model>-<cohort>.json`.

Details, outputs and what is checked: [`src/capture`](../../src/capture/README.md). Paper use: every routing result of the paper reads these captures.
