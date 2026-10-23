# Router capture (seven source models)

Replays the saved token ids of the generated traces through each model and records which experts the routers chose, with
their weights, the router signal and the hidden states of every routed layer. It produced the captures all routing
results of the paper use (gpt, glm, qwen330b, qwen36, nemotron, gemma, qwen35). `lfm_replay.py` is the same idea for
LFM2.5 inside the steering code.

## Run it

Everything needs the frozen manifests of the paper (see "Not in the repository"). The paper's capture folders
(`capture_dir` in the manifests) are only read, by the gate and the analyses: a run refuses to write into a folder that
another manifest or other code started, and never overwrites `verification.json` or `result.json`. A repeat capture of one
model and cohort (`A` or `B`) therefore goes to a new tree with the check in `ROOT/verify` and the shards in `ROOT/0`, `ROOT/1` ...:

```
M=/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/capture-gpt-A          # any new folder under an allowed root
sbatch scripts/capture/capture.sbatch gpt $M/manifests/gpt/verify.json $ROOT/verify verify   # once per model and tree
sbatch scripts/capture/capture.sbatch gpt $M/manifests/gpt/A/0.json $ROOT/0                  # once per shard
scripts/capture/gate.sh $M/manifests/gpt/A acceptance/capture-gpt-A.json --captures $ROOT    # when all shards are done
```

`capture.sbatch` is one GPU job (2 GPUs, 240 GB, 16 h). It stops starting traces in the last 30 minutes (`drained.json`,
exit 75), and the same command resumes after that or after a failure, as long as the code is unchanged (the folder is bound to
the manifest and the hash of `src/`). The verify run (longest trace plus a short one) must pass first, and a shard in a new
tree only accepts a verification from the same tree written by the same code: gpt compares its query-chunked attention with
the native one, glm compares vLLM with a Hugging Face reference; the other models only check that every layer and token is
covered. Captures are only written under the scratch campaign folder `$M`, `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r` or
`/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3`, never inside a git checkout. The gate re-hashes every captured file; without `--captures` it reads the paper's folders.
Run from a clean checkout or a release snapshot: the outputs record the commit and the hash of `src/` (`code` in
`load.json`, `*.complete.json`, `verification.json` and the gate result). The paper's acceptance files are in
`tools/tmp/claude-analysis-2026-09-24/campaign-v3/acceptance/capture-<model>-<cohort>.json`.

## What is here

| File | What it does |
|---|---|
| `run.py` | the GPU job: checks the manifest, replays each trace, writes the packet and tensor files and the markers |
| `gate.py` | the CPU acceptance of a finished cohort |
| `manifest.py` | loads and checks a manifest (binding, cohort file, label files, source trace lines), resume markers, verification receipt |
| `backends.py` | `HFBackend` (transformers) and `VLLMBackend` (two GPUs, tensor parallel) |
| `vllm_capture.py` | the vLLM worker hooks (`worker_extension_cls`): selection, dispatch and decoder inputs of every routed layer |
| `buffers.py` | memory budget and in-memory buffers of the vLLM capture |
| `agreement.py` | glm: HF against vLLM; gpt: chunked against single-pass |
| `reduce.py` | one raw capture to a packet (features, routing windows, class and position views, expert counts) and tensors |
| `features.py`, `views.py`, `dynamics/` | the reductions; `dynamics/classes.saved_layout` is also used by the analyses |
| `native_selection.py`, `sentence_schema.py`, `gpt_attention.py`, `hf_chunked_capture.py` | selector scores and margins, sentence splitter versions, gpt-oss attention and chunked replay |

Output per trace: `<id>.pt.gz` (ids, fp16 weights, selection margins, reasoning token indices), `<id>.json.gz` (packet),
`<id>.complete.json` (marker with sizes and hashes); per shard `result.json`, `progress.json`, `load.json`, `backend.json`,
`run-binding.json`. The gate checks that `result.json` equals the markers and that tokens and traces add up to the plan.

Tests: `tests/capture/` (manifest, gate, resume markers, buffers, agreement checks, `reduce_capture` on a synthetic trace; the
torch ones run in the vLLM environment). The GPU path itself (`backends.py`, `vllm_capture.py`, `run.main`) is not covered.

The checks that must stay: the manifest binding and every file it names, the exact saved token ids (`frame_entries`
compares each token the engine runs with the saved one; `validate_token_replay`), the dispatched experts being the top-k
of the scores, equal hashes on both tensor-parallel ranks, full layer and token coverage, the agreement checks, and the
verification receipt before any shard. Version numbers written into outputs (`capture_version` 2, packet `schema_version` 2,
`FEATURE_SCHEMA_VERSION` 3, dynamics `SCHEMA_VERSION` 1) must not change.

## Not in the repository

- The planner that cut the cohorts into shards and wrote the manifests (`campaign_v3_plan`, `production_plan`, `join_collection`
  and the bundle scripts `submit_capture_v3.py`, `run_campaign_v3_module.py`). Its output is on disk and is the input of a
  rerun: `campaign-v3/manifests/<model>/{A,B}/<shard>.json`, `ready.json`, and `<model>/verify.json` (also failed attempts
  as `<model>.failed-*`). They name the cohort files under `tools/tmp/claude-analysis-2026-09-24/campaign-v3/cohorts/`
  and the label files; both are checked against their hashes.
- The frozen bundles. Every manifest carries `code_binding`, the hash of the bundle it was planned for; a run now records the
  repository commit instead (the old check against the bundle is gone).
- `attach_labels` (attached class labels to cohort B captures after the fact; nothing in the repository reads its output).

## Where it came from

Copied from `routing-study/measurement_study/` (commit 9f16c55), which is byte-identical to the last frozen bundle
`routing-study/results/measurement-study/campaign-v3/bundles/t2-20260927e/`. The captures themselves ran from several bundles:

| model | bundle |
|---|---|
| gpt, qwen330b (Hugging Face backend) | `t2-20260926a` |
| glm | `t2-20260927b`, `-c`, `-d`, then `-e` (the dispatch file was edited four times on 2026-09-27) |
| nemotron, gemma, qwen36, qwen35 (vLLM backend) | `t2-20260927a` |

Against `-e`, `t2-20260926a` and `t2-20260927a` differ in `campaign_v3_capture` (56 lines), `campaign_v3_plan` (4) and
`capture_buffers` (155); `-d` differs in `capture_buffers` (15). The code here is the `-e` logic, so a rerun of gpt or the
vLLM models uses the later glm fixes (spawned HF reference, per-layer drift limits). The step 6 changes keep every computation
and drop the rest: the bundle freeze and import checks, the shard planner, the deadline drain (`MEASUREMENT_DEADLINE_EPOCH`),
the runtime measurements in `verification.json` and the unused `identical_kernels` option of the HF/vLLM comparison. The
`capture_code_sha256` in the packets is the hash of `vllm_capture.py` and differs from the historical receipts, as the file changed.
