# src/capture

Replays the saved token ids of the generated traces through a model and records which experts the routers chose, with their weights, the router signal and the hidden states of every routed layer. It produced the captures that all routing results of the paper use (`gpt`, `glm`, `qwen330b`, `qwen36`, `nemotron`, `gemma`, `qwen35`). `lfm_replay.py` does the same for LFM2.5 inside the steering code.

## Run

```bash
python -m capture.run MODEL MANIFEST OUTPUT_DIR [--verify-only] [--deadline-epoch EPOCH]   # GPU, 2 GPUs
python -m capture.gate MANIFEST_DIR OUTPUT.json [--captures ROOT]                           # CPU, hashes every file
```

The launchers are `scripts/capture/capture.sbatch` and `gate.sh` (commands and the folder layout of a repeat capture are in [`scripts/capture`](../../scripts/capture/README.md)). Inputs are the frozen manifests of the paper. A manifest lists the traces of one shard of one model with the byte position of each saved trace line and its chosen sentences, the model spec (checkpoint, routed layers, experts, top-k), the cohort file it was cut from and the label files. Every file it names is checked against the hash it was frozen with before use.

- The paper's capture folders (`capture_dir` in the manifests) are only read, by the gate and the analyses. A run refuses to write into a folder that another manifest or other code started, and it never overwrites `verification.json` or `result.json`.
- The verify run (the longest trace and a short one) must pass before any shard. `gpt` compares its query-chunked attention with the native one, `glm` compares vLLM with a Hugging Face reference, the other models only check that every routed layer and token is covered. A shard in a repeat tree only accepts a verification written by the same code.
- Captures are written only under the scratch campaign folder `/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3`, `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r` or `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3`, never inside a git checkout.
- The job stops starting traces in the last 30 minutes (`drained.json`, exit code 75). The same command resumes, as long as the code and the manifest are unchanged.

Output per trace: `<id>.pt.gz` (selected expert ids, fp16 weights, selection margins, reasoning token indices), `<id>.json.gz` (the packet: features, routing windows, class and position views, expert counts) and `<id>.complete.json` (a marker with sizes and hashes). Per shard: `result.json`, `progress.json`, `load.json`, `backend.json`, `run-binding.json`. The gate checks that `result.json` equals the markers and that tokens and traces add up to the plan.

## Files

| File | What it does |
| --- | --- |
| `run.py` | The GPU job: checks the manifest, replays each trace, writes the packet, the tensors and the markers. |
| `gate.py` | The CPU acceptance of a finished cohort. |
| `manifest.py` | Loads and checks a manifest, resume markers and the verification receipt. |
| `backends.py` | `HFBackend` (transformers: gpt, qwen330b, and the glm reference) and `VLLMBackend` (two GPUs, tensor parallel: glm, qwen36, nemotron, gemma, qwen35). |
| `vllm_capture.py` | The vLLM worker hooks: expert selection, dispatch and decoder inputs of every routed layer. No installed package is edited and nothing is intervened on. |
| `buffers.py` | Memory budget and the in-memory buffers of the vLLM capture. |
| `agreement.py` | The glm HF-against-vLLM check and the gpt chunked-against-single-pass check. |
| `reduce.py` | One raw capture to a packet and a tensor file. |
| `features.py`, `views.py`, [`dynamics/`](dynamics/README.md) | The reductions (entropy, margin, overlap, geometry; whole, class and position views; trailing-window routing dynamics). |
| `native_selection.py`, `sentence_schema.py`, `gpt_attention.py`, `hf_chunked_capture.py` | Selector scores and margins; the sentence splitter version of each annotation; bounded gpt-oss attention; chunked gpt-oss replay with a KV cache. |
| `lfm_replay.py` | LFM2.5 replay, started by `steering.pulse lfm-replay`. |

## Views

`views.compute_views` reduces the router and hidden tensors of one trace to per-layer features in three kinds of view, stored in the packet under `cohorts/<cohort>/views` next to the sentence spans.

- `full`: all reasoning tokens of the trace.
- `class`: the tokens of the sentences with one label. Only cohorts that carry labels have class views.
- `position`: fixed windows of reasoning tokens with boundaries `ceil(model_mean_reasoning_tokens * i / 10)` and one overflow window beyond the mean. The mean comes from `position_reference` in the manifest (all traces of the model's cohorts), not from each trace.

How tokens are assigned: the reasoning text is the saved reasoning content or the `<think>` body (the whole completion for a model without either). Sentences come from `labeling.spans` with exact character offsets. A boundary token belongs to the sentence with the largest overlap (the earlier one on a tie, and whitespace goes to the preceding sentence), so every reasoning token is counted once. Switching rate, top-k overlap and hidden-step distance use only adjacent tokens of the same sentence. A scope without tokens has no features (never zero), and `label_coverage` reports the coverage.

## Checks that must stay

The manifest binding and every file it names; the exact saved token ids (each token the engine runs equals the saved one); the dispatched experts being the top-k of the scores; equal hashes on both tensor-parallel ranks; full layer and token coverage; the agreement checks; the verification receipt before any shard. Versions written into outputs (`capture_version` 2, packet `schema_version` 2, `FEATURE_SCHEMA_VERSION` 3, dynamics `SCHEMA_VERSION` 1) must not change without recomputing.

## Not in this repository

- The planner that cut the cohorts into shards and wrote the manifests. Its output is on disk and is the input of a rerun: `campaign-v3/manifests/<model>/{A,B}/<shard>.json`, `ready.json` and `<model>/verify.json` (failed attempts are kept as `<model>.failed-*`). They name the cohort files under `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/cohorts/` and the label files.
- The frozen code bundles. A manifest carries the hash of the bundle it was planned for; a run now records the repository commit and the hash of `src/` instead.
- The step that attached class labels to cohort-B captures after the fact. Nothing here reads its output.

## Where this code came from

It was copied from an earlier measurement study (commit `9f16c55`), which matches the last frozen bundle. The captures themselves ran from several bundles: `gpt` and `qwen330b` (Hugging Face) from the 2026-09-26 bundle, `glm` from four later bundles (its dispatch file was edited four times on 2026-09-27), and `nemotron`, `gemma`, `qwen36`, `qwen35` (vLLM) from the 2026-09-27 first bundle. The code here is the last version, so a rerun of `gpt` or of the vLLM models uses the later glm fixes (a spawned Hugging Face reference, per-layer drift limits). The copy keeps every computation and drops the bundle freeze, the shard planner, the deadline variable, the runtime measurements in `verification.json` and the unused `identical_kernels` option. The `capture_code_sha256` in the packets is the hash of `vllm_capture.py` and differs from the historical receipts because that file changed. Git history before the cleanup is in the tag `archive/pre-cleanup-20261023`.

Tests: [`tests/capture`](../../tests/capture/README.md). The GPU path itself (`backends.py`, `vllm_capture.py`, `run.main`) has no unit test.
