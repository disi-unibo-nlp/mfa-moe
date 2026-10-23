# Ported router capture (seven source models)

Provenance note for code copied unchanged into the repository. Step 6 of the cleanup plan simplifies it.

## Where it came from

- Source: `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/measurement_study/` (a separate checkout,
  last commit 9f16c55). Its files are byte-identical to the last frozen bundle
  `routing-study/results/measurement-study/campaign-v3/bundles/t2-20260927e/measurement_study/`.
- Copied (17 modules, the import closure of `campaign_v3_capture` plus the gate and the worker extension):
  `annotation_provenance, attach_labels, bundle, campaign_v3_capture, campaign_v3_plan, capture_buffers,
  capture_gate, contract, gemma_cache_policy, gpt_attention, hf_chunked_capture, join_collection,
  native_selection, production_capture, production_plan, sentence_schema, vllm_capture`.
  `capture_gate` is not imported by the capture module but is the capture-agreement gate of the same chain;
  `vllm_capture` is reached through the `worker_extension_cls` string in `production_capture`.
- Only the imports changed: `measurement_study.X` is now `capture.X`, including the
  `worker_extension_cls` string `capture.vllm_capture.MeasurementWorkerExtension`.
- Not copied: `production_analyze, native_replay, recovery_*, nemotron_*, gemma_download, gemma_source_recovery,
  final_source_join, label_*, plan_collection, prepare_campaign, size_collection, smoke, completed_labels,
  artifact_*` (pilot, recovery and storage tooling that the capture chain does not import) and the bundle
  `scripts/` (`submit_capture_v3.py`, `run_campaign_v3_module.py` and others: Slurm plumbing).

## Bundles that actually ran

Captures were produced from frozen bundles under `routing-study/results/measurement-study/campaign-v3/bundles/`:

| model | bundle |
|---|---|
| gpt, qwen330b (Hugging Face backend) | `t2-20260926a` |
| glm | `t2-20260927b`, `-c`, `-d`, then `-e` (the dispatch file was edited four times on 2026-09-27; the last one, 18:48, names `-e`) |
| nemotron, gemma, qwen36, qwen35 (vLLM backend) | `t2-20260927a` |

The copy here is `t2-20260927e`. Against it, `t2-20260926a` and `t2-20260927a` differ in `campaign_v3_capture`
(56 changed lines), `campaign_v3_plan` (4) and `capture_buffers` (155); `t2-20260927d` differs in
`capture_buffers` (15). The repository modules the capture imports were bundled with the code
(`bundle/src`); the repository's current versions differ from the `t2-20260927e` bundle in
`correlation_pipeline/continuation.py` (additive), `dynamics/routing.py` (adds `expert_weight_occupancy`),
`dynamics/evaluation.py` and `dynamics/cli.py`.

## How it was launched (commands recorded; sbatch files are not copied)

Driver chain (stdlib Python in the bundle, `scripts/submit_capture_v3.py`, dispatched per model by
`tools/tmp/claude-analysis-2026-09-24/campaign-v3/code/capture_dispatch-*.py`):

```
submit_capture_v3.py start --model M --cohort A|B --bundle BUNDLE
  plan-verify  -> CPU job: run_campaign_v3_module.py --module measurement_study.production_plan ...
  verify       -> GPU job (below, MODE=verify)
  after-verify -> plan-cohort: production manifests sized from the verify receipt -> shard GPU jobs (MODE=capture)
  gate         -> CPU job: run_campaign_v3_module.py --module measurement_study.capture_gate --manifest-dir ...
```

GPU job (`tools/tmp/claude-analysis-2026-09-24/campaign-v3/code/capture_sbatch-*.sbatch`): partition
`boost_usr_prod`, qos `normal`, 1 node, 2 GPUs, 16 CPUs, 240 GB, 16 h, `--no-requeue`; `module load
python/3.11.7 cuda/12.6`; offline Hugging Face; `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`,
`VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_RPC_BASE_PATH=$RUN/tmp`,
`MEASUREMENT_BUNDLE=$BUNDLE`, `MEASUREMENT_DEADLINE_EPOCH` from `squeue %e`; for gpt only,
`LOCAL_KERNELS=kernels-community/gpt-oss-triton-kernels=<snapshot 0f351046...>` and the
`envs/forward-kernels-0.16` overlay on `PYTHONPATH`. Command:

```
srun envs/vllm-cu129/bin/python -B -u -m measurement_study.campaign_v3_capture \
    SOURCE --manifest MANIFEST --bundle BUNDLE [--verify-only]
```

The ported module does not run yet: `main()` calls `bundle.worker_init`, which refuses any repository module (a module or
package directly under `src/`) loaded from outside a frozen bundle, and the old bundles hold
`measurement_study/*.py`, not `capture/*.py`.
Step 6 replaces the bundle check with a release snapshot.

## Known leftovers (not touched in the port)

- `bundle.py` verifies loaded modules against a bundle's `SHA256.json` and so blocks running from the repository
  (see above); it goes away when the release script takes its place.
- `join_collection.join_cohort` appends the bundle's `scripts/` to `sys.path` and imports
  `analyze_available_metrics` from it. That helper (a pilot analysis script) is not ported; the function is only
  used when planning cohorts, never by capture or the gate.
- `production_capture` hashes `vllm_capture.py` into `capture_code_sha256`; the value of a re-run differs from the
  historical receipts because the file text changed (imports).

## Later additions in this package

- `lfm_replay.py` is `replay` from the former `study/lfm.py`, unchanged.
- `features.py`, `views.py` and `dynamics/` (sentence tables and per-layer routing reducers) came from
  `correlation_pipeline/` and are imported by `production_capture`.
- The import lines of the ported modules that named `correlation_pipeline.*`, `jsonl` or `provenance` now name
  `capture.*`, `store` and `provenance`; nothing else in them changed.
