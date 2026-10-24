# Complete data collection plan

Requested scope: all available labeled completions, all six benchmarks, every actual routed MoE layer, the existing metric suite and new routing/class dynamics. Keep older Qwen3.5 checkpoints as separate model conditions. AIME stays in benchmark reports and the equally weighted six-benchmark summary.

1. Establish checkpoint provenance from the actual generation server configuration, not served aliases or the client's intended replay target. Freeze source hashes, tokenizer revision, layer topology and precision. Record unresolved sources explicitly.
2. Complete native dispatch capture. HF native capture is working for GPT-OSS, GLM and Qwen3-30B. Implement exact-token vLLM replay for Qwen3.6 FP8 and Nemotron NVFP4; resolve Gemma's original execution path. Capture selected IDs, executed weights, full affinities, native selection scores and decoder-input hidden states at every routed depth; preserve shared branches.
3. Verify token/request isolation, padding, recomputation, all-layer coverage, dispatch execution, checkpoint identity and metric completeness. Run short and long saved traces, preserving caps and original answers. Fit boundary thresholds only on question-disjoint calibration data.
4. Recover the four interrupted CPU bundles with per-trace checkpoints. Keep the eleven finished bundles. Recovery job 58471939 completed all four missing tasks, with zero rejected rows; all 15 routing bundles are now complete.
5. Freeze the full collection inventory and resumable work shards by model/benchmark/attempt. Retain missing answers and caps. Refuse to mark partial coverage complete. Wait for the existing GLM/Qwen3 labeling jobs before their class-conditioned comparisons.
6. Measure long-trace throughput, peak GPU/host memory and output bytes. Prepare the full normal-QoS collection scripts and a concrete capped resource proposal. The existing 8 GPU-hour validation authorization permits at most eight two-GPU 30-minute tests; it does not yet authorize the larger full-corpus campaign.
7. Submit approved full collection shards, verify each job ID, and arm bounded watchers. Process complete outputs with identical metrics and question-level weighting. Report per-benchmark and equal six-benchmark summaries, uncertainty, multiple-testing correction, and missing-data limitations.
8. Deliver frozen manifests, per-attempt metrics, capture diagnostics, correlation/class-transition reports, standalone plots and the existing formal LaTeX metric section. No model training or fine-tuning.

## Acceptance by source

| Source | Layers | Current capture status | Next required check |
|---|---:|---|---|
| GPT-OSS | 24 | Short/long native MXFP4 capture passed | Full corpus metrics and resource acceptance |
| GLM | 46 | Short/long native BF16 capture and biased-gap checks passed; final labels and CPU class analysis complete | Full corpus routing |
| Qwen3-30B | 48 | Short/long native BF16 capture passed | Resumed label shards and full corpus routing |
| Qwen3.6 | 40 | Native FP8 short/long capture passed in 58472802 | Full content-bound corpus, calibration thresholds and resource sizing |
| Nemotron | 23 | Native NVFP4 short/long capture passed in 58474338 | Full corpus routing and calibrated metrics |
| Gemma | 30 (topology reference) | Pinned native checkpoint transfer and custom capture code prepared | Resolve data-mover host trust, historical revision and GPU capture |
| Earlier Qwen3.5 sources | All routed depths required | Existing data retain six probe layers | Resolve each execution checkpoint/backend and reacquire missing layers |

A passed capture smoke is not complete data collection. No missing model, layer, marker, label or applicable metric will be silently dropped from the comparative scope.

Current job IDs, measured results and the additional three-job long-trace proposal are recorded in STATUS.md and long-trace-proposal.json. CPU source join 58474478 freezes exact content identities; CPU sizing 58474659 measures lossless storage without removing source artifacts.

## Submitted continuation (2026-09-23)

Long-trace validation extension: 58487262/58487265/58487288 completed and passed. Qwen3 label resumptions: 58487704 and 58487708 after context/checkpoint preflight 58487662. GLM complete-label analysis 58487905 completed; Qwen3 analysis 58487962 and final all-label content join 58488161 are dependency-queued. Long-trace storage/resource sizing 58488054 completed. See STATUS.md for exact counts, context-limit diagnosis, storage constraint and approval scope.
