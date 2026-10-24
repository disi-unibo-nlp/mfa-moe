# Recovery and checkpoint retirement order

Updated 2026-09-24. All new recovery outputs belong in the user's WORK directory. HOME holds small private agent state. Existing healthy Qwen3.6 jobs keep their previously approved output paths.

## Observed inventory

| Model | Saved generations | Label state | Layer capture and analysis | Weight-retirement gate |
|---|---:|---|---|---|
| GPT-OSS | 4,647 | Selected cohorts complete | 1,580 attempts; all four capture shards and both analyses complete | Verify complete WORK archive and offline analysis inputs |
| GLM | 4,647 | 100,000 reconciled; one unknown retained | 1,390 attempts; all four capture shards and both analyses complete | Verify complete WORK archive and offline analysis inputs |
| Qwen3.6 FP8 | 4,647 | Selected cohorts complete | 1,354 selected attempts; three shards running, fourth after label retry | All four captures, both analyses and WORK archive accepted |
| Qwen3 | 4,647 | 90,616 saved; 9,384 remain | 1,530 selected attempts; full collection awaits completed labels/preparation | Full capture, analyses and archive accepted |
| Nemotron NVFP4 | 4,647 | Selected cohorts complete | Four corrected validation captures accepted; 1,411 production attempts prepared | Full capture, analyses and archive accepted |
| Gemma NVFP4 | 4,647 | Selected cohorts reconciled | Short and longest-input validation accepted; 1,642 attempts prepared | Full sensitivity capture, analyses and archive accepted |

| Qwen3.5 GPTQ-Int4 | 4,647 | Partial: 120 labeled trace versions; 114 with token IDs | Historical 4-bit replay retained; native all-layer recovery required, not queued | Resolve native checkpoint, labels/token IDs, validation, capture, analyses and archive |
| Qwen3.5 GGUF UD-Q4-K-XL | 6,627 including GPQA | Partial: 10 labeled trace versions; no saved token IDs in these versions | Historical 4-bit replay retained; native GGUF all-layer recovery required, not queued | Resolve original backend/token IDs, labels, validation, capture, analyses and archive |

Gemma uses BF16 KV cache on A100 and is explicitly a sensitivity variant. Historical generation revision and cache equivalence remain unverified. Older Qwen GPTQ/GGUF experiments retain checkpoint/backend and token-ID coverage gaps.

## Execution order

1. Retire only the approved unused Google Gemma BF16, Nemotron BF16 and unsloth Qwen3.6 BF16 weight shards after hashes, metadata preservation and live-job dependency checks. Target: 176.34 GiB. Job 58530789 completed successfully; all 42 removals and preserved metadata were verified. WORK usage fell to 788.2 GiB / 1 TiB.
2. Archive GPT and GLM generations, token IDs, annotations and checkpoints, native capture packets, metrics, provenance and frozen code into WORK. Existing WORK files remain referenced in place; external files receive checksum-verified copies. Preserve originals. Only then retire their 70.97 GiB of weights. Data-mover host trust currently prevents the copying step.
3. Preserve Qwen3.6's healthy jobs and the one authorized Qwen3 labeling retry. Complete label reconciliation with unknowns retained. Retain the Qwen3.8 judge until Qwen3 labels and source join pass AND Qwen3.5 labeling requirements are resolved and completed.
4. Use released GPU job slots for Nemotron's four balanced shards: 2 GPUs and 4.5 hours each, 36 GPU-hours total; measured compressed output estimate 9.18 GiB, cap 12 GiB. CPU analysis and supplement follow accepted capture.
5. Audit the required Qwen3.5 GPTQ and GGUF recovery tracks before committing additional full captures. Preserve their existing data, resolve native checkpoint/backend and token-ID gaps, define remaining labeling, and measure GPU/storage cost. Prepare and size Qwen3 after label reconciliation. Provisional full collection estimate: 68 GPU-hours; confirm actual prepared manifests and sizing before submission.
6. Subject to the revised budget including Qwen3.5, run Gemma's four sensitivity shards: 2 GPUs and 10.5 hours each, 84 GPU-hours total; measured compressed output estimate 26.18 GiB, cap 28 GiB. Keep its results separate from primary native-cache results.
7. After each model's accepted capture, analysis and archive, retire only its verified unneeded weight files. Generate the final report from observed artifacts and list remaining gaps.

## Resource and retention rules

- Campaign ceiling: 360 GPU-hours. Maximum four concurrent GPU jobs and two debug jobs. Requests use 1.5 times measured runtime including startup, rounded up to 30 minutes.
- Accounting after Nemotron submission: 191.683 GPU-hours spent/committed, 168.317 remaining. Provisional Qwen3 + Gemma would add 152 GPU-hours, for 343.683 total BEFORE Qwen3.5. Qwen3.5 cost is unmeasured; this is no longer a complete campaign estimate. Refresh accounting and stage excess work within the unchanged 360 GPU-hour ceiling.
- Retained experimental-output envelope: 200 GiB. Check quota and concurrent temporary-file allowances before admitting jobs. Stage work that does not fit; no new FAST/SCRATCH destination is authorized.
- Preserve configurations, tokenizer files, revision identities, weight hashes and deletion receipts after retirement. Never delete generations, saved token IDs, labels, unknowns, capture packets or final metrics to free space.
- Full hidden-state/router tensors were temporary and were not retained. Existing native dispatched expert IDs/weights and derived measurements are retained. Additional measurements requiring full tensors need checkpoint reacquisition and replay.
- Frozen bundles remain immutable. Replacement manifests bind their code and any reused accepted artifacts to their original provenance.
- A Slurm COMPLETED state requires exit-code, log and artifact verification. Bounded watchers and acknowledgement receipts attach notifications to the current session.

## Submitted recovery chain

Nemotron captures: 58531562, 58531615, 58531684, 58531747; primary analysis: 58531977; supplement: 58532720. Each capture follows one existing Qwen3.6 GPU lane. Coverage refresh 58532809 replaces cancelled obsolete report 58521917. Extend/regenerate final reporting after the staged Qwen3 and Gemma full collections. All new jobs have verified bounded session watchers.

GPT/GLM stable archive inventories require 18.59 GiB and 15,148 unique external files copied into `results/measurement-study/archive-v2/external/`. The prepared CPU copy remains pending the requested one-time transfer-method exception; no GPT/GLM weight retirement is authorized to execute before archive acceptance.

## Required Qwen3.5 inclusion

Qwen3.5 GPTQ and GGUF are required model conditions, not optional omissions. Recovery requirements and checksum-bound generation manifests are in `results/measurement-study/qwen35-required-recovery.json`. Both remain staged, with no GPU jobs submitted and no estimated full-capture cost yet. The existing direct-HTTPS transfer exception applies only to Gemma. Existing frozen bundles remain immutable; future reporting bundles must inherit the explicit required-source metadata.

Qwen3.5 recovery audit submitted as job 58537311 (2 CPUs, 7 GiB, 30 minutes, zero GPUs). It verifies every content-bound labeled trace location and annotation, checks exact token-ID evidence, and records concrete native-recovery gaps. Native capture jobs remain gated on its result, checkpoint/backend verification and measured sizing.

Qwen3.5 audit 58537311 passed: GPTQ 120 labeled versions / 114 exact token-ID versions; GGUF 10 / 0; no conflicting saved token IDs. Reporting validation 58537579 passed with both required model rows. Queued coverage job 58537773 now uses frozen reporting v8 and replaces cancelled 58532809. Native checkpoint restoration, backend validation and measured GPU sizing remain pending; no Qwen3.5 GPU cost has been added.
