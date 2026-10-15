# Final-routing progress recap

LATEST UPDATE — 2026-10-15T16:23:32.857118+02:00: all eight first GPU qualifications failed with the same external/internal request-ID mismatch. GPT and LFM also failed their reserved recovery. No model is qualified and no production admission exists. The earlier RUNNING table below is a historical snapshot. Automatic progression will stop at the receipt gates and finalize incomplete unless a separately bound correction and successful qualification resolve this failure. See QUALIFICATION_REQUEST_ID_FAILURE_AUDIT.json for native accounting and the installed-function reproduction. No extra workload was submitted by this audit.

Observed: 2026-10-15T16:03:27.044976+02:00

All eight model qualification allocations are RUNNING, including LFM 60643482. CPU preparation completed successfully, the LFM checkpoint was re-verified in allocation, and all seven inherited nomination inventories were produced. The original allocated validation passed all 31 tests and the real-tokenizer/original-renderer trace-to-export fixture.

CPU pricing 60643316 and its recovery 60643319 failed at the historical seven-model name lookup. Both failures remain recorded. The separately bound correction passed nine tests, preserved all original null streams, and was committed locally as 7269293 before release. Corrected pricing 60647932, admission 60647934, controller 60647935 and guards 60647936/60647937 are accepted, resource-verified and released. Old unused owners retired only after replacement acceptance; native accounting confirms zero elapsed time, Start=None and no allocated resources.

Actual terminal CPU usage plus complete remaining CPU time limits, including the correction, reserve 29.977778 of the original 40 margin-inclusive local preparation hours. No allowance was reset. The corrected 120-minute CPU pricing allocation has no automatic retry. Qualification and reader envelopes are unchanged. Complete measured study admission under the original 42,050-hour allowance and separate WORK/FAST reserves still precedes production.

| Job | Work | Model | Attempt | Shape | Observed state |
| --- | --- | --- | --- | --- | --- |
| 60643311 | validate | all | 0 | 0 GPU / 4 CPU / 24 GiB / 15 min | COMPLETED |
| 60643314 | prepare | all | 0 | 0 GPU / 4 CPU / 24 GiB / 60 min | COMPLETED |
| 60643315 | prepare | all | 1 | 0 GPU / 4 CPU / 24 GiB / 30 min | COMPLETED |
| 60643316 | cpu_price | all | 0 | 0 GPU / 4 CPU / 24 GiB / 120 min | FAILED |
| 60643319 | cpu_price | all | 1 | 0 GPU / 4 CPU / 24 GiB / 60 min | FAILED |
| 60643321 | legacy_price | all | 0 | 0 GPU / 4 CPU / 24 GiB / 120 min | PENDING |
| 60643322 | legacy_price | all | 1 | 0 GPU / 4 CPU / 24 GiB / 30 min | PENDING |
| 60643324 | qualify | gpt | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643325 | qualify | gpt | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643328 | qualify | nemotron | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643332 | qualify | nemotron | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643334 | qualify | qwen35 | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643462 | qualify | qwen35 | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643464 | qualify | qwen36 | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643466 | qualify | qwen36 | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643469 | qualify | gemma | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643471 | qualify | gemma | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643473 | qualify | glm | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643476 | qualify | glm | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643478 | qualify | qwen330b | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643479 | qualify | qwen330b | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643482 | qualify | lfm25 | 0 | 2 GPU / 16 CPU / 120 GiB / 240 min | RUNNING |
| 60643485 | qualify | lfm25 | 1 | 2 GPU / 16 CPU / 120 GiB / 240 min | PENDING |
| 60643487 | reader_select | all | 0 | 0 GPU / 4 CPU / 24 GiB / 15 min | PENDING |
| 60643492 | reader_select | all | 1 | 0 GPU / 4 CPU / 24 GiB / 15 min | PENDING |
| 60643496 | reader_pilot | all | 0 | 2 GPU / 16 CPU / 120 GiB / 120 min | PENDING |
| 60643498 | reader_pilot | all | 1 | 2 GPU / 16 CPU / 120 GiB / 120 min | PENDING |
| 60643499 | admit | all | 0 | 0 GPU / 4 CPU / 24 GiB / 15 min | CANCELLED |
| 60643500 | advance | all | 0 | 0 GPU / 4 CPU / 24 GiB / 5 min | CANCELLED |
| 60643502 | guard | all | 0 | 0 GPU / 4 CPU / 24 GiB / 10 min | CANCELLED |
| 60643535 | guard | all | 1 | 0 GPU / 4 CPU / 24 GiB / 10 min | CANCELLED |
| 60647932 | cpu_price | all | 0 | 0 GPU / 4 CPU / 24 GiB / 120 min | RUNNING |
| 60647934 | admit | all | 0 | 0 GPU / 4 CPU / 24 GiB / 15 min | PENDING |
| 60647935 | advance | all | 0 | 0 GPU / 4 CPU / 24 GiB / 5 min | PENDING |
| 60647936 | guard | all | 0 | 0 GPU / 4 CPU / 24 GiB / 10 min | PENDING |
| 60647937 | guard | all | 1 | 0 GPU / 4 CPU / 24 GiB / 10 min | PENDING |

Reader selection and pricing remain dependency-gated. No production admission or primary results exist yet. Existing coordination 60572515_6 and reporting owners 60509618–60509621 remain owned. Canonical manuscript edits and remote pushes are outside this task.

Resume the existing journal through `scripts/final_routing_ops_r1.py resume`; status is available through either entrypoint. The operational wrapper verifies both bindings and routes subsequent LFM readouts through the same appended registry. Never edit either accepted source binding.

Collection terminal: false. Required results complete: false. Manuscript integrated: false. Later stage receipts, FINAL_EXPORT.json, TERMINAL.json and STATUS_INDEX.json determine completion.

Receipt index: RECOVERED_AUTHORIZATION.json; BINDING.json; PRELAUNCH_VALIDATION.json; ALLOCATED_VALIDATION.json; LFM_CHECKPOINT_VERIFIED.json; LAUNCH.json; waves/00/; OPS_R1_BINDING.json; OPS_R1_PREFLIGHT.json; OPS_R1_ORIGINAL_FAILURES.json; OPS_R1_PREPARATION_ADMISSION.json; OPS_R1_LOCAL_DELIVERY.json; OPS_R1_INITIAL_GRAPH.json; OPS_R1_ACCEPTED.json; OPS_R1_OWNERSHIP_TRANSFER.json; OPS_R1_RELEASED.json; OPS_R1_LAUNCH_OBSERVED.json; submissions/; stages/.
