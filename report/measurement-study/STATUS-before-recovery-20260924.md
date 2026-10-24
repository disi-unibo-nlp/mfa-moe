# Measurement campaign status

Verified 2026-09-23T23:40:25.921052+00:00.

## Automatic execution chain

User-authorized native all-layer collection is submitted for five validated models. Each model has a verification run, four resumable GPU shards, a CPU correlation job and a CPU reasoning-dynamics supplement. Final report job **58492964** runs after all model analysis stages terminate and explicitly reports incomplete stages.

| Source | Actual routed layers | Verification | Collection shards | Primary analysis | Class/temporal supplement |
|---|---:|---|---|---|---|
| gpt | 24 | 58493270 | 58493271, 58493272, 58493273, 58493274 | 58493275 | 58493277 |
| nemotron | 23 | 58492234 | 58492236, 58492238, 58492246, 58492248 | 58492250 | 58492962 |
| glm | 46 | 58492251 | 58492254, 58492255, 58492257, 58492259 | 58492261 | 58492955 |
| qwen36 | 40 | 58492400 | 58492402, 58492407, 58492432, 58492434 | 58492436 | 58492958 |
| qwen330b | 48 | 58492262 | 58492264, 58492266, 58492268, 58492269 | 58492271 | 58492957 |

Collection waves: GPT → Nemotron → GLM → Qwen3.6 → Qwen3. Cumulative predecessor dependencies preserve a maximum of four collection GPU jobs even if an intermediate stage fails. Full collection requires its own verification to pass. Failure notifications request inspection; they do not authorize new budgets.

## Resources and notifications

GPU maximum: **310 of 360 GPU-hours**, leaving **50 unreserved** for bounded recovery. Cancelled-before-start Qwen v2 jobs had zero allocation and remain in the superseded ledger. Existing Qwen label resumptions 58487704/58487708 retain their separate 48 GPU-hour maximum. FAST artifacts are capped at 140 GiB plus four transient allocations of at most 15 GiB (200 GiB overall envelope).

GPT verification 58492176 failed on a 33,024-token input because native eager attention attempted a 129.49 GiB allocation. Repaired verification 58493270 completed successfully in debug QoS (8m59s), after CPU preparation 58493269 passed. Long jobs use normal. Dependency-blocked CPU jobs 58487962 and 58488161 now show normal QoS. All new jobs have Slurm BEGIN/END/FAIL email and bounded 24-hour completion watchers. Slurm dependencies continue without this session; watcher delivery depends on the active terminal/session and expires after its bound.

## Preparation and verification

GPT, GLM, Nemotron and Qwen3.6 full-corpus preparations completed successfully. Qwen3.6 preparation 58492396 completed in 8m25s and bound all 1,354 completions / 15,445,926 completion tokens with a maximum 33,490-token replay input. Qwen3 preparation 58492193 waits for the already-submitted missing-label resumptions and final label analysis.

Two preparation bugs were repaired without changing source labels: compact batch annotations inherit omitted selection metadata from their content-bound trace, and explicit legacy annotations select the content-identical source copy whose recorded sampling selection matches. Label indices, text, offsets and source checksums remain validated. Six production regression tests and three new supplemental analysis tests passed; ten native-capture/production tests passed before these metadata repairs.

Frozen capture bundles: v2 4e462bb5cf051ba9c3f215f04243295367509f55c075078356fe4ce71eb9a28d; Qwen v3 41bdc8ac8fdd4cec322f2d392b0ca9abf3d8e3bf73d5fd6ece9b2947c72d700b. Reporting bundle e0cb0a849d5c1c679ebf3c306ee241d38831b8f8ac557923feddd7f74aa3b41e.

## Outputs and remaining coverage

Primary CPU analyses compute all legacy metrics at every actual routed layer, class/position views, native routing dynamics, held-out weak-boundary fractions, question-grouped accuracy correlations and expert/class associations. The supplement completes corrected/shuffled overlap correlations, class token occupancy, second-order transitions, bigrams/trigrams, censor-aware exit hazards, and endpoint-conditioned class routing contrasts. Unknown labels are excluded from both contrast arms. Original generations and labeling checkpoints remain unchanged.

Final automatic outputs: report/measurement-study/production-results/report.md, results.json, significant-adjusted.jsonl and standalone association plots when supported results exist. Six-benchmark summaries use equal benchmark weights and 1,000 question-bootstrap replicates; incomplete benchmark coverage stays unavailable. AIME24 and AIME25 remain included.

Gemma native NVFP4 checkpoint transfer remains blocked by unverified data-mover host trust; no host-key checks were weakened. Earlier Qwen GPTQ/GGUF native checkpoints/backends and missing exact token IDs remain unresolved. These models are explicitly unavailable in the automatic coverage report.

Full-corpus routing correlations are not yet available. Prior results remain in glm-class-results.md and recovered-routing-associations.md. GLM label analysis covered 1,390 completions and 100,000 sentence records (99,999 canonical, one unknown) with no rejected/unmatched records.

Job provenance and exact dependencies: results/measurement-study/production-submissions.json. Previous detailed status is archived in STATUS-before-production-20260924.md.

## GPT attention memory recovery

Notification 58492954 was acknowledged. The supplement and all original GPT collection shards were cancelled before allocation after verification 58492176 failed (4m31s, approximately 0.1506 actual GPU-hours). The budget conservatively retains its full 1 GPU-hour reservation. No completed GPT attempt artifact existed; failed load diagnostics are archived in results/measurement-study/failed-gpt-58492176.

The process-local attention wrapper computes native GPT eager attention in 128-query blocks, retaining all keys/values, causal/sliding masks and attention sinks. Installed libraries, model precision and routing code remain unchanged. Two clean offline native-equivalence tests passed. The GPU verification additionally requires identical native selected expert sets and checks native weights, router signals and hidden states on a short saved trace before attempting the longest input. GPU equivalence and the longest input in shard 0 passed in job 58493270.

Frozen capture v4: a731ede18a2abf0768f5aad084550e08c999e5229d1ffaac87b9a789f892b575. Replacement jobs: preparation 58493269; debug verification 58493270; collection 58493271–58493274; primary analysis 58493275; supplement 58493277. Nemotron verification 58492234 and final report 58492964 dependencies were updated and read back; the temporary hold on Nemotron was released. All replacement jobs have email plus bounded 24-hour watchers.

## GPT GPU recovery verified

Job 58493270 completed with exit 0:0 in 8m59s. On a 257-completion-token saved trace, selected expert sets were identical and router/hidden relative L2 differences were both exactly 0.0; native weights passed the configured rounding check. Full native capture and reduction also passed on a 32,768-completion-token trace (32,958 total input tokens, 268.1 seconds). These are two complete reusable attempt artifacts spanning all 24 routed layers; their four saved-file SHA-256 checks passed. The manifest-wide maximum is 33,024 input tokens; the tested long trace was the maximum in shard 0.

The four full collection jobs 58493271–58493274 are pending allocation after their verification prerequisite passed. This success verifies the execution repair, not final corpus correlations.

## First full GPT shard completed

Job 58493271 completed with exit 0:0 in 2h39m14s. Shard 0 contains all 395 planned completions, 1,535,025 completion tokens and 910,745,971 compressed artifact bytes. Manifest/record coverage and all 790 file sizes were verified. Jobs 58493272–58493274 are still running; primary analysis 58493275 and class supplement 58493277 remain dependent on complete four-shard coverage. This is completed data collection for one shard, not the final correlation result. Snapshot: gpt-shard0-completion.json.
