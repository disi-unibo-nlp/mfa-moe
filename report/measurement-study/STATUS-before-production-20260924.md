# Measurement campaign status

Verified 2026-09-23 after resumed labeling and dependent CPU analysis submissions.

| Work | Observed status |
|---|---|
| GPT-OSS native MXFP4, 24 routed layers | 58487262 completed; two short and one long capture passed at all 24 layers |
| GLM native BF16, 46 routed layers | 58487265 completed; short/long capture and native biased-gap checks passed; labels finishing |
| Qwen3-30B native BF16, 48 routed layers | 58487288 completed; short/long capture passed at all 48 layers; two missing label shards queued to resume |
| Qwen3.6 native FP8, 40 routed layers | 58472802 completed; two short and one 13,287-token capture passed |
| Nemotron native NVFP4, 23 routed layers | 58474338 completed; native NVFP4 capture passed at all 23 layers on two short traces and a 16,085-token trace; BF16 KV cache |
| Gemma native NVFP4, 30 routed layers (topology reference) | Pinned 18.8 GB transfer manifest prepared; strict data-mover SSH host trust unresolved; custom native capture implementation awaiting GPU validation |
| Earlier Qwen3.5 GPTQ and GGUF | Native checkpoints absent in inspected HF cache. The full join found token ID arrays in 114/120 GPTQ records and 0/10 GGUF records. Original checkpoint/backend and missing-token recovery remain unresolved |
| Existing CPU routing analysis | All 15 bundles complete after recovery 58471939; four recovered bundles have zero rejected rows |
| Full labeled inventory | Superseded by completed content-bound source join: 15 cohorts and 6,117 saved trace versions |
| Content-bound source join | 58475151 completed the corrected join of all 15 cohorts; 6,117 saved trace versions, 6,101 with token ID arrays; zero unmatched/rejected matches |
| Lossless storage sizing | 58474659 completed successfully; all 30 compressed artifacts passed lossless round-trip verification; zero GPUs |
| Existing GLM/Qwen3 labeling | GLM parts 0 and 2 complete, parts 1 and 3 running. Qwen3 parts 2 and 3 complete; parts 0 and 1 resumes submitted as 58487704/58487708 after CPU preflight 58487662. 13,376 saved records will be reused |

The source join distinguishes content hashes within each model and preserves overlapping label cohorts. The earlier logical-identity count is provisional. It must not be treated as the final number of independent completions.

## Verified capture and limits

Qwen3.6 captured 17,087 exact completion tokens across three traces. Both tensor-parallel ranks returned identical native signal hashes, every trace covered all 40 layers, and all 19,400 temporal windows had native selection gaps. Weak-boundary fractions are still pending disjoint calibration thresholds. These are engineering measurements, not new accuracy correlations.

Eleven distinct local tests passed across two runs: ten contract/native-capture/GLM tests, followed by four native-capture tests including one new Gemma case. A native GPU run is still required for Gemma.

| Model | Load seconds | Extraction/capture seconds | Completion tokens | Uncompressed artifact bytes |
|---|---:|---:|---:|---:|
| GPT-OSS | 145.4 | 31.6 | 1,299 | 5,226,058 |
| GLM | 330.6 | 30.0 | 1,399 | 7,727,060 |
| Qwen3-30B | 303.8 | 39.5 | 3,859 | 45,383,139 |
| Qwen3.6 | 522.9 | 35.7 | 17,087 | 137,409,754 |

The Qwen3.6 capture time excludes CPU feature reduction; its post-load replay/reduction/output phase took about 136 seconds according to the log. The other three timings include their extraction reducers. These are different timing scopes and must not be compared as throughput directly. Long-trace and lossless storage sizing are required before a production estimate.

## Resource and execution state

The original validation cap is fully reserved: eight jobs, each at most two GPUs for 30 minutes. All eight validation jobs are finished; actual elapsed GPU usage was 2.253 GPU-hours including failures. The cap is based on maximum reservations, not recycling unused time into extra submissions.

The user authorized the three-GPU-hour long-trace extension by requesting the missing work be resumed and queued. GPT-OSS 58487262, GLM 58487265 and Qwen3-30B 58487288 all completed with native_capture_and_long_trace_passed and zero recorded failures. Qwen3 used normal QoS with the same 30-minute cap because debug QoS permits only two submitted/running jobs.

The two failed Qwen3 labeling shards are queued as 58487704 and 58487708, each preserving the original two-GPU/12-hour cap. CPU preflight 58487662 verifies all source/checkpoint identities, sizes remaining judge prompts and copies existing checkpoints to isolated WORK outputs. Judge prompt, completion cap and sampling remain unchanged; server context capacity can increase to fit verified prompts. Original scratch outputs and other labeling jobs are unchanged. Combined maximum resumed-label reservation: 48 GPU-hours, separate from capture validation.

Dependent CPU analyses are queued: GLM 58487905 and Qwen3 58487962, each zero GPUs, 8 CPUs, 32 GiB and 30 minutes in debug QoS. They verify all 100,000 records, export complete annotations and compute class occupancies, transitions, motifs, runs and correlations after labeling succeeds. They do not replace missing all-layer GPU routing collection. All new jobs have email notifications and 24-hour bounded watchers.

No full-corpus GPU collection has been submitted. Its measured compute/storage proposal is pending long-trace acceptance and source/storage inventory results. Preserve all existing labels, generations, pilot jobs and artifacts.

AIME remains included in per-benchmark reporting and the equally weighted six-benchmark summary. Sparse outcome support is reported per association. GPQA data recovered from a legacy routing bundle remain an extra benchmark and are excluded from the six-math-benchmark aggregate.

## Lossless storage measurements (58474659)

All source artifacts were preserved. Packed sizes: GPT-OSS 868,578 bytes; GLM 1,586,063 bytes; Qwen3-30B 7,926,555 bytes; Qwen3.6 41,670,916 bytes. Each packed artifact was decompressed and SHA-256 compared with its source. Qwen3.6 falls from 137.4 MB to 41.7 MB on the three-trace fixture; full-corpus estimates remain contingent on the content-bound inventory.

Source-join recovery 58475151 uses frozen code c41373747e7f9f28, 8 CPUs, 32 GiB, zero GPUs, 30 minutes in debug QoS. It is continuation of the already-authorized full-label CPU processing. Five bounded identity checks passed: nonzero legacy sample, duplicate source location, explicit sample mismatch, ambiguous legacy sample rejection and model isolation. Output: results/measurement-study/source-join-v2; completed source-join cohorts remain unchanged.

Completed source-join coverage and exact limitations are in collection-coverage.md. The provisional 6,110 logical-ID inventory is superseded for collection planning by 6,117 content-bound trace versions.

Nemotron verification: 16,866 completion tokens over three traces, all 23 routed layers, identical native routing hashes across TP ranks, and 11,362 reasoning windows with native sigmoid-plus-bias boundary gaps. Load: 276.6 s; capture: 20.7 s excluding CPU reduction; post-load replay/reduction/output about 104 s. Uncompressed outputs: 89,383,164 bytes. Weak-boundary calibration and full corpus metrics remain outstanding.

Live label recheck: approximately 88,360/100,000 GLM sentence records and 55,808/100,000 Qwen3 sentence records persisted; these are changing checkpoint counts, not final quality-audited label totals. Two Qwen3 shards stopped on HTTP 400; exact cause remains unconfirmed because the logs omit the response body. Five active label jobs have bounded watchers armed. See labeling-live-status.json for job IDs, timestamp and watcher sessions.


Long-trace results: GPT-OSS 10,492 tokens, load 144.7 s, extraction/reduction 82.1 s, 71,262,539 uncompressed bytes; GLM 12,075 tokens, load 359.4 s, extraction/reduction 83.6 s, 217,673,894 bytes; Qwen3 15,001 tokens, load 310.9 s, extraction/reduction 98.3 s, 208,607,066 bytes. Each covered three traces and every routed layer. Full-corpus compression sizing and the production resource proposal must use these measurements, not just the earlier short fixtures.


## Latest queue and completed labels

All four GLM labeling shards completed. The dependent analysis 58487905 also completed (157 seconds): 100,000 records across 1,390 completions, 99,999 canonical labels and one explicitly unknown record, no unmatched/rejected traces or conflicting labels. Qwen3 parts 2/3 completed; part 2 has one unknown record. Final Qwen3 analysis 58487962 waits for resumed shards 58487704/58487708. Final content-bound source join 58488161 waits for the Qwen3 analysis; GLM results are already available. All these submissions have bounded watchers armed.

The CPU preflight has verified Qwen3 part 0 and copied 6,720 exact checkpoint records. The failing next batch contains item 6,771 with a 35,766-token prompt. Adding the unchanged 16,384-token output allowance gives 52,150 tokens, exceeding the former 49,152 server limit. The resumed server uses 57,344 context capacity. No prompt or output allowance is truncated. Part 1 is also verified: 6,656 saved records copied, oversized item 6,660 has a 35,769-token prompt, and the same 57,344 context capacity fits all remaining inputs. Both preflight outputs are ready.

Collection sizing 58488054 completed with all lossless checks: projected packed output for the five validated models is 98,481,253,933 bytes, before Gemma and older checkpoints. WORK quota reports 939G used of 1T. A resource draft totals 360 GPU-hours including repeated model loads and a 1.5 timing allowance. It is not an executable production allocation: full-run code/shards, storage, remaining native checkpoints and final resource approval are outstanding. Permission was requested for up to 200 GiB in the user's FAST subtree, as required by the user-provided AGENTS.md rule for additional write locations.
