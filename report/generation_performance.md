Update — 9 September 2026

The correlation pipeline now uses vLLM for both generation and frozen-GEPA
sentence tagging, with eight concurrent requests per stage. The server uses
prefix caching, an 8,192-token batch budget and three-token MTP. Generation
uses Qwen3.5-35B-A3B GPTQ-Int4; tagging uses Qwen3.8-27B NVFP4. GPQA-Diamond
and MMLU-Pro are excluded from default runs.

The one-worker recommendation below describes the earlier llama.cpp experiment,
whose MTP server had only one decoding slot. It is not a recommendation for the
new vLLM server. The measurements below distinguish the short pilot from the
production run; eight requests in flight do not guarantee an eightfold
throughput increase.

A live generation smoke test completed eight simultaneous requests on the
RTX 5090: **8,192 completion tokens in approximately 5.8 seconds** (about
1,410 tokens/s aggregate, excluding startup and dataset loading). All eight
requests reached the deliberate 1,024-token smoke cap; reasoning text was
preserved. The server log confirms eight running requests and successful MTP.
This uses one MATH-500 prompt with eight seeds, so it validates concurrency
and serving compatibility rather than full-benchmark speed or accuracy.
Artifacts: `results/correlation_pipeline/vllm-generation-smoke/` and
`slurm_logs/correlation-vllm-generation-20260909_172039.log`.

A live tagging smoke test completed all **47 units** of one existing MATH-500
trace, using the frozen program and eight client workers. Model calls ran from
approximately 17:27:34 to 17:28:35 UTC (about 61 seconds, excluding server and
Python startup). All final labels were valid and saved in source order under
`results/correlation_pipeline/vllm-tagging-smoke/reasoning-vllm-v1/annotations`.
The server log is `slurm_logs/correlation-vllm-judge-20260909_172518.log`.

The validated default remains `GPU_MEMORY_UTILIZATION=0.90`, with the judge
using eager execution and FP8 KV cache. Its pilot usually ran six requests
while two waited for cache space. Raising utilization to 0.94 or 0.95 caused
out-of-memory failures during NVFP4 autotuning. Disabling autotuning at 0.94
allowed startup but failed during actual inference. These experiments were
isolated; their settings were not applied to `run_all.sh`. Eight client workers
and `max-num-seqs=8` set a concurrency ceiling, not guaranteed simultaneous
decoding at every context length. The short tagging pilot does not provide a
reliable runtime estimate for millions of varied sentences.

**Production observation, 9 September at 22:43 UTC.** The user-started tmux
run is generating new GPTQ responses. MATH-500 completed 500 attempts in 56:01,
producing 3,518,788 completion tokens (about 1,047 tokens/s aggregate). AIME24
had saved 709 of 960 attempts and 12,218,988 completion tokens at the snapshot.
These are observations of this run, not a controlled comparison with the older
GGUF checkpoint. Longer contexts sometimes reduce active decoding below eight
requests because GPU cache space is full. The full command will next perform
tagging, forward replay and analyses. Snapshot:
`results/correlation_pipeline/vllm_progress_20260909_2244.json`.

The previous GGUF generations of the six retained datasets contain
**4,647 saved attempts and 3,890,404 annotation units**, counted directly after fixing pathological regex backtracking
without changing the segmentation boundaries. The count is recorded in
`results/correlation_pipeline/vllm_migration_counts_20260909.json`.

| Dataset | Saved attempts | Annotation units |
|---|---:|---:|
| MATH-500 | 500 | 308,567 |
| AIME24 | 960 | 920,094 |
| AIME25 | 960 | 1,035,763 |
| OlympiadBench | 675 | 616,836 |
| AMC23 | 1,280 | 794,376 |
| Minerva | 272 | 214,768 |

The launch settings follow the official vLLM recipes for
[Qwen3.5 MoE](https://recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B) and
[Qwen3.8 27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

Generation performance check — 8 September 2026

The recommendation on 8 September was to keep the one-worker native-MTP configuration. In a repeated live
comparison, one worker delivered 281.56 completion tokens/second and two
delivered 284.42, a 1.02% difference. This pilot provides no evidence of a
substantial scheduling speedup. The historical workload explains the reported
five-day generation time.

The existing Qwen3.5 native-MTP pipeline already generates one attempt at a
time. `avg@32` means 32 separately seeded attempts per problem, not one request
with 32 simultaneous completions. At that time, `run_all.sh` defaulted to `WORKERS=1`, and the
MTP server launcher enforced `PARALLEL=1`. Reducing completion concurrency from
32 to 2 therefore does not apply to this run. The launcher's `BATCH_SIZE=1024`
controls token processing, independently of request concurrency.

The historical run spent 110.71 hours decoding 109,628,705 completion tokens,
at 275.07 tokens/second. Its measured wall time was 111.28 hours (4.64 days),
at 273.65 completion tokens/second including prompt processing and gaps between
requests. Decoding accounts for 99.48% of that wall time. The long duration is
explained by the volume of generated reasoning and benchmark repeat counts.

**Historical benchmark costs**

| Benchmark | Saved attempts | Samples/problem | Completion tokens | Mean tokens/attempt | Wall hours | Wall tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| GPQA-Diamond | 1,980 | 10 | 35,783,400 | 18,072 | 37.98 | 261.73 |
| AIME25 | 960 | 32 | 19,280,106 | 20,083 | 19.18 | 279.22 |
| AIME24 | 960 | 32 | 17,341,451 | 18,064 | 17.40 | 276.77 |
| AMC23 | 1,280 | 32 | 14,112,053 | 11,025 | 13.78 | 284.57 |
| OlympiadBench | 675 | 1 | 10,706,878 | 15,862 | 10.67 | 278.81 |
| MATH-500 | 500 | 1 | 4,687,590 | 9,375 | 4.45 | 292.34 |
| MMLU-Pro **partial** | 387 | 1 | 4,032,925 | 10,421 | 4.13 | 271.04 |
| Minerva | 272 | 1 | 3,684,302 | 13,545 | 3.68 | 277.83 |

There are 6,627 attempts in the seven completed datasets, plus 387 saved
MMLU-Pro attempts without a completed manifest. The latter explain why this
timing audit includes 7,014 requests while the correlation report uses 6,627.
The MMLU-Pro duration is only the observed partial workload, not an estimate
for the entire benchmark. Dataset spans exclude the tiny transitions between
datasets; the total span includes them. Model loading, the final interrupted
request, and forward extraction are excluded.

All 7,014 logged completion token counts match the saved shards in completion
order. The shard-write timestamps and server completion timestamps differ by
a consistent offset, with a range of 2.60 seconds. This checks the attribution
of server timings to benchmarks rather than estimating their cost from token
counts alone.

**Live comparison protocol**

The pilot uses the same local GGUF, native MTP with six draft tokens, 49,152
context tokens, one server slot, token-processing batch size 1,024, and the
RTX 5090. An isolated server on port 18080 avoids sharing a running experiment.

Each trial executes the same 32 saved prompt/seed pairs, round-robin across
the seven completed benchmarks. Sampling parameters remain temperature 0.6,
top-p 0.95, and top-k 0. Only this timing pilot caps completion length at 1,024
tokens; production generation keeps its 32,768-token limit and all required
samples. The trial order is workers 1, 2, 2, 1, with a short untimed warmup
before each. Throughput is total completion tokens divided by trial wall time.

The measured workload covers the first few saved examples per benchmark and
mostly reaches the pilot token cap. It is a short speed comparison, not an
accuracy evaluation or a full-length runtime forecast. Two client workers
measure queueing on one MTP slot; they do not measure two simultaneous decodes.

**Live results**

| Client workers | Trial 1 tokens/s | Trial 2 tokens/s | Pooled tokens/s | Mean seconds / 32 requests | Mean request latency |
|---|---:|---:|---:|---:|---:|
| 1 | 279.98 | 283.16 | 281.56 | 115.66 | 3.61 s |
| 2 | 284.75 | 284.09 | 284.42 | 114.50 | 7.05 s |

Pooled throughput divides the total tokens in both repeats by their total
wall time. Each trial produced 32,566 completion tokens from all 32 requests;
31 of 32 responses reached the pilot token cap. All 32 response hashes matched
across all four trials. The only changed setting was the number of outstanding
client requests.

The complete pilot ran from 15:47:43 to 15:55:26 UTC, including warmups.
Two workers saved an average of 1.16 seconds per trial and nearly doubled
individual request latency because of queueing. With just two repeats and
short capped outputs, the 1.02% throughput difference does not justify changing
the production default or forecasting a comparable full-run improvement.
The isolated server was stopped after measurement and GPU memory returned to
40 MiB.

**Next action supported by these measurements**

Preserve `WORKERS=1`, the current MTP settings, and each retained benchmark's
required sample count. If the next model's benchmark scope must be reduced,
GPQA-Diamond offers the largest observed saving (37.98 hours in this run).
Minerva is much cheaper (3.68 hours). These are measured costs for this model,
not guaranteed runtimes for a different model. No benchmarks were removed by
this check.

**Reproduction and evidence**

- Historical measurements: [historical.json](../results/generation_performance/20260908/historical.json).
- Pilot settings and trial measurements: [summary.json](../results/generation_performance/20260908/concurrency/summary.json).
- Exact pilot prompts and seeds: [workload.json](../results/generation_performance/20260908/concurrency/workload.json).
- Per-request timings and response hashes: [requests.jsonl](../results/generation_performance/20260908/concurrency/requests.jsonl).
- Pilot server log: [server.log](../results/generation_performance/20260908/server.log).
- Local server image ID: [server-image.txt](../results/generation_performance/20260908/server-image.txt).
- Historical server log: [correlation-mtp-20260903_152848.log](../slurm_logs/correlation-mtp-20260903_152848.log).
- Reusable commands: [pipeline README](../src/moe_exp/correlation_pipeline/README.md#measure-generation-speed).

The performance tools only write diagnostic outputs. Existing generation and
forward artifacts are read, and the production concurrency/sample defaults
are preserved. Focused tests check sample/seed preservation across settings,
the concurrency limit, wall-throughput calculation, and historical alignment.
Validation: `PYTHONPATH=src python3 -m pytest -q tests/test_generation_performance.py`
completed with **3 passed**; `git diff --check` also passed.
