# Content-bound collection coverage

Verified from completed CPU resume 58475151 (exit 0:0). All 15 annotation cohorts joined with zero unmatched annotations, rejected matches, label conflicts or within-cohort metadata conflicts. This verifies source matching; annotation-span and native-backend acceptance are separate gates.

The inventory contains 6,117 distinct saved trace versions and 44,212,916 completion tokens. 6,101 records contain prompt and completion token ID arrays. Their presence does not independently prove the historical generation checkpoint/backend.

| Model | Trace versions | With token ID arrays | Completion tokens | Labeled benchmarks |
|---|---:|---:|---:|---|
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 120 | 114 | 712,630 | aime24, math500 |
| Qwen/Qwen3.6-35B-A3B-FP8 | 1354 | 1354 | 15,445,926 | aime24, aime25, amc23, math500, minerva, olympiad |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 1642 | 1642 | 14,819,371 | aime24, aime25, amc23, math500, minerva, olympiad |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 1411 | 1411 | 6,998,743 | aime24, aime25, amc23, math500, minerva, olympiad |
| openai/gpt-oss-20b | 1580 | 1580 | 6,140,145 | aime24, aime25, amc23, math500, minerva, olympiad |
| qwen3.5-35b-a3b-mtp-ud-q4-k-xl | 10 | 0 | 96,101 | math500 |

Six Qwen3.5 GPTQ logical attempt IDs each map to two different saved texts; both versions remain represented, with cohort/source provenance. No automatic merging across content hashes is allowed. Six GPTQ and ten GGUF traces lack token ID arrays. Existing all-layer capture and original-generation provenance are still required.

GLM and Qwen3-30B labels are being produced by the eight existing jobs; their 2,920 planned attempt memberships remain outside this completed-label snapshot.

The four main currently labeled models cover all six mathematics benchmarks. The two older Qwen3.5 labeled cohorts cover only the benchmark subsets in the table; their broader saved routing cohorts are separate data.

Inventory SHA-256: `eb07af23d96dfdec344f9e0ecb25910b0034ef3421d9e81497874ab992b20fa2`.
