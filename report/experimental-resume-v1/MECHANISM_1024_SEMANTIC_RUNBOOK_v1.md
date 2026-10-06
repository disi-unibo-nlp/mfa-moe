# Mechanism 1,024-token blinded semantic measurement v1

This measurement begins only after the generation runner has sealed
`STAGE_COMPLETION.json`. The generation manifest and price must both be sealed,
and the generation price must read `PASS_COMPLETE_STAGE`. A missing shard,
assignment receipt, batch result, routed array, exact assigned UID, or stage
completion stops frame construction before any blind input is written.

The [CPU frame builder](../../scripts/experimental_resume/build_mechanism_blind_frame_v1.py)
takes `--manifest`, `--generation-price`, `--run-out` (generation stage root)
and `--out` (a new private measurement directory). Run it in a CPU Slurm step
with the local Qwen3.6 tokenizer available offline. The working
`/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129` Python 3.11
environment supplies the tokenizer; use `module load python/3.11.7` before
its Python executable. `ARM_MAP.json` keeps all
assigned UIDs, arm identities, execution positions, generation errors, route
presence, emitted and reasoning token counts, first reasoning closure, 1,024
caps, and action dose. `BLIND_FRAME.json` contains only an opaque blind ID and
the allowlisted transition, problem, complete native prefix, triggering
sentence and decoded preclosure continuation. The builder uses the source
frame's sealed prompt/prefix token hashes to bind native reader text to the
generation prefix. The model-world question key is an ID, not reader text.

The [CPU price](../../scripts/experimental_resume/price_mechanism_semantics_v1.py)
uses `--frame`, `--out`, `--max-wall-seconds` and `--gpu-hour-ceiling`. Run it
in an `lrd_all_viz` CPU Slurm step after checking the live account,
partition, QoS and wall limit. The script checks the Slurm job, step and
partition rather than the hostname. Use the same working Python 3.11
environment and offline Qwen3.8 tokenizer. It tokenizes the exact Qwen3.8
chat message for every gradeable row,
prices two 1,024-token reader caps per row against the sealed prior measured
throughput, and packs complete eight-row/two-reader blocks into walltime-safe
shards. It includes cold loads, shutdowns, repeat work and a recovery load.
Only `PASS_COMPLETE_STAGE` permits rating. The price records exact prompt
length per row, and the GPU driver checks each actual prompt length against it.

The unsubmitted [dependent CPU wrapper](../../scripts/experimental_resume/build_price_mechanism_semantics_v1.sbatch)
runs frame construction then exact pricing after the generation stage seal. It
uses `lrd_all_viz`, the working Python 3.11 venv, the manifest SHA in its
private output directory, and a 7,200-second reader job walltime matching the
GPU rating wrapper. Its 100 GPU-hour comparison ceiling is above the 41.69
GPU-hour conservative all-760-gradeable, maximum-valid-context calculation
with the frozen prior throughput; it is a sanity gate for the price, not an
allocation or claim that 100 GPU-hours will be spent. The exact stage estimate
is printed and sealed for review before any rating job is launched.

The [GPU driver](../../scripts/experimental_resume/rate_mechanism_semantics_v1.py)
and [Slurm wrapper](../../scripts/experimental_resume/rate_mechanism_semantics_v1.sbatch)
read only the blind frame and price. Supply their paths through
`MECHANISM_BLIND_FRAME`, `MECHANISM_READER_PRICE`, and
`MECHANISM_READER_OUT`; use the priced shard index as `SLURM_ARRAY_TASK_ID`.
Verify live resources and the sealed price before submitting any job. The
wrapper requests two GPUs for two hours and writes account-private runtime
caches. Each blind ID receives two independently seeded Qwen3.8 reads with
the frozen rubric and 1,024-token response cap. Each reader batch records an
append-only attempt before inference and one committed result afterward.
Uncommitted attempts may have consumed compute. Review remaining cost before
setting `MECHANISM_RESUME_UNCOMMITTED=1` for a same-bound recovery; old attempts
remain in the ledger. No missing row is replaced.

Before GPU submission, run the read-only [rating launch gate](../../scripts/experimental_resume/gate_mechanism_semantic_rating_v1.py)
with the sealed `--generation-out`, `--semantic-out` and a new manifest-bound
`--rating-out` directory. The gate verifies complete generation, every mapped
assignment, blind IDs, the exact two-reader price, prior measured throughput,
and the 7,200-second/two-GPU profile, then prints a reviewable `sbatch --array`
command. It does not submit. The GPU wrapper obtains its Slurm end time and
passes a deadline to the reader driver; pending work checkpoints before a new
model load or reader batch when the remaining time is too short.

After every priced shard has a complete `SUMMARY.json`, run the rating driver
with `--seal-stage` and the same frame, price and output root. This checks
every committed result and attempt receipt and writes `STAGE_SUMMARY.json`.
Only then run the [CPU ITT analysis](../../scripts/experimental_resume/analyze_mechanism_semantics_v1.py)
with `--manifest`, `--arm-map`, `--frame`, `--price`, `--ratings-out`, and
`--out`. It verifies the complete reader stage before joining arms. The
primary observed semantic outcome requires both readers to produce a valid
stopped positive vote. All assigned errors, zero reasoning, invalid votes,
early closures and 1,024-token caps stay in the denominator. Report the
individual votes, reader coverage, disagreement, status, closure, cap and
emitted-token summaries.

The frozen family-clustered ITT contrasts are target +1 versus native, target
+1 versus matched random +1, native duplicate versus native (sampling noise),
and target +1 versus the average of both native draws (precision sensitivity).
They are reported pooled and by the two supported starting transitions,
with 5,000 joint parent-family bootstrap draws and Bonferroni intervals across
the twelve contrasts. Execution-position rates and paired target-native
differences by target position are descriptive sensitivity checks. The
enrolled 78-family / 95-start manifest makes this an exploratory accepted
family subset, not the registered 128-family validation. The start audit had
122 positive candidate ratings before the first accepted event per family and
transition was selected. The two readers are
LLM audits, not human labels; no second action or action-order claim follows.

No job is submitted by these scripts or by this runbook.

If the frozen generation runner exits 3 before its first batch with no Python
traceback, the separate [diagnostic entry](../../scripts/experimental_resume/diagnose_mechanism_validation_v1.py)
and [same-manifest shard-0 Slurm wrapper](../../scripts/experimental_resume/diagnose_mechanism_validation_v1.sbatch)
print the active exception before the qualified shutdown helper exits. The
entry SHA-256 is
`e6813abd82b53ae6c0f4365b7657d038c855aa3ebe8d9f23dfb7c79837c4a695`.
The wrapper does not edit the generation runner, base driver, overlay, worker
tree, manifest or existing results. It uses the same shard-0 output directory;
if startup succeeds, normal same-manifest generation can continue. Review
the traceback and resource budget before submitting a recovery job.

The traceback exposed `worker import resolves outside the qualified overlay`
before model construction. The versioned [v2 import-recovery entry](../../scripts/experimental_resume/diagnose_mechanism_validation_v2.py)
pins `moe_exp` and verifies the frozen worker module path before importing the
runner, world or action packages. Its SHA-256 is
`a06b81c4a03184be6e13bfc6984e04a1854873efe95e2d8af4e634f21573149e`,
pinned by the [v2 Slurm wrapper](../../scripts/experimental_resume/diagnose_mechanism_validation_v2.sbatch).
The wrapper accepts `SHARD_INDEX=0` or `1`, checks the priced shard, and uses
the same manifest and shard output binding. A fresh-process CPU test verifies
that importing the runner, world and action modules leaves the worker package
and worker module spec in the qualified overlay. No generation source hash is
modified.
