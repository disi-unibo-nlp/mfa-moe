# Routing-control execution status — 2026-10-04

The mechanism validation generation has been recovered and sealed without
rerunning GPUs. Jobs 59272676 and 59272700 left all 190 batch records and
760 routed continuations, then exited 2 because the base runner's shutdown
uses `os._exit(0)` before the outer completion wrapper can run. CPU-only job
59342254 validated every saved batch and array and wrote the manifest-bound
`STAGE_COMPLETION.json` (`b021da04ef9d5df4adc5674711013e2044dd4b7b8f7d764ce3606653a7356bae`).
Counts are 760 assigned, zero errors, 760 routed arrays, 769,161 emitted
tokens, 723 length stops at 1,024, and 37 natural stops. The stage status is
`COMPLETE_UNGRADED_GENERATION`. It enrolls 95 starts from 78 families; the
registered 128-family feasibility gate failed, so this is an exploratory
accepted-family subset.

The blind-frame and exact-reader-price job 59342554 was submitted to
`lrd_all_viz` as a one-hour, two-CPU, 16-GiB job after live account/partition
checks, focused tests and `sbatch --test-only`. It started on `viz09` at
2026-10-04 21:34 CEST and completed `0:0` in 3:42. The frame builder emitted
a sealed arm map and blind frame, with all 760 continuations gradeable. Exact
Qwen3.8 token pricing passed for 1,520 ratings in nine two-GPU shards,
30.592625 projected GPU-hours. The read-only complete-stage rating gate passed.
The two-reader GPU array was submitted as **59342858** at 21:39 CEST with
two A100s, 16 CPUs, 120 GiB and a two-hour walltime per shard. Six shards
began immediately; three were pending placement at last check. Verify each
task's final state, exit code, result summary, and attempt ledger before a
complete-stage seal and ITT analysis. Neither a Slurm start estimate nor a
partial rating is an effect estimate.

CPU postprocessing job **59342938** is pending with Slurm dependency
`afterok:59342858_*`. If and only if all nine array tasks succeed, it will
seal the complete arm-blind reader stage and write the family-clustered ITT
artifact `MECHANISM_1024_SEMANTIC_ITT_v1.json`. A reader shard with an
uncommitted attempt or incomplete assignment prevents the seal; inspect its
ledger and remaining cost before same-bound recovery.

**Correction at 21:45 CEST:** Original array task `59342858_8` failed `1:0`
after four seconds, before creating its shard directory or loading the model.
The v1 wrapper queried `squeue -j "$SLURM_JOB_ID" -o '%e'`, which returned
nine array end times; `date` rejected the multiline input. The versioned
`rate_mechanism_semantics_v2.sbatch` asks for the exact
`SLURM_ARRAY_JOB_ID`/`SLURM_ARRAY_TASK_ID` end time and rejects multiline
answers. Its syntax and single-shard Slurm test passed. Same-input recovery
task **59342983_8** started on `lrdn0633`; the eight original tasks remained
running. Pending CPU job 59342938 was canceled because its all-array
dependency could never pass. Replacement **59343015** depends on successful
original tasks 0–7 and recovery task 8, then runs the same seal and ITT
analysis. The old failure and logs remain in accounting; no reader result was
overwritten or silently reused.

The prior 13-family, 256-token positive pilot's target +1 versus native
immediate-semantic estimate is +0.115 with simultaneous interval
[-0.115, 0.385]. Its target +1 versus matched random +1 estimate is +0.038
[-0.154, 0.308]. These do not establish semantic control. A separate routing
first-stage audit found target +1 changed expert membership in 28.6% of active
approach rows versus 8.4% for matched random +1; this is a routing effect, not
the semantic endpoint. No 16k accuracy/token utility scout has run. The paper
cannot yet claim validated steering or an accuracy/token benefit.

The independent 220-family extension reader qualification job 59273283 failed
before model load because `tokenizer_ids` rejected the chat-template return
container. The v1 qualification has an append-only attempt receipt; versioned
repair and fresh cost review are required before a retry. Do not edit or
overwrite its sealed v1 manifest, receipt, or output root.
