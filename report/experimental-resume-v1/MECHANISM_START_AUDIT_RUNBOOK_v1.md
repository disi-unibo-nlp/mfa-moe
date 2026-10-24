# Mechanism start audit v1 (prepared, not submitted)

The frozen 128-family mechanism pool and v2 detector audit yield 454 proposed
starts under a preoutcome hash rule: 368 candidate-to-verification starts from
126 families and 86 approach-to-commit starts from 52 families. Each family
remains in the frozen pool, including those with no supported start. At most
three events per family and supported transition are retained. Native prefixes
past 8,192 tokens are ineligible. Failed-check-to-revise has only five families
within this cap and is unsupported; no replacement search is performed.

The selection is [MECHANISM_START_SELECTION_v1.json](MECHANISM_START_SELECTION_v1.json)
with internal seal `c9859eedced42a41e37577b7ebe64f2a1fd5d24cb200ebcd9fc6b9db4c99d721`.
It is derived solely from sealed native units and the frozen detector audit.
The exact-prefix CPU job reconstructs selected prefixes from source trace
byte offsets and checks trace digests, token alignment, contiguous successor,
tokenizer identity and reasoning closure. Its arm-blind frame exposes only the
original problem, emitted prefix and triggering sentence to the readers.

Run order after normal account and queue checks:

1. Submit `scripts/experimental_resume/prepare_mechanism_start_audit_v1.sbatch`
   with `PREP_DRIVER_SHA=72687bc1beba1602c9414f3fb9677481c61973beb12ee933420d8e171ef33b3e`
   and `PRICE_DRIVER_SHA=e7ec73a5eeb3b357f86ce9fde276a0f39d83d2bc6f958a45eca5fc2bc42c6be9`.
   It uses one serial CPU, 16 GiB and a 30-minute walltime. The two `srun`
   steps write the exact native frame and a sealed complete-stage Qwen3.8
   price. Verify Slurm state, exit code and both artifacts before GPU work.
2. Inspect `MECHANISM_START_READER_PRICE_v1.json`. The GPU driver requires
   `PASS_COMPLETE_20_GPUH`, binding the frame, code, rubric, exact chat-template
   prompt count, 908 maximum 1,024-token reader outputs, conservative prior
   throughput, two cold loads, and two shutdowns. If the stage does not fit,
   leave it unsubmitted and revise the resource plan. The price is **not yet
   available**; no exact GPU-hour estimate is claimed before the CPU step.
3. Set `RATING_OUT` under the private `dense-mechanism/ratings-v1-*` directory
   and pass `RATING_DRIVER_SHA=aa8f99d2b6f232069bdfd8f880c16cfd1a3cfa0c2c4c6507286b2c968b89ba89`
   and `RATING_PRICE_FILE_SHA` equal to the exact on-disk SHA-256 of the sealed
   price. Submit `scripts/experimental_resume/rate_mechanism_start_readers_v1.sbatch`
   only after the price gate. It requests two A100 GPUs, 16 CPUs, 120 GiB
   and 10 hours. Its resumable output includes assignment receipts, reader
   batches, a binding and summary. An interrupted attempted batch requires
   adjudication before retry, avoiding silent duplicate or skipped ratings.

The two readers are different seeded draws of the same Qwen3.8-27B model.
They are an LLM audit of the start condition, not human truth. The later
native-generator veto and 1,024-token intervention generation are separate
stages; this audit alone is not a steering result.

Validation on 2026-10-03: three focused tests passed; all new Python files
compiled; both batch scripts passed `bash -n` and `sbatch --test-only` against
the live `iscrc_miosr` association. `sbatch --test-only` gave predicted IDs
59267699 and 59267700; these were **not submitted jobs**. Live Booster
partition limit was one day, so the proposed ten-hour GPU shape is within
the partition wall limit, subject to the exact price gate and later queue
checks.
