# GLM discovery execution review

Read-only Sol review, 2026-10-08 at 04:42:58 UTC (06:42:58 Europe/Brussels, CEST). The apparent stall in job **59645901** was resolved by two newly committed original discovery receipts. The snapshot contains **39/48** GLM assignments, including **11/20** assigned to this job. There is no evidence of a persistent deadlock across the reported warning bursts; the existing allocation should continue.

## Committed progress

The job's immutable card selects original discovery indices 21–40, each with the original seed and cap 16,384, full native routing capture, and no intervention. Its wall allocation is 23:22:00. These are original assignments, not recovery draws.

| Original index | Attempt-0 interval, UTC | Commit, CEST | Output / finish | Decode including capture hooks | Capture closeout RPC |
| --- | --- | --- | --- | --- | --- |
| 30 | 02:53:55.905–04:11:04.550 | 06:11:04.550 | 16,384 tokens / length | 4,281.99 s | 327.85 s |
| 31 | 04:11:04.558–04:36:22.884 | 06:36:22.884 | 5,664 tokens / stop | 1,481.42 s | 28.84 s |

Index 32 then began attempt 0 at 04:36:22.890 UTC and remained STARTED at the review snapshot. Its UID is `017a256383b484098c6181e755d6744d8cc6ef4147bee7fd865b0c994e5ec26b`. This establishes sequential progress after the previously observed count of 37; it does not establish completion of index 32 or of the full roster.

Both new receipt seals, their original manifest assignments and assignment identities, model/runtime/executor bindings, public question/family/discovery-population joins, COMMITTED attempt-to-receipt joins, and capture archive file hashes pass. Both retain the empty-prefix digest and original prompt-token digest. No tokenizer or inference was rerun. Capture archives contain 27,520,606 and 9,528,740 bytes respectively.

## Warning interpretation

The stdout warning times are local CEST: 03:59–04:03 and 06:06–06:10 correspond to 01:59–02:03 and 04:06–04:10 UTC. Both bursts were followed by committed progress. The latter overlaps the approximately 5.5-minute closeout of index 30 before its 04:11 UTC commit. This timing supports a slow full-capture/RPC explanation; the log alone does not identify which worker operation caused each wait.

The installed `shm_broadcast.py` read-side loop emits this INFO message while waiting for a readable broadcast block, then continues waiting. Sixty seconds is a warning interval, not an exception or a deadlock determination. Native [Backend.generate](../../../src/moe_exp/verification_study/backend.py) calls `verification_end` after generation. [NativeCapture.finish](../../../src/moe_exp/verification_study/adapters.py) concatenates captured tensors, copies them to CPU, validates them, converts them to lists and hashes the capture. This can take substantial time for a full-cap, all-layer request, as the receipt directly measures.

The native RPC calls supply no timeout; the installed executor consequently waits for responses without an RPC deadline. The allocation deadline is checked between engine steps and before requests, and cannot interrupt an individual blocked step/RPC. Slurm's finite allocation remains the hard bound. This is a monitoring limitation, not evidence that the current execution is stuck. Bounded stderr/stdout inspection found no traceback. The observed cumulative CPU and write counters also increased, although those counters alone would not prove current GPU progress.

## Binding evidence

All 35 frozen scientific source hashes pass. The exact relevant seals are:

- Campaign: `57df4a026a7be4c909ece85ac1ebe3dc47c3f27bb53ffcab88e3aa9d82bfbb3b`.
- Job card: `b00a97fdd41d50c006bd7c84e9c081e5eee293f5691448b619aef17512c5d94e`; submission: `a616f7d7542dd72ac237a0cd343190d5854035dbbdd93ef2895e3ee700b22f90`.
- Original GLM manifest: `3c38af069b0b1a92206ed663a6dfe3b77988d9e9ad5046fb765416e4bbc6a18a`; public inputs: `981c4f084b0ab32dacab1eb8012a128078a12a7ed47576f227614997205d952f`.
- Index 30 receipt: `a1492fc84c9136353aeba2a4d898070aaa5316b354541857fc1382630637a1b1`; archive: `703db21f02e1b361f3063c933137d58b2a44a6e249fe66f3e73e0c9a58d5668f`.
- Index 31 receipt: `209e9cce0642bf502ec7f38e9b95cbe7f908a1f89b02b41bb0f3fd3ddbee56e5`; archive: `39ac9e81a31a29a97be7f01fb6ab822a9035273e432012824f1050c8568d5a0a`.
- Frozen backend source: `f09a776717eabb7b4a7dfb1a0b8faefa67357662521fbcc671d0899e80a62d65`; adapters: `ae141fc747473f2f094513bc455d9e0f7933175a32fbdf3ed84375f27479e5b1`.
- Installed vLLM `multiproc_executor.py`: `3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a`; `shm_broadcast.py`: `f3e7a3f3ff713fccbd535e0d41650ee05bdf8677e83d4b90519910645022c30b`.

The safe action is ordinary bounded monitoring of receipt/attempt changes, accounting and log tails on the existing job. A quiet interval or this warning alone does not justify cancellation or another allocation. Preserve the full 48-receipt/archive and successful terminal-accounting gate before constructing the GLM audit. No jobs, scientific files, operational sources, configurations or runtime settings were changed by this review.
