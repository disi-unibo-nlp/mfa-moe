# Mechanism start reader recovery v2

The exact CPU frame/price job 59267739 completed with exit code `0:0` and
sealed 454 native starts. The Qwen3.8 audit has 908 reader assignments,
1,024 maximum generated tokens each and a complete-stage conservative price
of 15.8939078511 GPU-hours. The original exact price seal is
`9ea35133da6022e5a5dda261bb690b05dfa6275b7316b4379ca7b98731bd7a87`.
The versioned v2 price seal is
`497b2e8f57646826c3e27d3876b512c93f8ebaf14ef85dc64997a6b84f0ce4d6`.
It binds the unchanged frame and exact token count to the new recovery driver.

The v1 GPU reader driver remains untouched and must not be launched. Its
per-UID receipts preceded a batched `model.chat` call, so a crash after a
receipt but before a batch result required manual adjudication. The v2 driver
writes an append-only batch attempt record and per-UID receipt before each
call, then atomically commits one reader-batch result. On same-manifest resume,
it verifies every committed batch and its receipts. An uncommitted attempt is
preserved and a new attempt with the same UID/reader seeds is recorded; only
the committed result enters the rating summary. It reports the number of
uncommitted attempts. A repeated call can consume GPU time, so recovery
requires a fresh remaining-cost check. Exact physical inference once-only is
impossible to infer after a crash within a batched call and is not claimed.

Launch the versioned
`scripts/experimental_resume/rate_mechanism_start_readers_v2.sbatch` only in
a fresh `dense-mechanism/ratings-v2-*` directory. The driver SHA-256 is
`07f6e30396cbc40c7f7486516b207765b198c4aa23eddae2570d7be4be441765`;
the v2 price file's on-disk SHA-256 is
`4189b783b3a717cabeb3a9dc2920f19306687f63ccc79e29e23cc493cf1b37b6`.
Pass them as `RATING_DRIVER_SHA` and `RATING_PRICE_FILE_SHA`, respectively,
along with `RATING_OUT`. The batch script requires all three and verifies
them before model load.

Validation: five focused selection/recovery/price tests passed; the v2
driver compiled; the batch script passed `bash -n` and the live scheduler's
`sbatch --test-only`. The script requests two A100s for ten hours and has not
been submitted as part of this audit preparation.
