# Dense judge measurement qualification, 2026-10-01

The historical seven-class Qwen3.8-27B program has SHA256
`467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07`
and is present in WORK. Job **59112371** completed discovery-only preparation of
200 balanced sentence fixtures, seal
`8e939beed6a9ce7b8b4243b97495b527fabccc2ef42b80bd4ac7f14b0fc114e1`.
The historical Qwen36 label job **58769981** processed about 25,000 labels in
20,684 seconds on two GPUs. A 200-item direct offline audit, allowing up to
1,024 generated tokens per item, one cold load, prefill, and teardown, is
bounded by a **35-minute two-GPU** allocation. Including the observed
196-second shutdown allowance gives a conservative **1.28 GPU-hour** price;
the versioned measurement-qualification ceiling is **1.35 GPU-hours** under
the user's expanded 2026-10-01 compute authorization.

The direct vLLM chat prompt uses the pinned program instructions and fields
but does not claim DSPy prompt equivalence. It reports seven-class confusion,
coverage and agreement against historical labels as an LLM audit only. This
check does not qualify an online prefix detector or substantive verification.
No discovery, mechanism or utility generation is submitted on the strength
of this measurement check alone.
