# Native discovery route extraction, 2026-10-01

This CPU-only stage reads the 48 frozen Qwen3.6 discovery traces and their
639,018,109 bytes of captured native routing tensors. It preserves exact saved
sentence token ownership and produces, for each sentence and routed layer,
top-k expert selection frequencies and normalized selected-gate mass. It reads
neither class labels nor outcomes. The future matched behavioral contrasts may
use these profiles only after dense labels and semantic eligibility are audited.

Input sentence seal:
`84a85c92b595a7829446b6a53b376e1126df3b6c36eabaddb3b4e884eb52fa72`.
Driver SHA-256:
`40c3d3b0e9978db1b8b16525d22d6b72dcb60b8c9af809d2cb47ad66414999f4`.
Output directory:
`steering-v1/runs/routing-control-v1/dense-discovery/route-profiles-84a85c92-40c3d3b0`.

One two-CPU, 32-GiB, eight-hour Slurm allocation is a maximum of 16 CPU
core-hours. Each family has an atomic NPZ and source-hashed receipt so a
same-code retry can resume completed families. The result is observational;
it cannot establish a causal expert target or an optimal trajectory.
