# M9 matched-prefix replay qualification, 2026-10-01

The user authorized the named paper workloads and necessary GPU hours in the
current conversation. This is an engineering qualification of four frozen M9
closure branches, not a semantic result. The four fixtures are selected by the
fixed digest order from the sealed 400-assignment M9 manifest. The driver checks
the literal closure tokens and 16,384-token accounting, frozen source hashes,
exact CPU prefix-presence logits, GPU prompt-token identity, mixed-batch versus
isolated output identity, and a large-dose activation control.

Input manifest: `steering-v1/runs/m9-resume-v1/MANIFEST.json`, SHA-256 seal
`63772b263e855692ecc5a37b10f634f96b30fa51e7e54bf22f5026f16e43024b`.
Driver SHA-256:
`b56cf98850077c0c6d7b3f2cc628d40335b31e552804d9ad74378c04adf99033`.
Output directory: `steering-v1/runs/m9-resume-v1/replay-63772b26-b56cf988`.

One 25-minute, two-A100 Slurm allocation costs at most 0.83334 GPU-hour by
allocated time. It is charged to M9's generation/replay line. Production remains
held until the replay artifact passes, the complete generation and blind-grading
stage price is sealed, and the shared J1 grading reserve is reconciled. If the
qualification fails, its output remains a failure record and is not relabeled as
success. The qualification neither establishes accuracy nor validates a reasoning
trajectory.

First attempt job `59116442` failed before model load in 3m21s (0.11167
two-GPU hours): its launcher selected the later `f2ded395` sampler tree while
the M9 manifest is bound to `9a61e32f`. The failure and empty result directory
are preserved. A retry uses the verified `s1-9a61e32f48c04c24` snapshot,
whose tree digest equals the manifest, and a new output directory ending in
`-tree9a61`. The corrected launcher SHA-256 is
`e3b2dd9b6a4af617cd657ccd4ed830d17b033c0ffa4331094941eb74cc421a32`.
The corrected retry was submitted as job `59117029` and verified pending on
`boost_usr_prod` with two A100 GPUs and the same 25-minute limit.
