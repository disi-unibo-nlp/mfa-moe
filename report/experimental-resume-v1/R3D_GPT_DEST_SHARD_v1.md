# Isolated R3-D GPT destination shard, prepared 2026-10-02

The frozen canonical R3-D driver fits 900 GPT checkpoints followed by 750
Qwen checkpoints. At 10:40 CEST, canonical job `59111360` was running on
four CPUs with 398/900 GPT checkpoints; it was fitting `switch` and had no
`dest` or `dest-contained` checkpoint. Its current 24-hour slice ends at
17:38 CEST. The independent Qwen sidecar `59173234` had 218/750 checkpoints
on 16 CPUs, with both `next` cells complete. This progress observation is
time-specific; recheck before submission and consolidation.

`scripts/experimental_resume/r3d_anticipation_gpt_dest_shard.py` is a
separate scheduling adapter for exactly the two GPT destination cells:
`dest` and `dest-contained`, 150 saved fit tasks each. It imports and executes
the frozen R3-D `run_cv`, `augment_baseline`, `configure_blocks`,
`summarize`, folds, seeds and noise controls. It verifies the frozen driver,
specification and all input/code SHA256 bindings before fits. Its startup
guard exits if either canonical GPT destination cell has begun. Output is
isolated under `steering-v1/runs/resume-v1/r3d-anticipation-gpt-dest-shard-v1`.
The source SHA256 is
`266bc23d0730fde4e3dac1fd78b3695a93e9dbe0a46f048e98476f9d06a7355f`.

The prepared `scripts/experimental_resume/r3d_anticipation_gpt_dest_shard.sbatch`
requests one LEONARDO Booster node, 16 CPUs, 128 GiB, no GPU and an eight-hour
maximum: **128 CPU core-hours**. Shell syntax, Python compilation, frozen
preflight and Slurm `--test-only` passed. No job has been submitted. The
forecast from `--test-only` is provisional; the earlier Qwen sidecar began
immediately despite a multi-day forecast. At current observed rates, the
two GPT cells might take 3–6 hours on 16 CPUs; the eight-hour ceiling is a
limit, not a promise. The GPT population has more rows than Qwen, so Qwen
throughput should not be copied directly.

The existing bounded canonical chain and prior work were priced at no more
than 704 CPU core-hours. The current Qwen first slice adds at most 384, and
this GPT sidecar adds at most 128: **1,216 maximum CPU core-hours**, below
the 1,500-core-hour R3-D stage ceiling. A Qwen second slice is not included;
if needed, reprice the stage before launch. The observed Qwen progress makes
a second slice unlikely but does not certify completion.

The independent verifier `scripts/experimental_resume/verify_r3d_sidecar.py`
has a separate prepared 4-CPU, one-hour, zero-GPU sbatch (four maximum CPU
core-hours). If both GPT and Qwen sidecars are verified in separate full
slices, the conservative total becomes **1,224 CPU core-hours**. It checks
sealed source/cell receipts, exact model-chunk checkpoint names, frozen fit
bindings, held-out fold indices, finite prediction shapes and file hashes;
it writes a sealed inventory without touching the canonical tree. It has
not been submitted.

Before any submission, recheck that canonical `dest` and `dest-contained`
still have no fit checkpoints, that this source SHA matches the sbatch
pin, and that the task is still disjoint. Proposed command, after review:

```bash
sbatch --parsable scripts/experimental_resume/r3d_anticipation_gpt_dest_shard.sbatch
```

Consolidation is a later, separate operation. Wait for the sidecar's final
Slurm state `COMPLETED 0:0`, top-level sealed result, two sealed cell results,
all 300 exact-name fit checkpoints and finite saved predictions. Verify each
checkpoint's `binding` equals frozen specification SHA256
`62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92`;
each cell must have 150 checkpoints (six eight-model chunks × five repeats ×
five folds), and the saved `test`, `stats` and `lp:*` arrays must match the
expected frozen model chunk and have finite predictions. Compare the sidecar
adapter's two cell constructions against the frozen canonical driver: same
`dev`+`tune` population, reserved-B exclusion, `b!=a` destination subset,
contained filter `n_tokens>=32`, grouped folds, response labels, weights,
features and noise.

The guarded continuation `59115320` has an **afterany** dependency on
`59111360`: it can become runnable even if that job ends unsuccessfully.
Hold that exact continuation before the canonical slice ends. Once canonical
is stopped and no fit writer is active, recheck that the canonical GPT
destination directories and Qwen directories still lack fit checkpoints.
Copy only validated checkpoint `.npz` files into previously absent matching
canonical directories, using temporary filenames and atomic rename. Do not
copy sidecar `result.json` or `predictions.npz`; the unchanged canonical
driver must reconstruct these and validate every checkpoint binding. Release
the continuation only after the copy inventory and hashes agree. If canonical
has reached a target cell or either sidecar is incomplete, do not copy that
cell; let the unchanged driver finish it. Never count copied fits as an
independent replication.
