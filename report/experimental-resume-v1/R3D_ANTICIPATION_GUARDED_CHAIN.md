# R3-D anticipation guarded continuation, 2026-10-01

The user's instruction to submit the runs needed for the paper authorizes a
bounded Slurm dependency chain while the frozen 1,650-task anticipation grid
is running. This amends the manual one-slice-at-a-time scheduling procedure in
`R3D_ANTICIPATION_RESOURCE.md`; it does not change the statistical driver,
inputs, folds, responses, noise controls or 1,500 CPU-core-hour stage ceiling.

The active first 24-hour slice is job 59111360. At most six further 24-hour,
four-core, 64-GiB slices are pre-submitted sequentially. Their **combined
maximum allocation is 576 CPU core-hours**; with the current slice and a
conservative 32-core-hour allowance for historical work, the bound is 704,
below the 1,500 ceiling. Actual expenditure is reported from Slurm.

`r3d_anticipation_guard.py` checks the prior parent job's final state and exit
code, frozen driver/specification SHA256, and growth in the checkpoint count.
The first guard accepts a successful finish or a time limit with new valid
checkpoints; later guards require the previous guard to complete cleanly.
Every guard runs the unchanged frozen driver. It stops its child 20 minutes
before Slurm's 24-hour limit, verifies checkpoint growth, and exits cleanly
only on valid progress or a sealed final result. No-progress or unexpected
failure stops the `afterok` chain. Already-complete results cause dependent
jobs to exit immediately. Invalid dependencies are killed without inference.

This is a bounded Slurm workflow, not a login-node service or gateway. Each
job ID, state, exit code, checkpoints and final result are audited. If six
slices do not finish the grid, continuation requires a fresh measured runtime
review before more CPU allocation.
