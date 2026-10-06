# R3-D readout continuation, 2026-10-01

The user authorized all compute needed for the named paper analyses in the
2026-10-01 instruction recorded in `RESOURCE_AUTHORIZATION_EXPANDED.json`.
This is a resource amendment to the historical R3-D ceiling, not a change to
the frozen statistical specification or prior results.

The v3 driver at
`/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code/r3d_readout.py`
still has SHA256
`1de08eae6e8d85e352d091c829b33598d2a848b2a9b28f1937a14d687d749037`,
as recorded in `FROZEN.readout.v3.json`. The driver validates its input and
code hashes at runtime and reuses fold checkpoints only when their family seal
matches. Three of fourteen model/mode results are complete; 137 of 490 fold
checkpoints exist. The previous 1:57 job produced roughly 67 additional folds
before a Slurm time limit. The remaining 353 folds could take more than ten
hours at that rate, and later models may differ. One 24-hour continuation on
four CPU cores with 64 GiB is therefore the bounded recovery allocation,
at **at most 96 CPU core-hours**. It exits when all cells complete, so actual
charge is expected to be lower. The extension preserves seven models, both
contexts, five outer folds, five lexical nulls, family grouping and all input
hashes. Checkpointed progress remains usable if the bound proves insufficient.

The first continuation, job **59109995**, failed after 91 seconds because the
frozen v3 driver's Python equality check compares its in-memory `MODEL_ORDER`
tuple with the JSON-loaded list. The independent CPU audits **59110273** and
**59110653** verified all 3,776 frozen file hashes and the identical canonical
specification seal `a2520bd038bb511732bb6adb9423f620ca5e5a1db17034f69bbd3dbb864ca1cc`.
`r3d_readout_resume.py` converts that tuple to a list before invoking the
unchanged frozen driver. It preserves model order, statistics and the checkpoint
seal. The failed 91-second allocation spent about .1011 CPU core-hour.

Submit the corrected wrapper through the existing `steer_cpu.sbatch` wrapper on `boost_usr_prod`,
`iscrc_miosr`, `normal`. Verify the exact job ID, parent and step final states,
`readout-results.v3.json`, fourteen per-cell results, and input seal before
using findings in the paper. This continuation is a new spending line; the
frozen JSON's historical eight-hour reservation remains historical.
