# Ported readout, next-state and benchmark helpers

Provenance note for code copied into the repository from outside it. Step 6 simplifies it.

- `dp/`, `dynrt/`, `rkin/`: the import closure (23 modules) of the two drivers, copied from
  `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code/`
  (the files the frozen readout and next-state runs hashed). The drivers are `readout.py` (was
  `scripts/experimental_resume/r3d_readout.py`, byte-identical to the external copy) and `anticipation.py` (was
  `r3d_anticipation.py`). Unported siblings of the same packages are experiments the drivers do not import.
- Import rewrites only: `from dp|dynrt|rkin ...` and `from r3d_readout import ...` now name `routing.*`;
  the two `sys.path.insert(0, OUT / 'code')` lines are gone because the packages are importable.
- Changed since the port (drivers only, no fitting or scoring code): `readout.py` and `anticipation.py` write to a
  new `--out` folder, compare their inputs with the first run's frozen spec (`forum/tests/r3_context`) and record
  the hashes of the `routing` modules that ran instead of the `code/` copies; the `prepare()` helpers that copied
  the packages are gone; the login-node refusal is now a check for a Slurm batch job (`SLURM_JOB_ID` and
  `SLURMD_NODENAME`: the serial partition's nodes are called login08 and login13, and the shell `salloc` starts on
  a login node has no `SLURMD_NODENAME`). Functions and constants nothing referenced were removed from `dp`,
  `dynrt` and `rkin`; `rkin/common.py`, `dp/common.py` and `dynrt/common.py` take their root from
  `store.ANALYSIS`. Neither driver can be rerun in full (the readout is 19, the anticipation
  84 core-hours). A one-CPU smoke run (job 62007859) passed the input check against the first run's specs; the 37
  GPT readout fold checkpoints and the first anticipation task it finished were bit-identical to the first run's.
- `benchmark_accuracy.py`: copy of `tools/tmp/claude-analysis-2026-09-24/outcomes_analysis.py` (accuracy, budget,
  continuation, error-share and length-gap tables).
- `benchmark_inventory.py`: copy of `report/model-analysis-review-20261008/build_inventory.py` (which stays in the
  report folder). It takes the output folder as its argument.
