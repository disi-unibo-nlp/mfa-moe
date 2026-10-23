# Ported readout, next-state and benchmark helpers

Provenance note for code copied into the repository from outside it. Step 6 simplifies it.

- `dp/`, `dynrt/`, `rkin/`: the import closure (23 modules) of the two drivers, copied from
  `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code/`
  (the files the frozen readout and next-state runs hashed). The drivers are `readout.py` (was
  `scripts/experimental_resume/r3d_readout.py`, byte-identical to the external copy) and `next_state.py` (was
  `r3d_anticipation.py`). Unported siblings of the same packages are experiments the drivers do not import.
- Import rewrites only: `from dp|dynrt|rkin ...` and `from r3d_readout import ...` now name `moe_exp.routing.*`;
  the two `sys.path.insert(0, OUT / 'code')` lines are gone because the packages are importable.
- Left as they were: the `prepare()` helpers and the `code` hashes in both drivers still point at the old
  `forum/tests/r3_context/code` folder, and `dynrt.common`/`rkin.common` keep their absolute data paths.
- `benchmark_accuracy.py`: copy of `tools/tmp/claude-analysis-2026-09-24/outcomes_analysis.py` (accuracy, budget,
  continuation, error-share and length-gap tables).
- `benchmark_inventory.py`: copy of `report/model-analysis-review-20261008/build_inventory.py` (which stays in the
  report folder). Its `ROOT`/`OUT` still assume that folder.
