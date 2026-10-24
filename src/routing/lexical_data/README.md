# Lexicon data builders (provenance)

These scripts built the native and steered judge-labelled lexicon tables on 2026-10-17
from the campaign-v3 labels (`build_native.py`, `build_steered.py`, `build_extras.py`),
checked them (`validate.py`) and generated the table README (`make_readme.py`).
They are copied unchanged from the session scratch directory where they were run.

- Their `OUT` path points to session scratch
  (`/scratch_local/claude-133943/.../scratchpad/lexicon/data`); that directory is temporary.
- The native tables used by the efficiency-link pre-registration are copied, with their README,
  to `/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/final-routing-20261015-r1/lexical/`
  (sha256 recorded in `report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_PREREG.json`).
- Usage and column meanings: see the copied `README.md` in that directory.
