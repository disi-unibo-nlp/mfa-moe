# Corrected causal steering cohort

**Accepted 2026-10-20:** source commit `0084256`, manifest commit `fbc7ccd`.
CPU qualification 61554184 is observed RUNNING; GPT engineering 61554207 and
Qwen engineering 61554215 are PENDING on its afterok gate. All three resource
cards and accepted store owners were read back. The initial actual-terminal plus
full-outstanding ledger is 8 GPU-hours and 4.195556 CPU core-hours, including the
failed predecessor. Scientific generation is not yet submitted or qualified.
See `OBSERVED_LAUNCH.json`, the submission journal and source archive verification.

This is the separately bound continuation authorized by the user's “keep going”
after r2 CPU validation failed. Preserve r2 and its complete source archive,
allocated test receipt, terminal accounting, ownership and submission journals.
The diagnosed failure was fixture reliance on global temporary-directory
selection. Wrappers now set TMPDIR inside the Slurm task; worker startup pins
Python temporary files to an existing private HOME/WORK directory. Test fixtures
use explicit WORK staging. No installed environment or agent runtime changed.

The original 7,424 assignments, 2,007 authorized Qwen generation UIDs, 37 known
incompatible unknowns, scientific parameters, classifier, seeds, deadline and
64-GPU/16-CPU-hour additional limits remain fixed. COHORT.json binds r2 as its
terminal predecessor, including its archive and failure receipts. Every status
and admission refreshes native predecessor accounting. Its failed CPU charge is
176 × 4 / 3,600 = 0.195556 core-hours; its three cancelled GPU jobs have no
allocated resources, Start=None and zero runtime. Prior failures cannot satisfy
the corrected release's prerequisites. No budget is reset.

New stores live under user WORK in
`mfa-moe/campaign-runs/causal-rescue-20261020-r3`. SOURCE.json binds the committed
correction and installed runtime. ENGINEERING.json freezes the same metadata
selection recipe and 64 scientific engineering cells, plus separate canaries.
No model loads before exact ownership, private staging, source/cohort, budget,
deadline and actual prerequisite checks. Scientific generation requires both
allocated engineering qualifications. Software failures halt downstream release.

Use the maintained command prefix:

```bash
env PYTHONPATH=src:scripts/experimental_resume \
  TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp \
  /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
  -m moe_exp.study.rescue
```

All commands require `--cohort report/final-routing-rescue-20261020-r3/COHORT.json`.
Initial submissions are `submit --stage validate` and the two `engineering`
model submissions, with exact native readbacks and afterok gates. Submit
`generate --model qwen36` only after actual engineering qualification, including
its eligible reserve if needed. This keeps an unused generation job from being
cancelled by a recoverable earlier engineering interruption. `status` includes
predecessor actuals. After terminal generation,
use `submit --stage prepare`, `submit --stage pilot`, `reader-admit`, the two
sequential waves, the first eligible bounded reserve if needed, and protected
`submit --stage analysis`. Terminal extraction retains every planned unknown;
reader eligibility does not require all assignments to succeed.

Reader labels enter actual semantic effects. Six simultaneous contrasts include
identification bounds and shared family draws; compatibility midpoints are not
observed effects. The 512-occurrence trajectory audit diagnoses measurement
disagreement and does not certify a small bias or justify correction. Historical
sources keep their original provenance limitations. No observed post-pulse rescue
window exists. Canonical manuscript editing and remote pushes are excluded.

Final reporting still requires measured results, fresh main/strong/rescue
analyses, coverage and missingness, intervals and response matrices, engineering
dose and reader disagreement diagnostics, accuracy/token tradeoffs, standalone
exhibits with author-style/visual checks, and the manuscript checklist. Accepted
jobs or this README do not establish scientific completion.
