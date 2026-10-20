# Causal steering cohort

The user authorized an additional **64 GPU-hours and 16 CPU-only core-hours**
for the fixed final implementation plan. Historical failures remain charged
separately. Science stops at **2026-10-23 05:30 Europe/Rome**, with reporting due
at **11:30**. Canonical manuscript edits and pushes are excluded.

This release preserves the original 7,424 assignments, freezes the 2,007
untouched Qwen generation UIDs, and retains the 37 incompatible assignments as
unknowns. Historical candidates must pass allocated capture and saved-token
qualification. Historical r8 outputs retain their incomplete original source
closure; they are never products of this release.

`BINDING.json` changes only the private bulk/control destinations. `SOURCE.json`
will bind the scoped code commit, Python closure, wrappers, tests, inputs and
installed runtime. `COHORT.json` follows that source receipt and fixes the
population, classifier folds, reader configuration and historical references.
`ENGINEERING.json` fixes eight four-cell blocks per model using metadata only.

The approved allocation limits are:

| Stage | Shape and maximum | GPU-hours |
|---|---|---:|
| Engineering | Two separate 2-GPU, 16-CPU, 120-GiB, two-hour jobs | 8 |
| Untouched Qwen completion | 2 GPUs, 16 CPUs, 120 GiB, six hours | 12 |
| Reader pilot | Same GPU shape, one hour, 128 actual contexts | 2 |
| Two disjoint reader waves | Same GPU shape, six hours each | 24 |
| Single operational reserve | Same GPU shape, up to nine hours | 18 |
| Total additional ceiling | Full limits include the reserve | 64 |

Validation, preparation and final analysis each receive four CPUs and 24 GiB
for at most one hour. Four additional CPU-only core-hours remain reserve. Final
analysis retains its own four core-hours. These are CPU-only hours; GPU-stage
CPU billing is additionally checked against native credits and the inherited
42,050-hour study ceiling. Separate 10-GiB WORK and FAST reserves persist.

Use the existing study Python environment and export `PYTHONPATH=src` and a
user-WORK `TMPDIR`. The installed CINECA operations skill governs native cluster
commands. The maintained command prefix is:

```bash
/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
  -m moe_exp.study.rescue
```

Every execution command takes
`--cohort report/final-routing-rescue-20261020-r2/COHORT.json`. Submit validation,
the two engineering models, and Qwen generation through `submit --stage ...`.
Engineering and generation can queue with `afterok` dependencies; invalid
dependencies cancel their unused jobs and every worker checks its gate before
loading a model. `status` reconciles actual terminal use and full outstanding
limits. It never resets historical usage or resubmits an ambiguous intent.

After generation or its eligible reserve is terminal, submit `prepare`. It
publishes the complete quiescent generation index and reader inventories,
freezes the 16-stratum occurrence sample and actual context lengths, and fixes
the pilot and two disjoint waves before new annotation. Submit `pilot`, then
run `reader-admit` on its measured initialization/loading/batch/parsing/shutdown
costs. Submit admitted `wave-0`, then `wave-1`. The single `reserve` selects the
first eligible unfinished stage from operational state, never effects. Partial
generation and reader admissions without output remain unknown.

Submit protected `analysis` after every accepted owner is terminal. It writes
separate fresh main, strong and rescue analyses and standalone CSV/LaTeX/PDF
exhibits under the cohort's WORK analysis root. The semantic first-complete
endpoint is frozen on 20 October after historical generation; it is not claimed
to have been specified before r8. All six contrasts share family-bootstrap draws
and a joint critical value including identification components. Compatibility
midpoints are not observed effects, and nonsignificance does not establish
equivalence. There is no observed post-pulse rescue window.

No job or scientific success is implied by this README. Native accepted IDs,
actual qualification, annotation and accounting receipts determine the state.
At preparation, scheduler test-only checks accepted the requested CPU and GPU
shapes, with estimated starts late on 21 October. Those printed test-only job
numbers are not submitted jobs, and the estimates are not guaranteed.

Remaining manuscript work is separate: LFM/readout integration, accepted
text-judge corrections, outstanding page reviews and final PDF verification.
