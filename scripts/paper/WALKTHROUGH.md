# paper: walkthroughs

## overleaf-figure-build-trace: Native Overleaf figures, one routing preview
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: Overleaf checkout paper/6a510c6db0d9b08aa438c142 at its local commit c98ed00c46604c256056ff72481db97bd49b6fa4 (figures/journal/routing.tex, figures/journal/data/routing.dat, journal/style.tex, journal/results.tex); a fresh run on 2026-10-24 of paper.build_cl_figures routing, paper.verify_cl_figure_labels and paper.build_cl_manuscript on an export of that commit (build output in a scratch folder, not in the repository)
- code: src/paper/build_cl_figures.py, src/paper/build_cl_manuscript.py, src/paper/verify_cl_figure_labels.py, src/paper/cl_paths.py, scripts/paper/build.sh, tests/paper/test_table_order.py

**Input:**
> figures/journal/routing.tex and figures/journal/data/routing.dat from the active Overleaf paper.
This is the retrace of the walkthrough first written on 2026-10-22 (Overleaf commit bb9fb04), when the
tooling lived in `scripts/paper` and `paper/manuscript-support`; it is now `src/paper`. `cl_paths.py` finds the
manuscript relative to the repository root, and a fresh git worktree only holds the old PDF-only copy of the
paper folder. The run therefore used `git archive` of the Overleaf commit above, placed beside a copy of
`src/` from this commit. The paper folder itself was only read.

### 1. Resolve canonical sources (`cl_paths.FIGURES`, `cl_paths.DATA`, `build_cl_figures.build`)

`FIGURES` is the paper checkout's `figures/journal`, `DATA` its `data` subfolder, and no older editable copy
is overlaid. The compiler is the verified offline tectonic (sha256 a98aa59a…).

| Source | SHA-256 |
|---|---|
| `figures/journal/routing.tex` | 845ee0bfe798f90b7c31a82e92df52cb38244658db12576472797ea1504903fa |
| `figures/journal/data/routing.dat` | d12cc4aa14a74b32626e7e371a7cf2dcef432ff21bbe624c55cd3753ca99e6e7 |
| `journal/style.tex` | 55e886d701f0c2fcc2cb90aa28626bb507513326dc08e31c27a7e4efa0a79a1c |

Since bb9fb04 (where `routing.tex` was fae4827f…) the commit c98ed00 changed 94 lines of `routing.tex` and 14
lines of `routing.dat`.

### 2. Compile an independent preview (`build_cl_figures.build`)

The build copies the style and the whole `figures/journal` folder into a fresh snapshot, compiles
`\input{figures/journal/routing}` as a standalone page with `--only-cached --untrusted`, and requires one page.

| Result | Value |
|---|---|
| compile time | 8.9 s |
| page size | 379.77 x 327.44 pt (the same size as on 2026-10-22) |
| `routing.pdf` SHA-256 | 5262bc0396e3cea9580d8b29b58365998ac28808ce4b3bedb6ae3f1e55699385 |
| the original PDF in `figures/pdf-backup/routing.pdf` | 4a5515af…, not identical: 97,436 of 1,493,400 pixel bytes differ at 144 dpi |

The backup is the render of the earlier figure; the figure changed afterwards (step 1), so the pixel
identity measured on 2026-10-22 no longer holds, and nothing is overwritten.

### 3. Check values and marks (`verify_cl_figure_labels.main`)

The expected labels are read from the `\node` lines of `routing.tex`, each checked against the `routing.dat`
value it annotates; the PDF is then searched for rotated numeric text, bars and strokes.

| Check | Result |
|---|---|
| expected numeric labels / labels in the PDF / bars | 108 / 108 / 108 |
| short stroke segments checked | 510 |
| number and bar overlaps; number and interval-stroke overlaps | 0; 0 |
| smallest number-to-bar clearance | 25.73 pt |
| rotated x-tick label declarations in any figure source | 0 |

The receipt keeps `visual_inspection_still_required: true`; the script cannot see the picture.

### 4. Render inside the paper (`build_cl_manuscript.main`, `compile_bounded`)

The manuscript inputs `figures/journal/routing` directly (`journal/results.tex`, a `\journalfigure` in a
`figure` environment). Every source file goes into a fresh snapshot, then at most four compiler passes run
until the references stop changing.

| Result | Value |
|---|---|
| passes (seconds) | 14.0, 12.2, 12.1, 12.2; exit code 0 each; a rerun is no longer needed after the fourth |
| pages / SHA-256 of `main.pdf` | 55 / b1b420d599993af2560de0a9812df1e69bc54bbcc2365b6fcaed25d04ceee5e4 |
| the routing figure | Figure 8, page 20 |
| native figure inputs in the compiler log | all 11, once each (`conditioning` to `whole_states`) |
| `pdf-backup` folder in the build snapshot | absent |

The 14 tests of `tests/paper` (`test_table_order.py`, `test_paper_optimization.py`) pass in 0.12 s. They cover
the table order and the section optimizer, not the figure build.

### Worth noticing

- A figure preview takes 9 s and the whole 55-page manuscript 50 s with the cached offline compiler, far
  below the few minutes the build notes assume.
- The PDF backups are history, not a reference for today's figure: the routing figure moved on after them
  and its pixels differ from the backup in 6.5 percent of the render.
- Editable sources have one canonical place, the Overleaf checkout, shared by Overleaf and the local tools; the
  builds only read it and write below `paper/manuscript-support/build`.
- The label check ties each printed number to a row of `routing.dat`: 108 numbers, 108 bars, none touching.
- In a fresh worktree the paper folder is the old PDF-only copy: `build_cl_figures routing` stops there with
  "figure source missing for requested figure(s): routing" instead of rendering old PDFs, which is the useful failure.

## abstract-optimization-trace: Paper optimization, one retained abstract and final build
- commit: 4dfb367114824c4200e0042b60d0d4d48cdfea1b
- source: paper/manuscript-support/build/elm-optimization/sections/abstract/ (baseline.tex, candidate-10.tex, selection-10.json), round-10-judge-mapping.json, round-10-judge-results.jsonl, round-10-judges.jsonl, RUN_STATUS.json, COMPLETENESS.json, ADOPTION.json, TOOLING_SOURCE.json; paper/manuscript-support/build/editorial-closeout/BUILD_FINAL.json, POSTBUILD_METADATA.json, PRINTED_TABLE_READBACK.json; a run on 2026-10-24 of the code at this commit on those saved files and saved ELM responses, with no client and no network call
- code: src/paper/optimize_sections.py, src/paper/verify_cl_printed_tables.py, tests/paper/test_paper_optimization.py

**Input:**
> Abstract candidate-10.tex compared with incumbent candidate-09.tex against delivered baseline 9ff6d88.
This is the retrace of the walkthrough first written on 2026-10-23 at commit 54591f8, when the optimizer was
`scripts/paper/optimize_sections.py`; it is now `src/paper/optimize_sections.py`. The ELM calls were made once and are not
repeated: every step below reads the responses that were saved then. The Overleaf delivery commit is recorded separately.
`TOOLING_SOURCE.json` still binds the 2026-10-23 working files of the old tooling path.

### 1. Generate and compare (`optimize_sections.select_candidate`, `protected_changes`, `_complete_response`)

The generator is ELM gpt-6.1-sol with medium reasoning. Two independent ELM gpt-6-luna calls at high effort see
opposite A/B mappings (for the abstract the candidate is A in slot 1 and B in slot 2; across round 10 it is A in 8
and B in 8 judgments). Both saved abstract responses are complete JSON, and the guard against the fixed baseline finds
no protected change (`vetoes` empty, no added numeric literal). `select_candidate` on those two reviews gives the
saved `selection-10.json` on every field it returns:

| Value | Rerun result |
|---|---|
| Candidate mean | 8.19 |
| Incumbent mean | 8.05 |
| Candidate judge scores | 8.4 and 7.9799999999999995 |
| Incumbent judge scores | 8.28 and 7.82 |
| Votes (slot 1, slot 2) | A, B: the candidate in both |
| Scientific regressions | none |
| Retained candidate | candidate-10.tex |
| Retained SHA256 | ba80f19c8c1db20f3dd345d77630c30e7d6737e0e71fd420509485cd4e0fdbb0 |
| Author approval | false |

Scanning all `round-*-results.jsonl` files the way `_status` does gives 80 generator responses and 160 judge
responses, 7 incomplete responses and 24 superseded attempts, equal to `RUN_STATUS.json` and `COMPLETENESS.json`
(ten valid iterations for each of eight sections, 80 generators, 160 judgments). Failed and superseded attempts
remain archived; one interrupted batch has unknown usage. `RUN_STATUS.json` records `paper_replacements` 0.

### 2. Select useful edits (independent manuscript adoption, `ADOPTION.json`)

The coordinator and independent scientific and style reviewers compare the final candidate with the fixed baseline
and the complete author examples. For the abstract the record says "adopt with stated edits": it accepts the clearer
retention and intervention claims and the explicit noninferiority boundary, and rejects the speculative routing
question, the reversal-count inventory and interval jargon. The same file records the retained candidate of every
section (candidate-10 for abstract, related work, method, setup, results and conclusion; candidate-09 for the
introduction; candidate-08 for the discussion). Humanizer input, diagnosis and the revised source are retained.
`scores_are_not_scientific_evidence` is true: scores do not establish scientific correctness or author acceptance.

### 3. Build and inspect (`build_cl_manuscript.compile_bounded`, read from `BUILD_FINAL.json`)

The accepted optimization is the earlier stage. The later author-directed editorial closeout removes both superseded
historical steering studies, reorders model comparisons by size and repairs prose wrapping. Its final source-round-74
builds 55 pages in two bounded compiler passes (174.5333115090616 and 72.53676056512631 seconds), exit 0, with
references converged; no compiled source changed during that build. The PDF on disk still hashes to
a368faf78ecbf7265bb5a4c70bf06600b0f546892343eb45b5e786ee61b5327f and has 55 pages. The checklist `REVIEW_TODOS.md` was
refreshed after the build (noncompiled metadata, `POSTBUILD_METADATA.json`), so the snapshot file no longer matches
the hash in the build receipt.

### 4. Read back tables (`verify_cl_printed_tables.main`)

The checker reads the final PDF against the accepted 4c9d7d4 source rows by model and condition, after the
author-authorized model ordering and table removal (`tab:historical-interventions`). With the receipt's hash of
`REVIEW_TODOS.md` replaced by the refreshed one in a scratch copy (everything else untouched) it passes in 6.5 s: all
19 surviving tables, 453 rows and 2,641 printed cells, including 2,188 non-label scientific cells. The table section and
the source verification of the new run equal the saved `PRINTED_TABLE_READBACK.json` exactly. The swapped-row negative
control is rejected; printed empty cells and unavailable marks are included. The unit tests of the optimizer
(`tests/paper/test_paper_optimization.py`: counterbalancing, regression veto, tie rule, protected source, truncated and
placeholder responses) pass, 8 tests.

### Worth noticing

- The optimizer never directly overwrites the manuscript; adoption remains a separate recorded judgment.
- Selection can be replayed from saved responses alone: the whole round-10 decision follows from two JSON reviews, the
  blind mappings and the baseline text, with no model call.
- The printed-table check depends on the build snapshot being byte-identical to its receipt; a refreshed checklist is
  enough to stop it, so the post-build metadata edit has to be declared, as it is in `POSTBUILD_METADATA.json`.
- Model ordering changes categorical coordinates and presentation, while receipt values and model identities remain fixed.
- The closeout permits a local Overleaf commit only. Original comments remain open and remote delivery awaits approval.
