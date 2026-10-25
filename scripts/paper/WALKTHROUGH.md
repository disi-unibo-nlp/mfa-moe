# paper: walkthroughs

## overleaf-figure-build-trace: Native Overleaf figures, one routing preview
- commit: a413549
- source: Overleaf checkout paper/6a510c6db0d9b08aa438c142 at its commit 89234d7 (figures/journal/routing.tex, figures/journal/data/routing.dat, journal/style.tex, journal/results.tex); a fresh run on 2026-10-25 of the code at this commit, scripts/paper/build_paper_pdf.sh (paper.build_cl_figures, then paper.build_cl_manuscript) and paper.verify_cl_figure_labels, from a git worktree of this commit whose `paper` link points to the checkout (build output in the checkout's ignored paper/manuscript-support/build)
- code: src/paper/build_cl_figures.py, src/paper/build_cl_manuscript.py, src/paper/verify_cl_figure_labels.py, src/paper/cl_paths.py, scripts/paper/build_paper_pdf.sh, tests/paper/test_table_order.py

**Input:**
> figures/journal/routing.tex and figures/journal/data/routing.dat from the active Overleaf paper.
`cl_paths.py` takes the paper folder through the `paper` link of the repository root (`store.REPO`, the parent of
`src`) and resolves it, so a git worktree that links to the checkout reads the one paper folder and its receipts name
paths that outlive the worktree. The run was `scripts/paper/build_paper_pdf.sh` from such a worktree on the login node
(offline tectonic, 55.3 s for the figures and 17.1 s for the manuscript). The paper folder itself was only read.

### 1. Resolve canonical sources (`cl_paths.FIGURES`, `cl_paths.DATA`, `build_cl_figures.build`)

`FIGURES` is the paper checkout's `figures/journal`, `DATA` its `data` subfolder, and no other editable copy
is overlaid. The compiler is the verified offline tectonic (sha256 a98aa59a…), checked by `build_cl_manuscript.tectonic`.

| Source | SHA-256 |
|---|---|
| `figures/journal/routing.tex` | 845ee0bfe798f90b7c31a82e92df52cb38244658db12576472797ea1504903fa |
| `figures/journal/data/routing.dat` | d12cc4aa14a74b32626e7e371a7cf2dcef432ff21bbe624c55cd3753ca99e6e7 |
| `journal/style.tex` | 0a6a74114e65ffa17b6c3ed68f6471d3ff2eb0706bd55243aac3933be71bf611 |

Since bb9fb04 (where `routing.tex` was fae4827f…) the commit c98ed00 replaced 87 lines of `routing.tex` and 7 lines of
`routing.dat` (94 insertions and 94 deletions in all).

### 2. Compile an independent preview (`build_cl_figures.build`)

`build_paper_pdf.sh` builds all 11 figures (55.3 s in all, 2.8 to 10.1 s each), then the manuscript. For each figure the
build copies the style and the whole `figures/journal` folder into a fresh snapshot, compiles
`\input{figures/journal/routing}` as a standalone page with `--only-cached --untrusted`, and requires one page.

| Result | Value |
|---|---|
| compile time | 5.5 s |
| page size | 379.77 x 327.44 pt |
| `routing.pdf` SHA-256 | d7eccbb241b67058333f674939a0fcd08fecb03b444ca6637fdbf0c7df67e5e5 |
| the earlier trace, same `routing.tex` and `routing.dat` | 8558492561fd9f7b5eea4c7f4fe7815addf8e361ddb7749fc5bc2f0a14e5403a: another file hash, the same page size |
| the original PDF in `figures/pdf-backup/routing.pdf` | 4a5515af…, not identical: 97,436 of 1,493,400 pixel bytes differ at 144 dpi |

The PDF header carries a build-specific `/ID`, so the file hash changes with every build; the page size and the
render do not. The backup is the render of the earlier figure; the figure changed afterwards (step 1), so pixel
identity with it does not hold, and nothing is overwritten.

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
| passes (seconds) | 17.14; exit code 0; a rerun is not needed after the first |
| pages / SHA-256 of `main.pdf` | 53 / 9c2668155b376aa072bb0fdb38f0a28e8e3ff2c4380fb0043f7aa0072effc5e7 |
| the routing figure | Figure 8, page 20 |
| native figure inputs in the compiler log | all 11, once each (`conditioning` to `whole_states`) |
| `pdf-backup` folder in the build snapshot | absent |

`tests/paper/test_table_order.py` (6 tests, 0.03 s) covers the model display order and the table source
comparison, not the figure build; `tests/paper` holds 37 tests in all (0.58 s).

### Worth noticing

- A figure takes 3 to 10 s and the whole 53-page manuscript 17 s with the cached offline compiler, so
  `build_paper_pdf.sh` needs about 75 s, within the few minutes the script's comment states.
- The PDF backups are history, not a reference for today's figure: the routing figure moved on after them
  and its pixels differ from the backup in 6.5 percent of the render.
- Two builds of the same sources give different PDF hashes and identical pages, so a receipt should be compared by
  its sources, text and geometry, not by the PDF hash.
- Editable sources have one canonical place, the Overleaf checkout, shared by Overleaf and the local tools; the
  builds only read it and write below `paper/manuscript-support/build`.
- The label check ties each printed number to a row of `routing.dat`: 108 numbers, 108 bars, none touching.
- `paper/` is not tracked, so a git worktree reaches it through a link; `cl_paths` resolves that link, which keeps
  every path in `BUILD.json` valid after the worktree is removed.
## abstract-optimization-trace: Paper optimization, one retained abstract and final build
- commit: a413549
- source: paper/manuscript-support/build/elm-optimization/sections/abstract/ (baseline.tex, candidate-10.tex, selection-10.json), round-10-judge-mapping.json, round-10-judge-results.jsonl, round-10-judges.jsonl, RUN_STATUS.json, COMPLETENESS.json, ADOPTION.json, TOOLING_SOURCE.json; paper/manuscript-support/build/editorial-closeout/BUILD_FINAL.json, POSTBUILD_METADATA.json, PRINTED_TABLE_READBACK.json; a run on 2026-10-25 of the code at this commit on those saved files and saved ELM responses, with no client and no network call (the checker in step 4 read a clone of the Overleaf checkout at c98ed00, which has the same tree as the checkout's HEAD)
- code: src/paper/optimize_sections.py, src/paper/verify_cl_printed_tables.py, tests/paper/test_paper_optimization.py

**Input:**
> Abstract candidate-10.tex compared with incumbent candidate-09.tex against delivered baseline 9ff6d88.
The ELM calls were made once, on 2026-10-23, and are not repeated: every step below reads the responses that were saved
then. The Overleaf delivery commit is recorded separately. `TOOLING_SOURCE.json` records the research baseline d0b06ad2
and the sha256 of the three tooling files that made the run (optimize_sections 0e2963c2…, build_cl_manuscript d3c0be37…,
build_cl_figures 341453db…); the files at this commit hash differently (8cdb6ced…, 8c536357…, a1d3bc29…), so the replay
below shows that the code at this commit gives the saved result.

### 1. Generate and compare (`optimize_sections.select_candidate`, `protected_changes`, `_complete_response`)

The generator is ELM gpt-6.1-sol with medium reasoning (the 8 saved round-10 generator requests). Two independent ELM
gpt-6-luna calls at high effort see opposite A/B mappings (the 16 saved round-10 judge requests match the mapping file
by id; for the abstract the candidate is A in slot 1 and B in slot 2; across round 10 it is A in 8 and B in 8
judgments). Both saved abstract responses are complete JSON, and the guard against the fixed baseline finds
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

Scanning all `round-*-results.jsonl` files the way `_status` does (the loop copied into the trace, because `_status`
writes `RUN_STATUS.json`) gives 80 generator responses and 160 judge responses, 7 incomplete responses and 24 superseded
attempts, and 15,141,339 input, 1,807,708 output and 1,458,152 reasoning tokens, equal to `RUN_STATUS.json` and
`COMPLETENESS.json` (ten valid iterations for each of eight sections, 80 generators, 160 judgments). Failed and
superseded attempts remain archived; one interrupted batch has unknown usage (`COMPLETENESS.json`). `RUN_STATUS.json` records `paper_replacements` 0.

### 2. Select useful edits (independent manuscript adoption, `ADOPTION.json`)

The coordinator and independent scientific and style reviewers compare the final candidate with the fixed baseline
and the complete author examples. For the abstract the record says "adopt with stated edits": it accepts the clearer
retention and intervention claims and the explicit noninferiority boundary, and rejects the speculative routing
question, the reversal-count inventory and interval jargon. The same file records the retained candidate of every
section (candidate-10 for abstract, related work, method, setup, results and conclusion; candidate-09 for the
introduction; candidate-08 for the discussion). Humanizer input, diagnosis and the revised source are retained.
`scores_are_not_scientific_evidence` is true: scores do not establish scientific correctness or author acceptance.

### 3. Build and inspect (`build_cl_manuscript.compile_bounded`, read from `BUILD.json`)

The accepted optimization is the earlier stage. The later author-directed work moved the manuscript on (editorial
closeout, then the eight-model next-state, Qwen3.6 and completed-transition exhibits, then the sentence-level audit). The
final source-round-88 builds 53 pages in one bounded compiler pass (17.14 seconds), exit 0, with references
converged, no undefined reference, and the offline tectonic compiler `a98aa59a…`. Its receipt names the durable
paper folder, not the worktree that ran the build. The PDF hashes to
9c2668155b376aa072bb0fdb38f0a28e8e3ff2c4380fb0043f7aa0072effc5e7. The PDF named by the earlier closeout receipt
(`editorial-closeout/BUILD_FINAL.json`, source-round-74, 55 pages, a368faf7…) is no longer at the shared path
`build/build/main.pdf`, which each build replaces; it is kept at
`build/source-round-74/.paper-review/build/main.pdf`.

### 4. Read back tables (`verify_cl_printed_tables.main`)

The checker reads the final PDF against the source rows of the manuscript revision pinned in `ACCEPTED_REVISION`
(`89234d7`, the Overleaf commit of this build), by model and condition; no table is authorized to disappear
(`AUTHORIZED_REMOVALS` is empty). With `--build build/BUILD.json` it passes: all 19 tables, 457 rows and 2,669 printed
cells, including the next-state table with its lowest-is-best marks and the down-arrow headers. The swapped-row negative
control is rejected in the source and in the PDF. Run on the closeout receipt instead, the same code stops with
"PDF differs from build receipt", as it should, because that receipt names a PDF that no later build kept at its path.
The unit tests of the optimizer (`tests/paper/test_paper_optimization.py`: counterbalancing, regression veto, tie rule,
protected source, truncated and placeholder responses) are unchanged and pass with the other paper tests (34 passed with
`tests/test_imports.py`).

### Worth noticing

- The optimizer never directly overwrites the manuscript; adoption remains a separate recorded judgment.
- Selection can be replayed from saved responses alone: the whole round-10 decision follows from two JSON reviews, the
  blind mappings and the baseline text, with no model call.
- The printed-table check binds the PDF to its build receipt: a receipt whose PDF was replaced by a later build is refused, so the
  accepted revision is re-pinned with each delivered manuscript (a change of table cells needs this; a change of captions does not).
- Model ordering changes categorical coordinates and presentation, while receipt values and model identities remain fixed.
- The closeout permits a local Overleaf commit only. Original comments remain open and remote delivery awaits approval.
