# W-0005: Paper optimization, one retained abstract and final build
- date: 2026-10-24
- commit: 4dfb367114824c4200e0042b60d0d4d48cdfea1b
- source: paper/manuscript-support/build/elm-optimization/sections/abstract/ (baseline.tex, candidate-10.tex, selection-10.json), round-10-judge-mapping.json, round-10-judge-results.jsonl, round-10-judges.jsonl, RUN_STATUS.json, COMPLETENESS.json, ADOPTION.json, TOOLING_SOURCE.json; paper/manuscript-support/build/editorial-closeout/BUILD_FINAL.json, POSTBUILD_METADATA.json, PRINTED_TABLE_READBACK.json; a run on 2026-10-24 of the code at this commit on those saved files and saved ELM responses, with no client and no network call
- paths: src/paper/optimize_sections.py, src/paper/verify_cl_printed_tables.py, tests/paper/test_paper_optimization.py
- keywords: paper, ELM, optimization, abstract, native review, Overleaf

**Input:**
> Abstract candidate-10.tex compared with incumbent candidate-09.tex against delivered baseline 9ff6d88.

This is the retrace of the walkthrough first written on 2026-10-23 at commit 54591f8, when the optimizer was
`scripts/paper/optimize_sections.py`; it is now `src/paper/optimize_sections.py`. The ELM calls were made once and are not
repeated: every step below reads the responses that were saved then. The Overleaf delivery commit is recorded separately.
`TOOLING_SOURCE.json` still binds the 2026-10-23 working files of the old tooling path.

## 1. Generate and compare (`optimize_sections.select_candidate`, `protected_changes`, `_complete_response`)

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

## 2. Select useful edits (independent manuscript adoption, `ADOPTION.json`)

The coordinator and independent scientific and style reviewers compare the final candidate with the fixed baseline
and the complete author examples. For the abstract the record says "adopt with stated edits": it accepts the clearer
retention and intervention claims and the explicit noninferiority boundary, and rejects the speculative routing
question, the reversal-count inventory and interval jargon. The same file records the retained candidate of every
section (candidate-10 for abstract, related work, method, setup, results and conclusion; candidate-09 for the
introduction; candidate-08 for the discussion). Humanizer input, diagnosis and the revised source are retained.
`scores_are_not_scientific_evidence` is true: scores do not establish scientific correctness or author acceptance.

## 3. Build and inspect (`build_cl_manuscript.compile_bounded`, read from `BUILD_FINAL.json`)

The accepted optimization is the earlier stage. The later author-directed editorial closeout removes both superseded
historical steering studies, reorders model comparisons by size and repairs prose wrapping. Its final source-round-74
builds 55 pages in two bounded compiler passes (174.5333115090616 and 72.53676056512631 seconds), exit 0, with
references converged; no compiled source changed during that build. The PDF on disk still hashes to
a368faf78ecbf7265bb5a4c70bf06600b0f546892343eb45b5e786ee61b5327f and has 55 pages. The checklist `REVIEW_TODOS.md` was
refreshed after the build (noncompiled metadata, `POSTBUILD_METADATA.json`), so the snapshot file no longer matches
the hash in the build receipt.

## 4. Read back tables (`verify_cl_printed_tables.main`)

The checker reads the final PDF against the accepted 4c9d7d4 source rows by model and condition, after the
author-authorized model ordering and table removal (`tab:historical-interventions`). With the receipt's hash of
`REVIEW_TODOS.md` replaced by the refreshed one in a scratch copy (everything else untouched) it passes in 6.5 s: all
19 surviving tables, 453 rows and 2,641 printed cells, including 2,188 non-label scientific cells. The table section and
the source verification of the new run equal the saved `PRINTED_TABLE_READBACK.json` exactly. The swapped-row negative
control is rejected; printed empty cells and unavailable marks are included. The unit tests of the optimizer
(`tests/paper/test_paper_optimization.py`: counterbalancing, regression veto, tie rule, protected source, truncated and
placeholder responses) pass, 8 tests.

## Worth noticing

- The optimizer never directly overwrites the manuscript; adoption remains a separate recorded judgment.
- Selection can be replayed from saved responses alone: the whole round-10 decision follows from two JSON reviews, the
  blind mappings and the baseline text, with no model call.
- The printed-table check depends on the build snapshot being byte-identical to its receipt; a refreshed checklist is
  enough to stop it, so the post-build metadata edit has to be declared, as it is in `POSTBUILD_METADATA.json`.
- Model ordering changes categorical coordinates and presentation, while receipt values and model identities remain fixed.
- The closeout permits a local Overleaf commit only. Original comments remain open and remote delivery awaits approval.
