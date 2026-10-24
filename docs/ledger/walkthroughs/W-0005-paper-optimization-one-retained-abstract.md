# W-0005: Paper optimization, one retained abstract and final build
- date: 2026-10-23
- commit: 54591f8eb39757889e1d94c4a33126de9f21b36a
- source: paper/manuscript-support/build/elm-optimization/sections/abstract/selection-10.json, COMPLETENESS.json, ADOPTION.json, TOOLING_SOURCE.json; paper/manuscript-support/build/editorial-closeout/BUILD_FINAL.json, PRINTED_TABLE_READBACK.json, TOOLING_SOURCE.json
- paths: src/paper/optimize_sections.py, tests/paper/test_paper_optimization.py
- keywords: paper, ELM, optimization, abstract, native review, Overleaf

**Input:**
> Abstract candidate-10.tex compared with incumbent candidate-09.tex against delivered baseline 9ff6d88.

The commit identifies the research baseline. Tooling remains uncommitted under the user's publication boundary; TOOLING_SOURCE.json binds the actual working files. Overleaf has its own delivery commit, recorded separately.

## 1. Generate and compare (optimize_sections)

The generator is ELM gpt-6.1-sol with medium reasoning. Two independent ELM gpt-6-luna High calls see opposite A/B mappings. The saved round-ten abstract scores are:

| Value | Observed result |
|---|---|
| Candidate mean | 8.19 |
| Incumbent mean | 8.05 |
| Candidate judge scores | 8.4 and7.9799999999999995 |
| Incumbent judge scores | 8.28 and7.82 |
| Retained candidate | candidate-10.tex |
| Retained SHA256 | ba80f19c8c1db20f3dd345d77630c30e7d6737e0e71fd420509485cd4e0fdbb0 |
| Author approval | false |

COMPLETENESS.json checks ten valid iterations for each of eight sections, with80generators and160judgments. Failed and superseded attempts remain archived; one interrupted batch has unknown usage.

## 2. Select useful edits (independent manuscript adoption)

The coordinator and independent scientific/style reviewers compare the final candidate with the fixed baseline and complete author examples. ADOPTION.json accepts clearer retention/intervention claims and the explicit noninferiority boundary, while rejecting the speculative routing question, reversal-count inventory and interval jargon. Humanizer input, diagnosis and revised source are retained. Scores do not establish scientific correctness or author acceptance.

## 3. Build and inspect (build_cl_manuscript.compile_bounded)

The accepted optimization remains the earlier stage. The later author-directed editorial closeout removes both superseded historical steering studies, reorders model comparisons by size and repairs prose wrapping. Its final source-round-74 builds 55 pages in two bounded compiler passes (174.5333115090616 and 72.53676056512631 seconds), exit 0, with references converged. PDF SHA256 is a368faf78ecbf7265bb5a4c70bf06600b0f546892343eb45b5e786ee61b5327f. No compiled source changed during that build. The afterward-updated checklist is noncompiled metadata, recorded in POSTBUILD_METADATA.json.

## 4. Read back tables (verify_cl_printed_tables)

The checker reads the final PDF against the accepted 4c9d7d4 source rows by model and condition, after the author-authorized model ordering and table removal. PRINTED_TABLE_READBACK.json records all 19 surviving tables, 453 rows and 2,641 printed cells, including 2,188 non-label scientific cells. The source check separately verifies current benchmark and routing receipts. Swapped row, changed interval and incorrect rank controls are rejected; printed empty cells and unavailable marks are included.

## Worth noticing

- The optimizer never directly overwrites the manuscript; adoption remains a separate recorded judgment.
- Model ordering changes categorical coordinates and presentation, while receipt values and model identities remain fixed.
- The historical studies are removed by the author, not rerun; the separate candidate-recognition qualification remains.
- The closeout permits a local Overleaf commit only. Original comments remain open and remote delivery awaits approval.
- LFM native coefficients are integrated, but the r20 whole-reasoning result replacement and eight-model next-state extension remain pending external receipts.
