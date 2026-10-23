# F-0007: Optimize manuscript sections with ELM scoring
- status: active
- date: 2026-10-22
- request: "once u are donoe, pull the latest paperbackwriter skill and run 10 optiimization ruounds of that ELm based scoring. Using sol 6.1 medium as generator via ELM and 3 deepseek models. Do 10 iter for each section. The u comapre the new candidate with the one u are working on now (the best one when u stop), and decide what to take from that proposed solution. Allways following paparbackwriter indiations."
- purpose: Explore section revisions through the requested generator and two independent Luna High judges, as amended by the author on October 23, then retain only independently justified improvements over the delivered author-comment revision.
- acceptance: Each requested section has 10 valid generation/scoring iterations with actual model IDs, raw outputs, scores and best-candidate selection recorded; candidates are compared with the fixed current baseline and useful changes are chosen with reasons; scientific values, citations, equations and scope survive; final source/build/reference/visual checks and authorized delivery are read back.
- keywords: paper, ELM, optimization, scoring, Luna, Sol 6.1, generator, ten iterations
- paths: paper/manuscript-support, scripts/paper
- decisions: D-0004, D-0005

## Tasks

- [x] F-0007.1 Finish the existing comment delivery, preserve its best current source as the fixed comparison baseline, and pull the latest paper-writer source.; evidence: Overleaf 9ff6d88a021cc04e98f41ee9a65e8dcf87148895 and review-revision/DELIVERY.json; skill pull reports already up to date at 9885c78; local five-file feedback patch preserved.
- [x] F-0007.2 Assemble complete section-matched author examples, current paper rules, scientific invariants and reproducible scoring inputs for each section; preserve baseline versions.; evidence: elm-optimization/PROTOCOL.json, eight baseline/example/rule packets, LOC-049 and PAIRWISE_RUBRIC.md; REFERENCE_CONTEXT_AUDIT.json records the complete Related Work wrapper correction and supplemental Method examples. Each actual round has 16 counterbalanced judge calls. The active run covers the eight main section units; appendices are protected scientific source material, not optimization units.
- [x] F-0007.3 Validate gpt-6.1-sol at medium effort and the two author-approved gpt-6-luna High judge calls; record service responses and token usage without credentials. Generator validation succeeded in all eight actual ELM calls. The author replaced the three DeepSeek judges with two Luna 6.1 High judges. Live ELM catalog lists gpt-6-luna but not gpt-6.1-luna; an exact Responses request returned 404 model_not_found. The author approved gpt-6-luna with “use gpt 6 luna” and “keep going”; All 16 first-round GPT-6 Luna High judgments are complete; seven output-limited attempts remain charged and were recovered with larger response allowances. Evidence: actual round-01 judge requests/results and completion records; RUN_STATUS.json.
- [ ] F-0007.4 Run 10 feedback-driven generation/scoring iterations per requested section, preserving every candidate, two judge responses and iteration status; do not replace missing or failed model responses with invented scores.
- [ ] F-0007.5 Compare each best scored candidate with the fixed baseline, independently inspect scientific/style differences, and record what is accepted or rejected and why.
- [ ] F-0007.6 Apply selected revisions, preserve all existing comment threads and source/data bindings, complete the actual paper-writer and build checks, and deliver/read back the resulting authorized Overleaf revision.

- [ ] F-0007.7 Remove the model-table hyperlinks and AMC23 footnote and verify benchmark paper entries against official DBLP records; preserve honest contest provenance where there is no originating paper.
- [ ] F-0007.8 Trace Figure 6’s empty LFM Re row to its measured support/estimator, clarify undefined cells visibly, and verify that no zero is fabricated.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| Available requested model families | Current ELM /models response | Not checked for this task | gpt-6.1-sol available; no DeepSeek IDs among 59 listed models | Historical native ELM preflight, 2026-10-22; superseded by author-approved two-Luna configuration |
| Initial generator proposals | Eight main-section units | Fixed Overleaf9ff6d88 | Eight actual gpt-6.1-sol medium outputs; zero judged iterations or adoptions | elm-optimization/round-01-generator-results.jsonl; FIRST_PROPOSALS.json |
| Actual initial token use | Eight successful ELM generator calls | No calls for this task | 372092 input, 25284 output including 3780 reasoning tokens | Raw response usage and RUN_STATUS.json |
| Preliminary protected-source review | Eight first proposals | All fixed baseline values/equations/citations | No original values/citation keys removed; additions flagged in six sections, new inline math in Method, one PRO-050 fragment diagnostic; zero adoptions | FIRST_PROPOSALS.json; PRELIMINARY_SOURCE_CHECK.json; per-section diffs |

Author amendments on 2026-10-23: “run it now. BUt insetad of 3 deepseek use 2 6.1 luna with high reasoning using ELM”; “Also remove the hyperlinks from tab 2 and the footnote for MAC23. All benchmakr should use the offical dblp cit of their paper”; “Regarding Figure 6 check how cime the Re row of LFM is empty”. The three-DeepSeek configuration is superseded by the author, not silently substituted.

Progress observation 2026-10-23: five complete valid rounds in all eight main sections; round six judging is running. No candidate has yet replaced manuscript prose. RUN_STATUS.json and per-round requests/results are authoritative. RUBRIC_CORRECTION.json preserves the superseded placeholder-review attempt and excludes it from iteration counts; interrupted round-three startup usage is unknown and disclosed separately. Benchmark bibliography verification passes (43 OK, five existing accepted exceptions); model links and AMC23 footnote are removed locally. FIGURE6_AUDIT.json binds the unsupported AMC23 Read-origin adjacency, unchanged data and successful standalone n/e render. Final adoption, integrated review and Overleaf delivery remain pending.
