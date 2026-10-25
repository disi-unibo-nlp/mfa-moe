# paper: to do

## editable-overleaf-figures: Editable Overleaf figures and review TODOs
- status: done
- asked: "Now first pull the last version of the paper, then read all the comments I posted and record the todos to sovle them using the paperback writer skill. Finally also add to the overleaf repo the tikz source of plots and figures. So I can possibly change some stuff by my self manually"; approved with "Implement the plan." Amendment: "When addressing the comments, in the end, do not resolve/delete the comment. Add a subcommnt telling how u sovle it. And leave it there. So I can later judge and possibly solve it myself". (2026-10-22)
- why: Make the manuscript figures directly editable in Overleaf and preserve an auditable checklist of author feedback, leaving final comment resolution to the author.
- done when: Latest Overleaf revision preserved; all accessible review threads and project chat reconciled into actionable TODOs; all 11 native figure sources and required data compile inside the paper checkout; original PDFs retained byte-for-byte; current source-only manuscript renders inspected; scoped Overleaf delivery verified by remote revision. Inaccessible comments remain explicitly pending.
- decisions: figures-live-in-overleaf, author-resolves-comments
- code: src/paper/build_cl_figures.py, src/paper/build_cl_manuscript.py, src/paper/verify_cl_figure_labels.py, src/paper/cl_paths.py, scripts/paper/build_paper_pdf.sh
- [x] Pull the latest Overleaf paper without overwriting incoming author edits; evidence: git pull --ff-only advanced a01bfb6 to b189963cc39d03a373d753590a209a25c308d2ed, changing only main.tex, introduction.tex and setup.tex.
- [x] Read all author comments and chat, create thread-linked TODOs, and classify feedback with the paper-writer skill; evidence: overleaf-review/COMMENTS.json and TODO_AUDIT.json, 56/56 threads mapped to REVIEW_TODOS.md, zero chat messages, all fixes pending.
- [x] Record the author's reply-without-resolution instruction; evidence: author-resolves-comments, LOC-044, source-clone PRC-014 lint (254 rules, zero errors), and DELIVERY.json confirming zero comment-state or message writes.
- [x] Make all 11 TikZ sources and 35 data files canonical in Overleaf and retain all 11 original PDFs; evidence: source-migration/VERIFICATION.json and FIGURE_PIXEL_COMPARISON.json, original hashes preserved and every standalone render pixel-identical.
- [x] Update existing build, export and verification paths, keeping previews and receipts outside Overleaf; evidence: cl_paths.FIGURES points to the paper checkout, previews go to BUILD/figure-builds, and repo-ledger conduct reports changed code clean. TOOLING_SOURCE.json binds the local uncommitted tooling.
- [x] Build and inspect native figures and the source-only manuscript; evidence: offline NATIVE_BUILD_REVIEW.json, 57 pages, 11 observed native inputs, resolved references; all 11 standalone figures and manuscript pages 2/7/18/19/20/22/23/24/33/34/39 inspected. Routing geometry checks 108 labels and bars with zero overlaps. The 12 source-layout findings also exist in b189963; the old table checker reports its pre-existing source-roster mismatch, with every data/table byte preserved.
- [x] Commit and push only the scoped Overleaf changes and read back both source and comments; evidence: DELIVERY.json, remote bb9fb04e734d50ac7895b76b2d961ea9cc060fb3, clean paper checkout, 56 original open threads and all messages unchanged. No comment fix is claimed and no solved reply was sent for pending TODOs.

### Results

| Measure | Data | Before | After | Evidence |
|---|---|---|---|---|
| Comment coverage | October 22 Overleaf review | 56 open threads | 56 mapped TODOs; zero chat messages | overleaf-review/TODO_AUDIT.json |
| Comment preservation | Same 56 original threads after publication | All open | All threads, messages and resolution states unchanged | source-migration/DELIVERY.json |
| Source and backup preservation | Original eleven figures and compact source data | PDF-only Overleaf project | 11 native sources, 35 byte-identical data files, 11 byte-identical PDF backups | source-migration/VERIFICATION.json |
| Standalone appearance | Original compiled figure PDFs | Original 144-dpi renders | 11/11 native renders pixel-identical | source-migration/FIGURE_PIXEL_COMPARISON.json |
| Native manuscript build | Self-contained source-round-52 | Backup PDFs excluded | Offline exit 0; 57 pages; all references resolved | source-migration/NATIVE_BUILD_REVIEW.json |
| Source check regression | Pulled b189963 versus final native source | 12 pre-existing source-layout findings | Same 12, zero new findings | source-migration/BASELINE_PAPER_WRITER_CHECK.log and PAPER_WRITER_CHECK.log |
| Existing table checker | Unchanged eight-model source data | Checker expects its earlier source roster | Fails with benchmark source roster differs; no cells changed | source-migration/VERIFICATION.json |
| Publication | Existing Overleaf main branch | b189963 | bb9fb04e734d50ac7895b76b2d961ea9cc060fb3 read back remotely | source-migration/DELIVERY.json |

## author-review-revision: Revise the paper against author review and reference style
- status: active
- asked: "$paper-writer Go on then, fix the comments to align the style of the paper to my desires and previous paper references" (2026-10-22)
- why: Implement the manuscript revisions recorded by editable-overleaf-figures, using the author's full reference sections and preserving their control over comment resolution.
- done when: Every accessible author comment is mapped to a verified edit or an explicit remaining scientific task; revised sections receive independent scientific and reference-calibrated style reviews plus a Humanizer pass; the current native figures and integrated pages are inspected; source checks pass before the scoped Overleaf delivery; explanatory replies follow delivery without resolving or deleting any original thread.
- decisions: figures-live-in-overleaf, author-resolves-comments
- code: src/paper, scripts/paper
- [x] Pull and reconcile all current comments, preserving the author revision and baseline evidence; evidence: Overleaf bb9fb04e734d50ac7895b76b2d961ea9cc060fb3 already current, review-revision/BASELINE.json and 56 current open threads.
- [x] Revise abstract, Introduction and scientific narrative against the author's full source examples, without unsupported mechanisms or efficiency claims.; evidence: review-revision/independent-style-review.md, scientific-review-final.md and HUMANIZER.json; manuscript commit dcfc36e.
- [x] Revise Method and Setup, introducing classification before conditioned selection and moving essential supporting details to the appendix.; evidence: method-round2-review.md, method-round2-verification.json and scientific-review-final.md; manuscript commit dcfc36e.
- [x] Implement diagram, legend, clipping and table requests without changing measured cells or original PDF backups.; evidence: VISUAL_CELL_CHECKS.json, DOSE_GEOMETRY.json, FIDELITY.json and final visual verdicts; 35 compact data files and 11 original PDFs unchanged.
- [x] Audit the saved next-state analyses and inputs for all eight models; keep missing results pending and specify the additional workload needed.; evidence: science-audit.md; two-model completed estimates retained, reusable saved inputs identified, R51 allocated CPU analysis remains pending.
- [x] Complete independent scientific/style reviews, Humanizer diagnosis and revision, source checks, compilation and visual inspection.; evidence: three complete packets ending source-round-61; 58-page clean bounded build, zero raw-source violations; 262 judgments with 233 PASS, 27 N/A and explicit TAB-022/TAB-026 failures, zero invalid evidence records. This is a completed bounded review, not full rule compliance.
- [x] Deliver the authorized Overleaf revision, read back the remote revision, and add concrete explanatory replies while preserving every original thread and resolution state.; evidence: review-revision/DELIVERY.json, REPLY_POSTS.json and COMMENTS_AFTER_DELIVERY.json; Overleaf 9ff6d88 read back, 55 implemented replies plus one pending-analysis reply, all 56 originals and open states preserved.
- [ ] Complete the author-requested R51 next-state extension after a claim-linked CPU plan, measured budget and workload authorization; preserve the current two-model result until real additional estimates exist.
- [x] Complete TAB-022 printed numerical readback and TAB-026 ascending-size model order; evidence: editorial-closeout/PRINTED_TABLE_READBACK.json and ORDERED_TABLE_FIDELITY.json, source-round-74, 19 tables / 453 rows / 2,641 printed cells, all negative controls rejected; embedded tables and quota list included.
- [x] Close LAY-010 and all remaining visual short-tail findings; evidence: source-round-74 successful 55-page build, all pages visually inspected, zero undefined references, 107 style PASS / four N/A, native packet and visual verdict evidence in editorial-closeout. Pending r20 scientific integration remains separate.
- [x] Commit the completed editorial closeout locally in the Overleaf checkout after verification; evidence: Overleaf local c98ed00c46604c256056ff72481db97bd49b6fa4, clean checkout; remote main remains 4c9d7d4, no push. Remote delivery and explanatory replies await author approval; editorial-closeout/LOCAL_DELIVERY.json.
- [x] Remove both superseded historical steering designs and their references, retain the separate candidate-recognition qualification, and repair the verifier; evidence: local Overleaf c98ed00c46604c256056ff72481db97bd49b6fa4, source-round-74, 55 pages (58 before both removals; 57 before Qwen follow-up), zero undefined references; PREFIX_REMOVAL.json and LOCAL_DELIVERY.json. Strong-dose and transition exhibits preserved from the follow-up request.

### Results

| Measure | Data | Before | After | Evidence |
|---|---|---|---|---|
| Implemented review items | 56 author Overleaf threads | 0 manuscript fixes in preceding TODO-only pass | 55 delivered; R51 analysis pending | review-revision/DELIVERY.json; REVIEW_TODOS.md at 9ff6d88 |
| Original discussions retained | 56 threads and 56 original messages | 56 open threads | 56 open threads, 112 messages including 56 read-back replies | COMMENTS_AFTER_DELIVERY.json; REPLY_POSTS.json |
| Source/data fidelity | 35 Results interval triples, 35 data files and 11 PDF backups | bb9fb04 source bindings | All preserved | FIDELITY.json |
| Compilation and rule check | source-round-61 | Not a new baseline build | 58 pages, compiler exit 0, zero raw violations | build/BUILD.json; DELIVERY.json |
| Judge verification | 262 applicable packet items | Initial substantive Method style FAIL | 233 PASS, 27 N/A, 2 explicit FAIL; 0 invalid | source-round-61/.paper-review/verdicts.md |

## elm-section-optimization: Optimize manuscript sections with ELM scoring
- status: done
- asked: "once u are donoe, pull the latest paperbackwriter skill and run 10 optiimization ruounds of that ELm based scoring. Using sol 6.1 medium as generator via ELM and 3 deepseek models. Do 10 iter for each section. The u comapre the new candidate with the one u are working on now (the best one when u stop), and decide what to take from that proposed solution. Allways following paparbackwriter indiations." (2026-10-22)
- why: Explore section revisions through the requested generator and two independent Luna High judges, as amended by the author on October 23, then retain only independently justified improvements over the delivered author-comment revision.
- done when: Each requested section has 10 valid generation/scoring iterations with actual model IDs, raw outputs, scores and best-candidate selection recorded; candidates are compared with the fixed current baseline and useful changes are chosen with reasons; scientific values, citations, equations and scope survive; final source/build/reference/visual checks and authorized delivery are read back.
- decisions: figures-live-in-overleaf, author-resolves-comments
- code: src/paper/optimize_sections.py, tests/paper/test_paper_optimization.py
- [x] Finish the existing comment delivery, preserve its best current source as the fixed comparison baseline, and pull the latest paper-writer source.; evidence: Overleaf 9ff6d88a021cc04e98f41ee9a65e8dcf87148895 and review-revision/DELIVERY.json; skill pull reports already up to date at 9885c78; local five-file feedback patch preserved.
- [x] Assemble complete section-matched author examples, current paper rules, scientific invariants and reproducible scoring inputs for each section; preserve baseline versions.; evidence: elm-optimization/PROTOCOL.json, eight baseline/example/rule packets, LOC-049 and PAIRWISE_RUBRIC.md; REFERENCE_CONTEXT_AUDIT.json records the complete Related Work wrapper correction and supplemental Method examples. Each actual round has 16 counterbalanced judge calls. The active run covers the eight main section units; appendices are protected scientific source material, not optimization units.
- [x] Validate gpt-6.1-sol at medium effort and the two author-approved gpt-6-luna High judge calls; record service responses and token usage without credentials. Generator validation succeeded in all eight actual ELM calls. The author replaced the three DeepSeek judges with two Luna 6.1 High judges. Live ELM catalog lists gpt-6-luna but not gpt-6.1-luna; an exact Responses request returned 404 model_not_found. The author approved gpt-6-luna with “use gpt 6 luna” and “keep going”; All 16 first-round GPT-6 Luna High judgments are complete; seven output-limited attempts remain charged and were recovered with larger response allowances. Evidence: actual round-01 judge requests/results and completion records; RUN_STATUS.json.
- [x] Run 10 feedback-driven generation/scoring iterations per requested section, preserving every candidate, two judge responses and iteration status; do not replace missing or failed model responses with invented scores.; evidence: elm-optimization/COMPLETENESS.json verifies80 valid generator calls,160 valid judgments and10rounds for each of8sections; actual model IDs/efforts and counterbalanced mappings checked.
- [x] Compare each best scored candidate with the fixed baseline, independently inspect scientific/style differences, and record what is accepted or rejected and why.; evidence: ADOPTION.json, final-science-review.md, final-style-review.md, HUMANIZER.json and FINAL_REVIEW_CORRECTIONS.json bind independent baseline comparison and actual selected changes; prospective LFMscience remains R57.
- [x] Apply selected revisions, preserve all existing comment threads and source/data bindings, complete the actual paper-writer and build checks, and deliver/read back the resulting authorized Overleaf revision.; evidence: MANUSCRIPT_PUSH.json verifies3378cf5remote; COMMENT_DELIVERY.json verifies5newreplies,56openoriginalthreads,117totalmessages; source-round68build58pages and native266items(236PASS,26N/A,4open,0invalid). Three bounded correction rounds are complete; unresolved science/format checks remain in REVIEW_TODOS and are not claimed resolved.
- [x] Remove the model-table hyperlinks and AMC23 footnote and verify benchmark paper entries against official DBLP records; preserve honest contest provenance where there is no originating paper.; evidence: BENCHMARK_CITATIONS.json and native no-normalize verification43OK+5existingaccepted; links and footnote removed.
- [x] Trace Figure 6’s empty LFM Re row to its measured support/estimator, clarify undefined cells visibly, and verify that no zero is fabricated.; evidence: FIGURE6_AUDIT.json verifies AMC23zero Read-origin adjacentpairs, fixedsixbenchmark gate, explicitn/e and all35datafilesunchanged; balancedlinear0–.7chosenunderlatestauthorrequest.

### Results

| Measure | Data | Before | After | Evidence |
|---|---|---|---|---|
| Available requested model families | Current ELM /models response | Not checked for this task | gpt-6.1-sol available; no DeepSeek IDs among 59 listed models | Historical native ELM preflight, 2026-10-22; superseded by author-approved two-Luna configuration |
| Initial generator proposals | Eight main-section units | Fixed Overleaf9ff6d88 | Eight actual gpt-6.1-sol medium outputs; zero judged iterations or adoptions | elm-optimization/round-01-generator-results.jsonl; FIRST_PROPOSALS.json |
| Actual initial token use | Eight successful ELM generator calls | No calls for this task | 372092 input, 25284 output including 3780 reasoning tokens | Raw response usage and RUN_STATUS.json |
| Preliminary protected-source review | Eight first proposals | All fixed baseline values/equations/citations | No original values/citation keys removed; additions flagged in six sections, new inline math in Method, one PRO-050 fragment diagnostic; zero adoptions | FIRST_PROPOSALS.json; PRELIMINARY_SOURCE_CHECK.json; per-section diffs |
| Completed optimization | Eight main sections | 9ff6d88 | 10validrounds each;80generations and160judgments | COMPLETENESS.json |
| Final native review | Source-round68,58pages | Deliveredsource61 | 236PASS,26N/A,4open,0invalid | NATIVE_REVIEW.json |
| Overleaf delivery | Project6a510c6db0d9b08aa438c142 | 9ff6d88 | Manuscript 3378cf5 verified;56originalthreadsopen,117messages | MANUSCRIPT_PUSH.json;COMMENT_DELIVERY.json |

## finish-nonsteering-results: Every non-steering result in the paper, from receipts to prose
- status: active
- asked: "Finish the paper: every result except the steering ones" (writer and reviewer debate, folder mfa-moe/r/paper-finish, 2026-10-25); labels next-state-eight-models, qwen36-full-dynamics, next-sentence-labels
- why: The paper shows the next-state result for two models, Qwen3.6 on 509 families and transition cells marked n/e; every non-steering exhibit should cover the eight models from sealed receipts and read as the author writes.
- done when: Fig. 6, Fig. 7 and Tables 3 to 5 come from the exporter for eight models; every non-steering number matches its receipt; the writing rules report 0 violations; the paper-writer packet and the printed-table readback pass on the built PDF; manuscript and code are pushed.
- decisions: next-state-receipt-sign, exporter-owns-model-rows, added-labels-transitions-only
- code: src/paper/export_cl_exhibits.py, src/paper/verify_cl_tables.py, src/paper/verify_cl_printed_tables.py, src/paper/export_whole_results.py, tests/paper/test_next_state_export.py
- [x] Exporter step and figure and table sources for all eight models (next-state-eight-models); evidence: commits 3113e23, 0887ab3, 8602add; tests/paper/test_next_state_export.py
- [x] Qwen3.6 panel from the rebuilt receipt, 1,486 families (qwen36-full-dynamics); evidence: `qwen36_panel()` refuses any other family count
- [x] Completed transitions from the next-sentence labels (next-sentence-labels); result: 101.7 GPU-h of 200, jobs 62188584-62188641; evidence: driver/TRANSITION_PAIRS.read-verify-plan-explore-monitor.json
- [x] Rewrite Results, Method, Setup, Discussion, appendices and the abstract for the new design; evidence: Overleaf commits f432bd8, 70cc0ab
- [x] Sentence-level audit against the author's earlier papers; evidence: nine independent reviews, accepted edits in Overleaf 70cc0ab, `check_writing_rules.sh` 0 violations in 39 files
- [x] Number lineage of the external-annotation, probe, qualification and lexical numbers; result: 79 of 87 match, three corrected (LFM2.5 sampling statement, two probe margins), five exist only in the wiki; evidence: Overleaf 19009ba
- [ ] paper-writer packet and verify on the built PDF
- [ ] Merge the code branch into main and push both repositories

