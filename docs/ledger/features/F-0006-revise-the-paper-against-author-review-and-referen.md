# F-0006: Revise the paper against author review and reference style
- status: active
- date: 2026-10-22
- request: "$paper-writer Go on then, fix the comments to align the style of the paper to my desires and previous paper references"
- purpose: Implement the manuscript revisions recorded by F-0005, using the author's full reference sections and preserving their control over comment resolution.
- acceptance: Every accessible author comment is mapped to a verified edit or an explicit remaining scientific task; revised sections receive independent scientific and reference-calibrated style reviews plus a Humanizer pass; the current native figures and integrated pages are inspected; source checks pass before the scoped Overleaf delivery; explanatory replies follow delivery without resolving or deleting any original thread.
- keywords: paper, Overleaf, review, comments, style, Humanizer, TikZ, next-state
- paths: paper/manuscript-support, src/paper, scripts/paper
- decisions: D-0004, D-0005

## Tasks

- [x] F-0006.1 Pull and reconcile all current comments, preserving the author revision and baseline evidence; evidence: Overleaf bb9fb04e734d50ac7895b76b2d961ea9cc060fb3 already current, review-revision/BASELINE.json and 56 current open threads.
- [x] F-0006.2 Revise abstract, Introduction and scientific narrative against the author's full source examples, without unsupported mechanisms or efficiency claims.; evidence: review-revision/independent-style-review.md, scientific-review-final.md and HUMANIZER.json; manuscript commit dcfc36e.
- [x] F-0006.3 Revise Method and Setup, introducing classification before conditioned selection and moving essential supporting details to the appendix.; evidence: method-round2-review.md, method-round2-verification.json and scientific-review-final.md; manuscript commit dcfc36e.
- [x] F-0006.4 Implement diagram, legend, clipping and table requests without changing measured cells or original PDF backups.; evidence: VISUAL_CELL_CHECKS.json, DOSE_GEOMETRY.json, FIDELITY.json and final visual verdicts; 35 compact data files and 11 original PDFs unchanged.
- [x] F-0006.5 Audit the saved next-state analyses and inputs for all eight models; keep missing results pending and specify the additional workload needed.; evidence: science-audit.md; two-model completed estimates retained, reusable saved inputs identified, R51 allocated CPU analysis remains pending.
- [x] F-0006.6 Complete independent scientific/style reviews, Humanizer diagnosis and revision, source checks, compilation and visual inspection.; evidence: three complete packets ending source-round-61; 58-page clean bounded build, zero raw-source violations; 262 judgments with 233 PASS, 27 N/A and explicit TAB-022/TAB-026 failures, zero invalid evidence records. This is a completed bounded review, not full rule compliance.
- [x] F-0006.7 Deliver the authorized Overleaf revision, read back the remote revision, and add concrete explanatory replies while preserving every original thread and resolution state.; evidence: review-revision/DELIVERY.json, REPLY_POSTS.json and COMMENTS_AFTER_DELIVERY.json; Overleaf 9ff6d88 read back, 55 implemented replies plus one pending-analysis reply, all 56 originals and open states preserved.

- [ ] F-0006.8 Complete the author-requested R51 next-state extension after a claim-linked CPU plan, measured budget and workload authorization; preserve the current two-model result until real additional estimates exist.
- [ ] F-0006.9 Complete the remaining TAB-022 printed numerical readback and TAB-026 ascending-size model order; both remain explicit after the bounded third review round.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| Implemented review items | 56 author Overleaf threads | 0 manuscript fixes in preceding TODO-only pass | 55 delivered; R51 analysis pending | review-revision/DELIVERY.json; REVIEW_TODOS.md at 9ff6d88 |
| Original discussions retained | 56 threads and 56 original messages | 56 open threads | 56 open threads, 112 messages including 56 read-back replies | COMMENTS_AFTER_DELIVERY.json; REPLY_POSTS.json |
| Source/data fidelity | 35 Results interval triples, 35 data files and 11 PDF backups | bb9fb04 source bindings | All preserved | FIDELITY.json |
| Compilation and rule check | source-round-61 | Not a new baseline build | 58 pages, compiler exit 0, zero raw violations | build/BUILD.json; DELIVERY.json |
| Judge verification | 262 applicable packet items | Initial substantive Method style FAIL | 233 PASS, 27 N/A, 2 explicit FAIL; 0 invalid | source-round-61/.paper-review/verdicts.md |
