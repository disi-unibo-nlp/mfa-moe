# F-0005: Editable Overleaf figures and review TODOs
- status: done
- date: 2026-10-22
- request: "Now first pull the last version of the paper, then read all the comments I posted and record the todos to sovle them using the paperback writer skill. Finally also add to the overleaf repo the tikz source of plots and figures. So I can possibly change some stuff by my self manually"; approved with "Implement the plan." Amendment: "When addressing the comments, in the end, do not resolve/delete the comment. Add a subcommnt telling how u sovle it. And leave it there. So I can later judge and possibly solve it myself".
- purpose: Make the manuscript figures directly editable in Overleaf and preserve an auditable checklist of author feedback, leaving final comment resolution to the author.
- acceptance: Latest Overleaf revision preserved; all accessible review threads and project chat reconciled into actionable TODOs; all 11 native figure sources and required data compile inside the paper checkout; original PDFs retained byte-for-byte; current source-only manuscript renders inspected; scoped Overleaf delivery verified by remote revision. Inaccessible comments remain explicitly pending.
- keywords: overleaf, tikz, figures, comments, todos, manuscript, paper-writer
- paths: src/paper/build_cl_figures.py, src/paper/build_cl_manuscript.py, src/paper/verify_cl_figure_labels.py, src/paper/cl_paths.py, scripts/paper/build.sh
- decisions: D-0004, D-0005

Evidence paths without a repository prefix are under paper/manuscript-support/build/. The Overleaf checkout is a separate Git repository; its published commit is recorded below. Local tooling changes remain uncommitted and source-bound in TOOLING_SOURCE.json.

## Tasks

- [x] F-0005.1 Pull the latest Overleaf paper without overwriting incoming author edits; evidence: git pull --ff-only advanced a01bfb6 to b189963cc39d03a373d753590a209a25c308d2ed, changing only main.tex, introduction.tex and setup.tex.
- [x] F-0005.2 Read all author comments and chat, create thread-linked TODOs, and classify feedback with the paper-writer skill; evidence: overleaf-review/COMMENTS.json and TODO_AUDIT.json, 56/56 threads mapped to REVIEW_TODOS.md, zero chat messages, all fixes pending.
- [x] F-0005.3 Record the author's reply-without-resolution instruction; evidence: D-0005, LOC-044, source-clone PRC-014 lint (254 rules, zero errors), and DELIVERY.json confirming zero comment-state or message writes.
- [x] F-0005.4 Make all 11 TikZ sources and 35 data files canonical in Overleaf and retain all 11 original PDFs; evidence: source-migration/VERIFICATION.json and FIGURE_PIXEL_COMPARISON.json, original hashes preserved and every standalone render pixel-identical.
- [x] F-0005.5 Update existing build, export and verification paths, keeping previews and receipts outside Overleaf; evidence: cl_paths.FIGURES points to the paper checkout, previews go to BUILD/figure-builds, and repo-ledger conduct reports changed code clean. TOOLING_SOURCE.json binds the local uncommitted tooling.
- [x] F-0005.6 Build and inspect native figures and the source-only manuscript; evidence: offline NATIVE_BUILD_REVIEW.json, 57 pages, 11 observed native inputs, resolved references; all 11 standalone figures and manuscript pages 2/7/18/19/20/22/23/24/33/34/39 inspected. Routing geometry checks 108 labels and bars with zero overlaps. The 12 source-layout findings also exist in b189963; the old table checker reports its pre-existing source-roster mismatch, with every data/table byte preserved.
- [x] F-0005.7 Commit and push only the scoped Overleaf changes and read back both source and comments; evidence: DELIVERY.json, remote bb9fb04e734d50ac7895b76b2d961ea9cc060fb3, clean paper checkout, 56 original open threads and all messages unchanged. No comment fix is claimed and no solved reply was sent for pending TODOs.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| Comment coverage | October 22 Overleaf review | 56 open threads | 56 mapped TODOs; zero chat messages | overleaf-review/TODO_AUDIT.json |
| Comment preservation | Same 56 original threads after publication | All open | All threads, messages and resolution states unchanged | source-migration/DELIVERY.json |
| Source and backup preservation | Original eleven figures and compact source data | PDF-only Overleaf project | 11 native sources, 35 byte-identical data files, 11 byte-identical PDF backups | source-migration/VERIFICATION.json |
| Standalone appearance | Original compiled figure PDFs | Original 144-dpi renders | 11/11 native renders pixel-identical | source-migration/FIGURE_PIXEL_COMPARISON.json |
| Native manuscript build | Self-contained source-round-52 | Backup PDFs excluded | Offline exit 0; 57 pages; all references resolved | source-migration/NATIVE_BUILD_REVIEW.json |
| Source check regression | Pulled b189963 versus final native source | 12 pre-existing source-layout findings | Same 12, zero new findings | source-migration/BASELINE_PAPER_WRITER_CHECK.log and PAPER_WRITER_CHECK.log |
| Existing table checker | Unchanged eight-model source data | Checker expects its earlier source roster | Fails with benchmark source roster differs; no cells changed | source-migration/VERIFICATION.json |
| Publication | Existing Overleaf main branch | b189963 | bb9fb04e734d50ac7895b76b2d961ea9cc060fb3 read back remotely | source-migration/DELIVERY.json |
