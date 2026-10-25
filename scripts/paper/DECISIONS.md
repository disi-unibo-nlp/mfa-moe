# paper: decisions

## figures-live-in-overleaf: Keep native figure sources in Overleaf with original PDF backups
- status: active
- decided by: user (2026-10-22)
- applies to: Editable figures and local build/export tooling for the active CL Overleaf manuscript.
- not to: Scientific result changes, comment-driven redesigns, recomputation or frozen experiment releases.
- chosen: Canonical TikZ sources and plotting data live in the Overleaf checkout under figures/journal; the manuscript renders them directly. Original compiled PDFs stay byte-exact under figures/pdf-backup. Local builds use the canonical sources and write previews outside the checkout.
- rejected: PDF-only Overleaf delivery with editable sources kept solely in the sibling support folder: the author cannot edit the figures directly in the paper project.
- evidence: User: "yes render them directly, but make sure we reain the pdf versions in a folder of the repo for backup"; implementation approved with "Implement the plan." Overleaf commit bb9fb04e734d50ac7895b76b2d961ea9cc060fb3; source-migration/DELIVERY.json and FIGURE_PIXEL_COMPARISON.json verify eleven preserved PDFs and pixel-identical native renders.
- keywords: overleaf, tikz, figures, plots, editable, pdf backup

## next-state-receipt-sign: The next-state gain keeps the receipt's sign
- status: active
- decided by: user ("Its gain is the cross-entropy of routing minus text, so negative favours routing. Describe it as it is", 2026-10-25)
- applies to: the next-state table, Fig. 7, their captions, the equation and every sentence that quotes the gain or the alignment contrast
- not to: the present-state readout, whose RIG and alignment gain stay positive-favourable
- chosen: the anticipatory routing gain is routing minus text cross-entropy and the alignment contrast is routing minus the mean lexical-null cross-entropy, both in nats per adjacent pair, so negative values favour routing; Table 4 marks the lowest values; Fig. 7 hangs its bars below zero
- rejected: printing text minus routing so that larger is better, because it renames the receipt's quantity and the reviewer found the flip a presentation the receipt does not show
- evidence: report/final-routing-20261015-r1/driver/NEXT_STATE.json (routing_gain.P, routing_excess_over_null.P); `verify_next_state` in src/paper/verify_cl_tables.py checks the table, the plot data and the prose; Overleaf commit f432bd8
- keywords: next state, anticipatory routing gain, sign, alignment contrast

## exporter-owns-model-rows: The exporter writes the model rows of four non-steering tables
- status: active
- decided by: user ("Numbers enter the paper only through the exporters", 2026-10-25)
- applies to: the model rows of journal/tables/readout.tex, next_state.tex, occupancy.tex and state_transitions.tex, and the data files of the figures that read them
- not to: steering tables, whose exporters (`export_routing_results.py`, `export_whole_results.py`) run only with the user's go-ahead
- chosen: `python -m paper.export_cl_exhibits` (flags `--capture-next-state`, `--capture-qwen36-panel`, `--capture-pairs VARIANT`) writes the rows and the data files; `verify_cl_tables --self-test` runs the exporter into a copy and fails if a byte changes
- rejected: `scripts/paper/export_paper_tables.sh` on the manuscript as it stands, because its routing and whole-reasoning exporters switch the steering tables to the r20 numbers and the display model order
- evidence: src/paper/export_cl_exhibits.py, tests/paper/test_next_state_export.py (34 tests with tests/test_imports.py), `exporter_reproduces_manuscript` in src/paper/verify_cl_tables.py
- keywords: exporter, table rows, export_paper_tables, verify_cl_tables

## added-labels-transitions-only: Next-sentence labels enter the transition estimates only
- status: active
- decided by: user ("the new labels enter only transition estimates (occupancy and dwell keep the random sample)", 2026-10-23)
- applies to: the native-dynamics panels of Fig. 6 and Table 4 (transitions), their prose and Appendix E
- not to: occupancy, dwell, re-entry, coverage and the next-state classifiers, which use the original sample
- chosen: a label added for the unsampled successor of a sampled Read, Explore, Plan, Verify or Monitor sentence can be the destination of a transition; a pair whose source is an added label is excluded, so every source stays a draw of the original sample; the completed panel carries its own simultaneous critical value
- rejected: counting added labels as sources or in occupancy, because the sampled sentences would no longer be a draw of the original design
- evidence: report/final-routing-20261015-r1/driver/TRANSITION_PAIRS.read-verify-plan-explore-monitor.json; `routing.next_sentence_labels.completed_transitions`; `capture_pairs` in src/paper/export_cl_exhibits.py
- keywords: next sentence labels, completed transitions, added labels, transition pairs

