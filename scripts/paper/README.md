# scripts/paper

Turn finished receipts into the paper's tables and figure data, build the PDF, and check the writing rules. The manuscript is `paper/6a510c6db0d9b08aa438c142`, a separate Overleaf checkout that holds sources only. Nothing here pushes to Overleaf.

| Script | What it does |
| --- | --- |
| `export_paper_tables.sh STUDY_DIR [EFFECTS_WHOLE_JSON]` | Writes the routing tables, the whole-reasoning tables and the plotting data for the figures. It reads sealed receipts and fits nothing. `STUDY_DIR` is the routing study (`report/final-routing-20261015-r1`). Without a second argument it uses `report/whole-reasoning-20261021-r20/analysis-combined/EFFECTS_WHOLE.json`, the final whole-reasoning receipt. |
| `build_paper_pdf.sh` | Compiles the figure sources, then the manuscript, with the offline `tectonic` (a few minutes). Use it instead of `pw build`, which falls back to a system LaTeX and shows an old PDF here. |
| `check_writing_rules.sh` | Runs the paper-writer rule check on the manuscript. The rules are in `paper/manuscript-support/notes/paper-rules.md`. |

```bash
scripts/paper/export_paper_tables.sh report/final-routing-20261015-r1
scripts/paper/build_paper_pdf.sh
scripts/paper/check_writing_rules.sh
```

Never edit the exported tables by hand. `export_paper_tables.sh` runs `paper.export_routing_results`, `paper.export_whole_results` and `paper.export_cl_exhibits`; what each one writes, which receipt feeds which table or figure, and the other checks (`paper.verify_cl_tables`, `paper.verify_cl_printed_tables` and friends) are in [`src/paper`](../../src/paper/README.md).
