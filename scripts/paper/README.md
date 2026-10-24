# scripts/paper

Turn finished receipts into the paper's tables and figure data, build the PDF, and check the writing rules. The manuscript is `paper/6a510c6db0d9b08aa438c142`, a separate Overleaf checkout that holds sources only. Nothing here pushes to Overleaf.

| Script | What it does |
| --- | --- |
| `export.sh [EFFECTS_WHOLE.json]` | Writes the routing tables, the whole-reasoning tables and the plotting data for the figures. It reads sealed receipts and fits nothing. Without an argument it uses `report/whole-reasoning-20261021-r20/analysis-combined/EFFECTS_WHOLE.json`, the final whole-reasoning receipt. |
| `build.sh` | Compiles the figure sources, then the manuscript, with the offline `tectonic` (a few minutes). Use it instead of `pw build`, which falls back to a system LaTeX and shows an old PDF here. |
| `check.sh` | Runs the paper-writer rule check on the manuscript. The rules are in `paper/manuscript-support/notes/paper-rules.md`. |

```bash
scripts/paper/export.sh
scripts/paper/build.sh
scripts/paper/check.sh
```

Never edit the exported tables by hand. `export.sh` runs `paper.export_routing_results`, `paper.export_whole_results` and `paper.export_cl_exhibits`; what each one writes, which receipt feeds which table or figure, and the other checks (`paper.verify_cl_tables` and friends) are in [`src/paper`](../../src/paper/README.md).
