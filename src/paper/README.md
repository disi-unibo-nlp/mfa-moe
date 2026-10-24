# src/paper

Turns finished receipts into the tables and figure data of the manuscript, builds the PDF, and checks it. The manuscript is `paper/6a510c6db0d9b08aa438c142`, an Overleaf checkout: it holds the text, the tables they input, the figures and their data, and nothing else. Notes, build output and old report sources are in `paper/manuscript-support` (all locations are named once in `cl_paths.py`). Nothing here pushes to Overleaf. Launchers: [`scripts/paper`](../../scripts/paper/README.md).

Model order: tables, figure data and figure ticks list the models by ascending total parameters (GPT-OSS, LFM2.5, Gemma, Nemotron, GLM, Qwen3-30B, Qwen3.5, Qwen3.6; equal sizes keep the order of `store.MODELS`). `cl_models` defines this order once, with the plot labels. Receipts are still checked against `store.MODELS`, so the display order never decides which models a receipt must cover.

## Write the tables and figure data

No fitting or resampling happens in these three. They read sealed receipts, so a number in the paper can be traced to one.

| Module | Reads | Writes (in the paper folder) |
| --- | --- | --- |
| `export_routing_results` | `report/final-routing-20261015-r1/driver/`: `analysis-final/EFFECTS.json`, `analysis-strong/EFFECTS.json`, `EFFICIENCY_LINK_PREREG.json`, `EFFICIENCY_LINK_V2.json`, the link test | `routing.dat`, `routing_results.json`, figure `routing.tex`, tables `routing`, `routing_levels`, `routing_strong`, `efficiency_link` |
| `export_whole_results` | one `EFFECTS_WHOLE.json` (default: `report/whole-reasoning-20261021-r20/analysis-combined/`) and the `continuations.parquet` beside it; `--effects FILE`, `--out DIR` | tables `whole`, `whole_cells`, `whole_occupancy`, `whole_levels`; figures `whole_dose` and `whole_states` with their data; `whole_results.json` |
| `export_cl_exhibits` | the bundled copies of completed estimates in `figures/journal/data` (readout, selection and anticipation CSVs, native dynamics, benchmark cells) | `readout.dat`, `selection.dat`, `selection_summary.json`, `next_state.dat`, `occupancy.dat`, `state_transitions_*.dat`. `--capture-lfm` first adds the LFM2.5 rows from `report/final-routing-20261015-r1`. |

The tables `readout`, `selection`, `selection_folds`, `occupancy`, `state_transitions`, `next_state`, `benchmarks` and `benchmark_details` are edited by hand in the manuscript. `verify_cl_tables` checks every table cell against the accepted manuscript revision (`ACCEPTED_REVISION` in `verify_cl_printed_tables`), the benchmark tables against `benchmark_cells.csv` and the routing tables against the saved routing estimates. A new whole-reasoning receipt changes the `whole` cells, so the accepted revision must then be moved to a manuscript commit that holds them.

Which experiment feeds which exhibit:

| Exhibit | Comes from |
| --- | --- |
| Figure `readout`, table `readout` | `routing.readout`, `routing.lfm readout` (`routing_readout.csv`) |
| Figure `selection`, tables `selection`, `selection_folds` | `routing.selection export`, `routing.lfm derive` (`selection_coverage.csv`) |
| Figures `occupancy`, `state_transitions`, `transition_summary`; tables `occupancy`, `state_transitions` | `routing.dynamics`, `routing.lfm dynamics` (`native_dynamics.json`) |
| Figure `next_state`, table `next_state` | `routing.anticipation` (`anticipation.csv`) |
| Tables `benchmarks`, `benchmark_details` | `routing.benchmark_accuracy`, `routing.benchmark_inventory` (`benchmark_cells.csv`) |
| Figure `routing`; tables `routing`, `routing_levels`, `routing_strong`, `efficiency_link` | `steering.pulse_effects`, `routing.efficiency_link` |
| Figures `whole_states`, `whole_dose`; tables `whole`, `whole_cells`, `whole_occupancy`, `whole_levels` | `steering.whole_effects combine` |

## Build and check

```bash
P=/leonardo/home/userexternal/lmolfett/.cache/paper-writer/venv/bin/python        # has pymupdf
PYTHONPATH=src $P -m paper.build_cl_figures [NAME ...] [--timeout-seconds 120]     # figure sources -> one-page PDFs
PYTHONPATH=src $P -m paper.build_cl_manuscript [--timeout-seconds 300]             # up to four passes of the offline tectonic
```

`scripts/paper/build.sh` runs both. Output goes to `paper/manuscript-support/build` (the PDF, one image per page, a build receipt); the manuscript folder is only read, and the build stops if a source changes while it runs. The first pass reuses the `aux` and `bbl` files of the newest finished build whose references, class and bibliography style are unchanged, even if the latest build failed.

```bash
PYTHONPATH=src $P -m paper.verify_cl_printed_tables --paper paper/6a510c6db0d9b08aa438c142 \
  --build paper/manuscript-support/build/BUILD.json --output paper/manuscript-support/build/PRINTED_TABLE_READBACK.json
```

| Module | What it does |
| --- | --- |
| `verify_cl_tables` | Checks the table cells and rank marks against the accepted manuscript revision and the saved estimates (`--self-test` also shows that it rejects swapped rows, changed intervals and wrong marks, and that the exporter writes only figure data). |
| `verify_cl_printed_tables` | Reads every table cell back from the PDF of a successful build: column headers, cell geometry, bold best and underlined second estimates, and that blank cells hold no measurement. It first compares the tables with the accepted revision, read from the git history of the Overleaf checkout (so does `verify_cl_tables`). Needs pymupdf. |
| `verify_cl_figure_labels` | Checks that the routing figure's PDF text matches its bars and interval strokes and that no label touches a bar. The bars are matched to `routing.dat` by their x coordinate, so the source order of the labels may differ. |
| `verify_cl_author_style` | Checks the author's word exclusions for the abstract and headings. |
| `verify_cl_bibliography_offline` | Compares the citations with independently inspected public records, offline. |
| `format_cl_sentences` | Puts one sentence on each source line without changing prose tokens (`--check` only reports). |
| `package_cl_manuscript` | Zips the sources and checks that the unpacked copy builds the same PDF. |
| `prepare_cl_review` | Adds figure sources, data and notes to a paper-writer review packet. |
| `optimize_sections` | Runs a previously approved section-revision protocol through the ELM batch client: two independent counterbalanced reviews per proposal, ten rounds, all retained. It never edits, commits or pushes the manuscript. |

```bash
python -m paper.optimize_sections --run-dir paper/manuscript-support/build/elm-optimization \
  --client ~/.paper-writer-skill/skills/deep-research/scripts/deep-research
```

The run directory must hold the protocol, the section baselines and author examples, the first generator requests and the judge rubric. Requests and responses are immutable evidence; a rerun resumes. Credentials stay with the client and never enter the run folder.

Tests: [`tests/paper`](../../tests/paper/README.md).

<!-- ledger:begin -->
## Ledger

Decisions and features that govern this folder (generated by `ledger sync`):

- [D-0004](../../docs/ledger/decisions/D-0004-keep-native-figure-sources-in-overleaf-with-origin.md) [active] Keep native figure sources in Overleaf with original PDF backups: chosen Canonical TikZ sources and plotting data live in the Overleaf checkout under figures/journal; the manuscript renders them directly. Original compiled PDFs stay byte-exact under figures/pdf-backup. Local builds use the canonical sources and write previews outside the checkout.; scope: Editable figures and local build/export tooling for the active CL Overleaf manuscript.; not: Scientific result changes, comment-driven redesigns, recomputation or frozen experiment releases.
- [F-0006](../../docs/ledger/features/F-0006-revise-the-paper-against-author-review-and-referen.md) [active 11/12] Revise the paper against author review and reference style
- [F-0009](../../docs/ledger/features/F-0009-lfm2-5-on-its-own-efficiency-link-signs.md) [active 5/7] LFM2.5 on its own efficiency-link signs
- [F-0012](../../docs/ledger/features/F-0012-every-paper-exhibit-on-all-eight-models.md) [active 1/8] Every paper exhibit on all eight models

<!-- ledger:end -->
