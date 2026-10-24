# src/paper

Turns finished receipts into the tables and figure data of the manuscript, builds the PDF, and checks it. The manuscript is `paper/6a510c6db0d9b08aa438c142`, an Overleaf checkout: it holds the text, the tables they input, the figures and their data, and nothing else. Notes, build output and old report sources are in `paper/manuscript-support` (all locations are named once in `cl_paths.py`). Nothing here pushes to Overleaf. Launchers: [`scripts/paper`](../../scripts/paper/README.md).

## Write the tables and figure data

No fitting or resampling happens in these three. They read sealed receipts, so a number in the paper can be traced to one.

| Module | Reads | Writes (in the paper folder) |
| --- | --- | --- |
| `export_routing_results` | `report/final-routing-20261015-r1/driver/`: `analysis-final/EFFECTS.json`, `analysis-strong/EFFECTS.json`, `EFFICIENCY_LINK_PREREG.json`, `EFFICIENCY_LINK_V2.json`, the link test | `routing.dat`, `routing_results.json`, figure `routing.tex`, tables `routing`, `routing_levels`, `routing_strong`, `efficiency_link` |
| `export_whole_results` | one `EFFECTS_WHOLE.json` (default: `report/whole-reasoning-20261021-r20/analysis-combined/`) and the `continuations.parquet` beside it; `--effects FILE`, `--out DIR` | tables `whole`, `whole_cells`, `whole_occupancy`, `whole_levels`; figures `whole_dose` and `whole_states` with their data; `whole_results.json` |
| `export_cl_exhibits` | the bundled copies of completed estimates in `figures/journal/data` (readout, selection and anticipation CSVs, native dynamics, benchmark cells) | `readout.dat`, `selection.dat`, `selection_summary.json`, `next_state.dat`, `occupancy.dat`, `state_transitions_*.dat`. `--capture-lfm` first adds the LFM2.5 rows from `report/final-routing-20261015-r1`. |

The tables `readout`, `selection`, `selection_folds`, `occupancy`, `state_transitions`, `next_state`, `benchmarks` and `benchmark_details` are edited by hand in the manuscript. `verify_cl_tables` checks their cells and rank marks against the same bundled copies.

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

`scripts/paper/build.sh` runs both. Output goes to `paper/manuscript-support/build` (the PDF, one image per page, a build receipt); the manuscript folder is only read, and the build stops if a source changes while it runs.

| Module | What it does |
| --- | --- |
| `verify_cl_tables` | Checks the table cells and rank marks against the saved estimates (`--self-test` also shows that it rejects swapped rows, changed intervals and wrong marks). |
| `verify_cl_figure_labels` | Checks that the routing figure's PDF text matches its bars and interval strokes and that no label touches a bar. |
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
