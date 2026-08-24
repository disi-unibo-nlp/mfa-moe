# Historical report

This directory contains the combined manuscript and generated figures for the
archived numbered experiments:

- `legacy_full_report.tex`: original combined LaTeX source.
- `legacy_full_report.pdf`: last compiled version of that source.
- `paper_figures/`: generated figures and their generation script.
- `main.*`: retained build artifacts from the last legacy compilation.

The figure generator reads numbered outputs from `results/old/exp1` through
`results/old/exp5`. The LaTeX classes used by the manuscript remain in
`report/`, where they are also used by the current report.

The active report is `report/main.tex`; historical results should be edited
here instead of being added back to that entry point.
