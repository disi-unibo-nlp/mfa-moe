#!/bin/bash
# Build the figures and the manuscript PDF with the offline tectonic (a few minutes; `pw build` renders a stale PDF here):
#   scripts/paper/build.sh
set -euo pipefail
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
PY=/leonardo/home/userexternal/lmolfett/.cache/paper-writer/venv/bin/python
$PY -m paper.build_cl_figures
$PY -m paper.build_cl_manuscript
