#!/bin/bash
# Build the figures and the manuscript PDF (offline tectonic, at most 45 s per pass; `pw build` renders a stale PDF here):
#   scripts/paper/build.sh
set -euo pipefail
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
PY=/leonardo/home/userexternal/lmolfett/.cache/paper-writer/venv/bin/python
$PY -m moe_exp.paper.build_cl_figures
$PY -m moe_exp.paper.build_cl_manuscript --bounded-passes
