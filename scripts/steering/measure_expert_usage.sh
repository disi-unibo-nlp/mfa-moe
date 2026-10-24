#!/bin/bash
# How often each expert is used per reasoning token in every cohort-B attempt (CPU job, input of the outcome release):
#   scripts/steering/measure_expert_usage.sh REPORTS_DIR OUT_DIR HF_MODEL...
# REPORTS_DIR is the report/ folder that holds the base release whole-reasoning-20261021-r7 (its class nominations);
# OUT_DIR (it must not exist) receives MODEL/USAGE.json and MODEL/USAGE.npz. `seal_release.sh outcome` reads the
# usage of its models from steering.outcome.select.BULK, so a new OUT_DIR is for checking a measurement.
set -euo pipefail
reports=$(realpath "${1:?REPORTS_DIR}"); out=$(realpath -m "${2:?OUT_DIR}"); shift 2
test -d "$reports" || { echo "$reports is no folder" >&2; exit 1; }
test ! -e "$out" || { echo "$out exists: OUT_DIR must be new" >&2; exit 1; }
test $# -gt 0 || { echo "name at least one HF_MODEL" >&2; exit 1; }
sbatch -J os-usage -t 02:00:00 scripts/slurm/run_module_cpu.sbatch steering.outcome usage \
  --reports "$reports" --out "$out" --models "$(IFS=,; echo "$*")"
