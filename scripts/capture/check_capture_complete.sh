#!/bin/bash
# Check that a finished capture of one model and cohort is complete and intact (a CPU job that hashes every captured
# file) and write the acceptance file:
#   scripts/capture/check_capture_complete.sh MANIFEST_DIR OUT_JSON [CAPTURE_ROOT]
#   e.g.  .../campaign-v3/manifests/gpt/A acceptance/capture-gpt-A.json
# CAPTURE_ROOT is the tree of a repeat capture (ROOT/verify, ROOT/0, ...); without it the paper's folders are read.
test -d "${1:?MANIFEST_DIR}" || { echo "no such folder: $1" >&2; exit 1; }
: "${2:?OUT_JSON}"
sbatch -J check-capture scripts/slurm/run_module_cpu.sbatch capture.check_complete "$1" "$2" ${3:+--captures "$3"}
