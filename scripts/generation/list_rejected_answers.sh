#!/bin/bash
# List the answers that finished but that math_verify rejected, as the items the judge checks (a CPU job; the model is
# read from the traces):
#   scripts/generation/list_rejected_answers.sh TRACES_DIR ITEMS_JSONL
# TRACES_DIR holds <dataset>/traces.jsonl, e.g. .../card-v3/gpt/generation/openai--gpt-oss-20b. ITEMS_JSONL must not exist.
test -d "${1:?TRACES_DIR}" || { echo "no such folder: $1" >&2; exit 1; }
test ! -e "${2:?ITEMS_JSONL}" || { echo "already exists: $2" >&2; exit 1; }
sbatch -J list-rejected scripts/slurm/run_module_cpu.sbatch generation.judge_answers build-corpus \
  --traces-dir "$1" --out "$2"
