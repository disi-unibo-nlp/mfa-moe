#!/bin/bash
# Pick the sentences to label (cohorts A and B, four parts) from the finished traces of one model; submits a CPU job.
# The model is read from the traces. OUT_DIR must not exist: cohorts are never overwritten.
#   scripts/labeling/choose_sentences.sh TRACES_DIR OUT_DIR
# TRACES_DIR holds <dataset>/traces.jsonl, e.g. .../card-v3/gpt/generation/openai--gpt-oss-20b.
test -d "${1:?TRACES_DIR}" || { echo "no such folder: $1" >&2; exit 1; }
test ! -e "${2:?OUT_DIR}" || { echo "already exists: $2" >&2; exit 1; }
sbatch -J choose-sentences scripts/slurm/run_module_cpu.sbatch labeling.choose_sentences --traces-dir "$1" --out "$2"
