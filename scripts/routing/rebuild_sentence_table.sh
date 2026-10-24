#!/bin/bash
# Rebuild a model's labelled-sentence table on all of its question families from its saved traces and original judge
# labels (8 CPUs, about 1 minute) and print the job ID. Only Qwen3.6's saved table is partial; label_next_sentences.sh
# submits this job first for it:
#   scripts/routing/rebuild_sentence_table.sh STUDY_DIR HF_MODEL
set -euo pipefail
test -e "${1:?STUDY_DIR}/BINDING.json"
sbatch --parsable -J pairs-table -p boost_usr_prod -c 8 --mem=40G -t 01:00:00 scripts/slurm/run_module_cpu.sbatch routing.labelled_sentence_table rebuild --study "$1" --model "${2:?HF_MODEL}"
