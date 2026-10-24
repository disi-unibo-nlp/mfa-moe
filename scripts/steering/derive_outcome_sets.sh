#!/bin/bash
# Queue the CPU chain of a launched capture release for the models you name: class usage, expert sets, signal table
# and attempt selection (steering.outcome derive), each after that model's extraction job. It runs from the release
# snapshot and writes SELECTION_R2.json, SIGNAL.json and ATTEMPT_SELECTION.json to RELEASE_DIR/MODEL/.
#   scripts/steering/derive_outcome_sets.sh RELEASE_DIR HF_MODEL...        DRY_RUN=1 prints the sbatch lines only
# RELEASE_DIR is the capture release (report/NAME) after submit_release_jobs.sh queued it: JOBS.txt names its extractions.
set -Eeuo pipefail
R=$(realpath "${1:?RELEASE_DIR}"); shift
cd "$(dirname "$0")/../.."
unset $(compgen -v | grep -E '^(SLURM|SBATCH|SRUN|SALLOC)_' || true)  # started from inside a job? hand none of its state on
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
plan=$(PYTHONPATH=src $PY -B -m steering.whole_reasoning plan --release "$R")
snapshot=$(jq -r .release_root <<< "$plan")
prefix=$(jq -r .job_prefix <<< "$plan")
study=$(PYTHONPATH=src $PY -B -c 'from store import STUDY; print(STUDY)')
for model in "$@"; do
  key=$(jq -r --arg model "$model" '.models[] | select(.hf_model == $model or .model == $model) | .model' <<< "$plan")
  extract=$(test -z "$key" || awk -v job="$prefix-extract-$key" '$1 == job { print $2 }' "$R/JOBS.txt")
  test -n "$extract" || { echo "$R has no queued extraction of $model (see $R/JOBS.txt)" >&2; exit 1; }
  cmd=(sbatch --parsable --kill-on-invalid-dep=yes -J "$prefix-derive-$key" -p boost_usr_prod -c 4 --mem=30G -t 04:00:00
       --dependency=afterok:$extract --chdir="$snapshot" -o "$R/logs/%x-%j.out" --export=ALL,STUDY_DATA_ROOT="$study"
       scripts/slurm/run_module_cpu.sbatch steering.outcome derive --release "$R" --models "$key" --workers 4)
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; else echo "$prefix-derive-$key $("${cmd[@]}")" | tee -a "$R/DERIVE_JOBS.txt"; fi
done
