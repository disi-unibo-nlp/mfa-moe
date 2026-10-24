#!/bin/bash
# Queue the CPU chain of a launched capture release for the models you name: class usage, expert sets, signal table
# and attempt selection (steering.outcome derive), each after that model's extraction job. It runs from the release
# snapshot and writes SELECTION_R2.json, SIGNAL.json and ATTEMPT_SELECTION.json to report/NAME/MODEL/.
#   scripts/steering/outcome_derive.sh NAME MODEL...        DRY_RUN=1 prints the sbatch lines only
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
unset $(compgen -v | grep -E '^(SLURM|SBATCH|SRUN|SALLOC)_' || true)  # started from inside a job? hand none of its state on
name=$1; shift
R=$PWD/report/$name
mapfile -t plan < <(PYTHONPATH=src /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m steering.whole plan --release "$R")
for model in "$@"; do
  extract=$(awk -v job="${plan[1]}-extract-$model" '$1 == job { print $2 }' "$R/JOBS.txt")
  test -n "$extract" || { echo "$name has no queued extraction of $model (see $R/JOBS.txt)" >&2; exit 1; }
  cmd=(sbatch --parsable -J "${plan[1]}-derive-$model" -p boost_usr_prod -c 4 --mem=30G -t 04:00:00 --dependency=afterok:$extract
       --chdir="${plan[0]}" -o "$R/logs/%x-%j.out" --export=ALL,STUDY_DATA_ROOT=$PWD/report/final-routing-20261015-r1
       scripts/slurm/run_module_cpu.sbatch steering.outcome derive --release "$R" --models "$model" --workers 4)
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; else echo "${plan[1]}-derive-$model $("${cmd[@]}")" | tee -a "$R/DERIVE_JOBS.txt"; fi
done
