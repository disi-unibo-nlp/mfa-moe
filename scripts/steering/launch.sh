#!/bin/bash
# Submit the jobs of a sealed whole-reasoning style release (scripts/steering/whole.sh seals one first):
#   scripts/steering/launch.sh NAME [JOB...]
# Unit tests and manifests, then the GPU shards of every model, one extraction per model, then the final analysis
# (the combine, or for an outcome release its own analysis). JOB...: running jobs the final analysis also waits for,
# e.g. the parent's combine for a relink. DRY_RUN=1 prints the sbatch commands and submits nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset $(compgen -v SLURM_)  # a launcher started from inside a job must not hand its Slurm state on
name=$1; shift
export STUDY_DATA_ROOT=$PWD/report/final-routing-20261015-r1 PYTHONDONTWRITEBYTECODE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
R=$PWD/report/$name
mapfile -t plan < <(PYTHONPATH=src $PY -B -m steering.whole plan --release "$R")  # the snapshot, then "MODEL SHARDS" lines
cd "${plan[0]}"  # the jobs run the sealed code, not the working tree

S() {  # S NAME sbatch-options... script args...: submit one job, print its ID (its name in a dry run)
  local cmd=(sbatch --parsable --kill-on-invalid-dep=yes -J "$1" -o "$R/logs/%x-%j.out" "${@:2}")
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; echo "$1"; return; fi
  mkdir -p "$R/logs"
  local id=$("${cmd[@]}")
  echo "$1 $id" >> "$R/JOBS.txt"
  echo "$id"
}
join() { local IFS=:; echo "$*"; }

gate=($(S whole-tests -c 4 --mem=24G -t 60 scripts/steering/tests.sh))
test -e "$R/RELINK.json" ||  # a relink publishes no manifests
  gate+=($(S whole-prepare -c 4 --mem=24G -t 60 scripts/slurm/cpu.sh steering.whole prepare --release "$R"))
ok=afterok:$(join "${gate[@]}")

extractions=()
for line in "${plan[@]:1}"; do
  read -r model shards <<< "$line"
  runs=()
  for ((i = 0; i < shards; i++)); do
    runs+=($(S whole-run-$model --dependency=$ok --gres=gpu:2 -c 16 --mem=120G -t 240 \
      scripts/steering/gpu.sh steering.whole run --release "$R" --model $model --shard $i/$shards))
  done
  extractions+=($(S whole-extract-$model --dependency=$ok,afterany:$(join "${runs[@]}") -c 8 --mem=24G -t 90 \
    --export=ALL,MOE_EXP_SCORING_PROCESSES=8 scripts/slurm/cpu.sh steering.whole extract --release "$R" --model $model))
done

dependency=$ok
test ${#extractions[@]} -eq 0 || dependency+=,afterany:$(join "${extractions[@]}")
for job in "$@"; do  # a finished job needs no dependency (Slurm forgets old jobs, so it could not be checked)
  case $(sacct -j $job -X -n -P -o State) in
    COMPLETED) ;;
    PENDING|RUNNING|COMPLETING) dependency+=,afterok:$job ;;
    *) echo "job $job did not complete" >&2; exit 1 ;;
  esac
done
if [ -e "$R/OUTCOME.json" ]; then final=(steering.outcome analyze); else final=(steering.whole combine); fi
S whole-${final[1]} --dependency=$dependency -c 8 --mem=64G -t 240 scripts/slurm/cpu.sh "${final[@]}" --release "$R"
