#!/bin/bash
# Submit the jobs of a sealed whole-reasoning style release (scripts/steering/whole.sh seals one first):
#   scripts/steering/launch.sh NAME [JOB...]
# Unit tests, then the manifests, then the GPU shards of every model, one extraction per model, the final analysis
# (the combine, or for an outcome release its own analysis; a capture release has none) and the staged paper export.
# Job names start with the release's job_prefix (whole; oc for a capture, os for an outcome sweep), so scripts that
# wait on whole-* jobs never see the others. JOB...: running jobs the final analysis also waits for. The time
# limits of all the jobs must fit the ceilings sealed in the release, or nothing is submitted. DRY_RUN=1 prints the
# sbatch commands and submits nothing. A launch that fails or is interrupted half-way cancels the jobs it already
# queued and moves their IDs to JOBS.cancelled-TIME.txt; the release can be launched again.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
repo=$PWD  # the checkout this runs from: it holds report/, which the release snapshot does not
unset $(compgen -v | grep -E '^(SLURM|SBATCH|SRUN|SALLOC)_' || true)  # started from inside a job? hand none of its state on
name=$1; shift
export STUDY_DATA_ROOT=$PWD/report/final-routing-20261015-r1 PYTHONDONTWRITEBYTECODE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
R=$PWD/report/$name
test -z "${DRY_RUN:-}" -a -e "$R/JOBS.txt" && { echo "$name was launched already (see $R/JOBS.txt)" >&2; exit 1; }

# What to run: the snapshot, the job-name prefix, the final command (empty: none), then "MODEL SHARDS" lines.
out=$(PYTHONPATH=src $PY -B -m steering.whole plan --release "$R")
mapfile -t plan <<< "$out"
prefix=${plan[1]}
final=(${plan[2]})

# A finished job needs no dependency (Slurm forgets old jobs, so its state could not be checked later).
wait_for=()
for job in "$@"; do
  case $(sacct -j $job -X -n -P -o State) in
    COMPLETED) ;;
    PENDING|RUNNING|COMPLETING) wait_for+=($job) ;;
    *) echo "job $job did not complete" >&2; exit 1 ;;
  esac
done

queued=() cpu_minutes=0 gpu_minutes=0 counting=
join() { local IFS=:; echo "$*"; }
cpu=(-p boost_usr_prod)  # the serial partition caps a user at 8 CPUs, 30 GB and 2 running jobs
S() {  # S NAME CORES MINUTES GPUS sbatch-options... script args...: submit one job; its ID (its name in a dry run) is left in $JOB
  local cmd=(sbatch --parsable --kill-on-invalid-dep=yes -J "$1" -o "$R/logs/%x-%j.out" -c $2 -t $3)
  test $4 -eq 0 || cmd+=(--gres=gpu:$4)
  cmd+=("${@:5}")
  JOB=$1
  if [ -n "$counting" ]; then  # only add up the time limits
    if [ $4 -eq 0 ]; then cpu_minutes=$((cpu_minutes + $2 * $3)); else gpu_minutes=$((gpu_minutes + $4 * $3)); fi
    return 0
  fi
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; return 0; fi
  mkdir -p "$R/logs"
  JOB=$("${cmd[@]}")
  queued+=($JOB)
  echo "$1 $JOB ${cmd[*]}" >> "$R/JOBS.txt"
}

graph() {  # every job of the release, in order
  S $prefix-tests 4 60 0 "${cpu[@]}" --mem=24G scripts/slurm/run_module_cpu.sbatch steering.whole validate --release "$R"
  local gate=$JOB
  if [ ${#plan[@]} -gt 3 ]; then  # a release without shards (a relink) publishes no manifests
    S $prefix-prepare 4 60 0 "${cpu[@]}" --dependency=afterok:$gate --mem=24G \
      scripts/slurm/run_module_cpu.sbatch steering.whole prepare --release "$R"
    gate=$JOB  # it ran only because the tests passed
  fi

  local extractions=() line model shards runs i
  for line in "${plan[@]:3}"; do
    read -r model shards <<< "$line"
    runs=()
    for ((i = 0; i < shards; i++)); do  # a shard is 2 GPUs for 4 hours
      S $prefix-run-$model 16 240 2 --dependency=afterok:$gate \
        scripts/slurm/run_module_gpu.sbatch steering.whole run --release "$R" --model $model --shard $i/$shards
      runs+=($JOB)
    done
    S $prefix-extract-$model 8 90 0 "${cpu[@]}" --dependency=afterok:$gate,afterany:$(join "${runs[@]}") --mem=24G \
      --export=ALL,MOE_EXP_SCORING_PROCESSES=8 scripts/slurm/run_module_cpu.sbatch steering.whole extract --release "$R" --model $model
    extractions+=($JOB)
  done

  test ${#final[@]} -gt 0 || return 0  # a capture release ends with its extractions
  local dependency=afterok:$gate memory=96G
  test ${#extractions[@]} -eq 0 || dependency+=,afterany:$(join "${extractions[@]}")
  test ${#wait_for[@]} -eq 0 || dependency+=,afterok:$(join "${wait_for[@]}")
  test ${final[0]} = steering.whole || memory=64G
  S $prefix-${final[1]} 8 120 0 "${cpu[@]}" --dependency=$dependency --mem=$memory \
    scripts/slurm/run_module_cpu.sbatch "${final[@]}" --release "$R"
  if [ ${final[0]} = steering.whole ]; then
    # Paper tables and figure data of the combined receipt, staged beside it. Run from this checkout: the snapshot
    # has no report/ folder, and the exporter reads the efficiency-link receipt from there.
    S $prefix-export 2 60 0 "${cpu[@]}" --dependency=afterok:$JOB --mem=16G --chdir="$repo" \
      "$repo/scripts/slurm/run_module_cpu.sbatch" paper.export_whole_results --effects "$R/analysis-combined/EFFECTS_WHOLE.json" --out "$R/export"
  fi
}

# The time limits first: nothing is submitted when they exceed the ceilings of the release.
counting=1; graph; counting=
PYTHONPATH=src $PY -B -m steering.whole budget --release "$R" \
  --cpu-hours $(awk -v m=$cpu_minutes 'BEGIN { print m / 60 }') --gpu-hours $(awk -v m=$gpu_minutes 'BEGIN { print m / 60 }')

cancel() {  # failed or interrupted: cancel what is queued, keep the record of it, and fail
  local status=$?
  test $BASHPID = $$ || exit $status  # a command substitution failed: the shell that started it cancels
  trap - ERR INT TERM
  if [ ${#queued[@]} -gt 0 ]; then
    echo "launch failed; cancelling ${queued[*]}" >&2
    scancel "${queued[@]}" || true
    test ! -e "$R/JOBS.txt" || mv "$R/JOBS.txt" "$R/JOBS.cancelled-$(date +%Y%m%dT%H%M%S).txt"
  fi
  exit $((status == 0 ? 1 : status))
}
trap cancel ERR INT TERM
cd "${plan[0]}"  # the jobs run the sealed code, not the working tree
graph
