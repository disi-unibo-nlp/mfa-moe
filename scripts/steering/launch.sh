#!/bin/bash
# Submit the jobs of a sealed whole-reasoning style release (scripts/steering/whole.sh seals one first):
#   scripts/steering/launch.sh NAME [JOB...]
# Unit tests, then the manifests, then the GPU shards of every model, one extraction per model, the final analysis
# (the combine, or for an outcome release its own analysis) and the staged paper export. JOB...: running jobs the
# final analysis also waits for. DRY_RUN=1 prints the sbatch commands and submits nothing. A launch that fails
# half-way cancels the jobs it already queued.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
unset $(compgen -v | grep -E '^(SLURM|SBATCH|SRUN|SALLOC)_' || true)  # started from inside a job? hand none of its state on
name=$1; shift
export STUDY_DATA_ROOT=$PWD/report/final-routing-20261015-r1 PYTHONDONTWRITEBYTECODE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
R=$PWD/report/$name
test -z "${DRY_RUN:-}" -a -e "$R/JOBS.txt" && { echo "$name was launched already (see $R/JOBS.txt)" >&2; exit 1; }

# What to run: the snapshot, the final command, then "MODEL SHARDS" lines.
out=$(PYTHONPATH=src $PY -B -m steering.whole plan --release "$R")
mapfile -t plan <<< "$out"
final=(${plan[1]})
cd "${plan[0]}"  # the jobs run the sealed code, not the working tree

# A finished job needs no dependency (Slurm forgets old jobs, so its state could not be checked later).
wait_for=()
for job in "$@"; do
  case $(sacct -j $job -X -n -P -o State) in
    COMPLETED) ;;
    PENDING|RUNNING|COMPLETING) wait_for+=($job) ;;
    *) echo "job $job did not complete" >&2; exit 1 ;;
  esac
done

queued=()
trap 'test ${#queued[@]} -eq 0 || { echo "launch failed; cancelling ${queued[*]}" >&2; scancel "${queued[@]}"; rm -f "$R/JOBS.txt"; }' ERR
join() { local IFS=:; echo "$*"; }
cpu=(-p boost_usr_prod)  # the serial partition caps a user at 8 CPUs, 30 GB and 2 running jobs
S() {  # S NAME sbatch-options... script args...: submit one job; its ID (its name in a dry run) is left in $JOB
  local cmd=(sbatch --parsable --kill-on-invalid-dep=yes -J "$1" -o "$R/logs/%x-%j.out" "${@:2}")
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; JOB=$1; return; fi
  mkdir -p "$R/logs"
  JOB=$("${cmd[@]}")
  queued+=($JOB)
  echo "$1 $JOB ${cmd[*]}" >> "$R/JOBS.txt"
}

S whole-tests "${cpu[@]}" -c 4 --mem=24G -t 60 scripts/steering/tests.sh
gate=$JOB
if [ ${#plan[@]} -gt 2 ]; then  # a release without shards (a relink) publishes no manifests
  S whole-prepare "${cpu[@]}" --dependency=afterok:$gate -c 4 --mem=24G -t 60 \
    scripts/slurm/cpu.sh steering.whole prepare --release "$R"
  gate=$JOB  # it ran only because the tests passed
fi

extractions=()
for line in "${plan[@]:2}"; do
  read -r model shards <<< "$line"
  runs=()
  for ((i = 0; i < shards; i++)); do
    S whole-run-$model --dependency=afterok:$gate \
      scripts/steering/gpu.sh steering.whole run --release "$R" --model $model --shard $i/$shards
    runs+=($JOB)
  done
  S whole-extract-$model "${cpu[@]}" --dependency=afterok:$gate,afterany:$(join "${runs[@]}") -c 8 --mem=24G -t 90 \
    --export=ALL,MOE_EXP_SCORING_PROCESSES=8 scripts/slurm/cpu.sh steering.whole extract --release "$R" --model $model
  extractions+=($JOB)
done

dependency=afterok:$gate
test ${#extractions[@]} -eq 0 || dependency+=,afterany:$(join "${extractions[@]}")
test ${#wait_for[@]} -eq 0 || dependency+=,afterok:$(join "${wait_for[@]}")
S whole-${final[1]} "${cpu[@]}" --dependency=$dependency -c 8 --mem=96G -t 360 \
  scripts/slurm/cpu.sh "${final[@]}" --release "$R"
if [ ${final[0]} = steering.whole ]; then  # paper tables and figure data of the combined receipt, staged beside it
  S whole-export "${cpu[@]}" --dependency=afterok:$JOB -c 2 --mem=16G -t 60 \
    scripts/slurm/cpu.sh paper.export_whole_results --effects "$R/analysis-combined/EFFECTS_WHOLE.json" --out "$R/export"
fi
