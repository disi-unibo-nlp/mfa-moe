#!/bin/bash
# Fig. 6 and the state-transitions table on all eight models: label the sentences that follow a Read or Verify
# sentence (or the SOURCE_CLASSES you name) and redo each native-dynamics panel with them. Per model it submits a plan
# (CPU), SHARDS judge jobs (2 GPUs, 4 hours each) and a summary (CPU); one last job joins the eight panels into
# report/final-routing-20261015-r1/driver/TRANSITION_PAIRS.<variant>.json. Qwen3.6 first rebuilds its sentence table,
# which covers only 509 of its 1,486 question families, and its original-sample panel.
#   scripts/routing/transition_pairs.sh [SOURCE_CLASSES [SHARDS]]       default: Read,Verify and 2 shards per model
#   scripts/routing/transition_pairs.sh Read,Verify,Plan,Explore,Monitor 3
# AFTER="JOB ..." names jobs the judges wait for. A wider rule reuses the labels of a narrower one (one label store per
# model), so its judges must not run while the narrower ones still do. DRY_RUN=1 prints the sbatch lines only.
set -euo pipefail
classes=${1:-Read,Verify}; shards=${2:-2}
models=(gpt nemotron qwen35 qwen36 gemma glm qwen330b lfm25)
cpu=(-p boost_usr_prod -c 8 --mem=40G -t 01:00:00)
join() { local IFS=:; echo "$*"; }
submit() {  # sbatch -J NAME ...: prints the job ID (the name in a dry run) and shows "NAME ID"
  if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "$2"; return; fi
  local job; job=$(sbatch --parsable --kill-on-invalid-dep=yes "$@"); echo "$2 $job" >&2; echo "$job"
}
table=$(submit -J pairs-table "${cpu[@]}" scripts/slurm/cpu.sh routing.sentence_table rebuild --model qwen36)
submit -J pairs-original-qwen36 "${cpu[@]}" --dependency=afterok:$table \
  scripts/slurm/cpu.sh routing.transition_pairs summary --model qwen36 --original-sample-only > /dev/null
summaries=()
for m in "${models[@]}"; do
  after=""; test $m != qwen36 || after=--dependency=afterok:$table
  plan=$(submit -J pairs-plan-$m "${cpu[@]}" $after \
    scripts/slurm/cpu.sh routing.transition_pairs plan --model $m --source-classes $classes)
  dependency=afterok:$plan; test -z "${AFTER:-}" || dependency+=,afterany:$(join $AFTER)
  judges=()
  for ((i = 0; i < shards; i++)); do
    judges+=($(submit -J pairs-judge-$m-$i --dependency=$dependency \
      scripts/routing/pairs_judge.sbatch $m $i/$shards $classes))
  done
  summaries+=($(submit -J pairs-summary-$m "${cpu[@]}" --dependency=afterok:$(join "${judges[@]}") \
    scripts/slurm/cpu.sh routing.transition_pairs summary --model $m --source-classes $classes))
done
submit -J pairs-collect -p boost_usr_prod -c 2 --mem=8G -t 00:20:00 --dependency=afterok:$(join "${summaries[@]}") \
  scripts/slurm/cpu.sh routing.transition_pairs collect --source-classes $classes > /dev/null
