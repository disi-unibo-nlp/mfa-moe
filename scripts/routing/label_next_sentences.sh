#!/bin/bash
# Fig. 6 and the state-transitions table on all eight models: label the sentences that follow a Read or Verify
# sentence (or the SOURCE_CLASSES you name) and redo each native-dynamics panel with them. Per model it submits a plan
# (CPU), SHARDS judge jobs (2 GPUs, 4 hours each) and a summary (CPU); one last job joins the eight panels into
# STUDY_DIR/driver/TRANSITION_PAIRS.<variant>.json. The models are the ones STUDY_DIR/BINDING.json lists. Qwen3.6 first
# rebuilds its sentence table, which covers only 509 of its 1,486 question families, and its original-sample panel.
#   scripts/routing/label_next_sentences.sh STUDY_DIR [SOURCE_CLASSES [SHARDS]]    default: Read,Verify and 2 shards per model
#   scripts/routing/label_next_sentences.sh report/final-routing-20261015-r1 Read,Verify,Plan,Explore,Monitor 3
# AFTER="JOB ..." names jobs the judges wait for. A wider rule reuses the labels of a narrower one (one label store per
# model), so its judges must not run while the narrower ones still do. DRY_RUN=1 prints the sbatch lines only.
# A relaunch continues the chain: a stage whose receipt exists is not submitted again (a model whose panel exists is
# done, one whose plan exists goes straight to its judges, which resume), and the stages are idempotent in any case.
# Needs Python 3.11 and PYTHONPATH=src, for the name of the receipt folder of the rule.
set -euo pipefail
study=${1:?STUDY_DIR}; shards=${3:-2}; test -e "$study/BINDING.json"
rule=(); test -z "${2:-}" || rule=(--source-classes "$2")
variant=$(python -m routing.next_sentence_labels variant "${rule[@]}")
mapfile -t models < <(jq -r '.config.models[]' "$study/BINDING.json")
partial=qwen36  # the one model whose saved sentence table is not complete
cpu=(-p boost_usr_prod -c 8 --mem=40G -t 01:00:00)
labels=(scripts/slurm/run_module_cpu.sbatch routing.next_sentence_labels)
join() { local IFS=:; echo "$*"; }
have() { test -e "$study/$1"; }
submit() {  # sbatch -J NAME ...: prints the job ID (the name in a dry run) and shows "NAME ID"
  if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "$2"; return; fi
  local job; job=$(sbatch --parsable --kill-on-invalid-dep=yes "$@"); echo "$2 $job" >&2; echo "$job"
}
afterok() {  # --dependency=afterok:ID:... for the jobs that were submitted (nothing when none was)
  local ids=(); for id in "$@"; do test -z "$id" || ids+=("$id"); done
  test ${#ids[@]} -eq 0 || echo "--dependency=afterok:$(join "${ids[@]}")"
}
table=""
have transition-pairs/$partial/SENTENCES.json ||
  table=$(submit -J pairs-table "${cpu[@]}" scripts/slurm/run_module_cpu.sbatch routing.labelled_sentence_table rebuild --study "$study" --model $partial)
have transition-pairs/$partial/NATIVE_DYNAMICS_SUMMARY.original-sample.json ||
  submit -J pairs-original-$partial "${cpu[@]}" $(afterok $table) "${labels[@]}" summary --study "$study" --model $partial --original-sample-only > /dev/null
summaries=()
for m in "${models[@]}"; do
  have transition-pairs/$m/$variant/NATIVE_DYNAMICS_SUMMARY.json && continue
  plan=""
  have transition-pairs/$m/$variant/PAIR_PLAN.json || plan=$(submit -J pairs-plan-$m "${cpu[@]}" \
    $(test $m != $partial || afterok $table) "${labels[@]}" plan --study "$study" --model $m "${rule[@]}")
  dependencies=(); test -z "$plan" || dependencies+=(afterok:$plan)
  test -z "${AFTER:-}" || dependencies+=(afterany:$(join $AFTER))
  dependency=""; test ${#dependencies[@]} -eq 0 || dependency=--dependency=$(IFS=,; echo "${dependencies[*]}")
  judges=()
  for ((i = 0; i < shards; i++)); do
    judges+=($(submit -J pairs-judge-$m-$i $dependency scripts/routing/judge_next_sentences.sbatch "$study" $m $i/$shards "${2:-}"))
  done
  summaries+=($(submit -J pairs-summary-$m "${cpu[@]}" $(afterok "${judges[@]}") "${labels[@]}" summary --study "$study" --model $m "${rule[@]}"))
done
have driver/TRANSITION_PAIRS.$variant.json ||
  submit -J pairs-collect -p boost_usr_prod -c 2 --mem=8G -t 00:20:00 $(afterok "${summaries[@]}") \
    "${labels[@]}" collect --study "$study" "${rule[@]}" > /dev/null
