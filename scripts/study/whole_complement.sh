#!/bin/bash
# Seal a whole-reasoning complement release (the suppress cells the base releases left out), submit
# its job graph (allocated validation -> shards -> extraction, monitor, combined analysis of r7-r17
# plus this release) and queue the paper export of the combined receipt into a staging folder.
#   scripts/study/whole_complement.sh NAME LOCAL_TESTS
#   e.g. scripts/study/whole_complement.sh whole-reasoning-20261021-r18 107
# Needs a clean src/tests/scripts tree (commit first).
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
name="${1:?release name}"; tests="${2:?locally passed tests}"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
TMP=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/tmp
mkdir -p "$TMP"

# 1. Seal: receipts, COMPLEMENT.json, SOURCES.json, the code snapshot and launch.env.
TMPDIR="$TMP" PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B -m moe_exp.steering.whole_complement --name "$name" --local-tests "$tests"
set -a; source "report/$name/launch.env"; set +a
REL=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/releases/$COMMIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REL/src TMPDIR="$TMP" \
       STUDY_DATA_ROOT=$REPO/report/final-routing-20261015-r1 WHOLE_ROOT=$REPO/report/$name \
       WHOLE_MODELS=$MODELS WHOLE_SWEEP=$SWEEP WHOLE_SHARDS=$SHARDS WHOLE_CUTOFF=$CUTOFF \
       WHOLE_GPU_CEILING=$GPU_CEILING WHOLE_CPU_CEILING=$CPU_CEILING OMP_NUM_THREADS=1

# 2. The ordinary whole-reasoning graph runs it; SOURCES.json makes it queue the combined analysis.
(cd "$REL" && "$PY" -B -m moe_exp.steering.whole launch)

# 3. Paper tables and figure data from the combined receipt, staged outside the manuscript folder.
combine=$("$PY" -c "import json; print(json.load(open('$WHOLE_ROOT/JOBS.json'))['analysis']['id'])")
out=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/final-export-${name##*-}
env -u SLURM_JOB_ID sbatch --parsable --account=iscrc_miosr --partition=lrd_all_serial --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=60 --job-name=whole-final-export \
  --dependency="afterok:$combine" --output="$WHOLE_ROOT/logs/%x-%j.out" --error="$WHOLE_ROOT/logs/%x-%j.err" \
  --wrap "cd $REPO && module load python/3.11.7 && mkdir -p $out && PYTHONPATH=src:scripts/paper \
PYTHONDONTWRITEBYTECODE=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python -B \
scripts/paper/export_whole_results.py --effects $WHOLE_ROOT/analysis-combined/EFFECTS_WHOLE.json --out $out" \
  | tee "$WHOLE_ROOT/EXPORT_JOB"
