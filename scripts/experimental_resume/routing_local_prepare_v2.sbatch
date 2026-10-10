#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=routing-v2-prepare
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
[[ "${SLURM_CLUSTER_NAME:-}" == leonardo && "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
: "${ROUTING_V2_OUT:?new preparation output required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/rc2p-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$ROUTING_V2_OUT"
export TMPDIR="$RUNTIME/tmp" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY="$REPO/.venv-native-finalization-tests/bin/python"
cd "$REPO"
srun --ntasks=1 "$PY" -B -m pytest -q \
  tests/experimental_resume/test_routing_study_v2.py \
  tests/experimental_resume/test_routing_analysis_v2.py \
  tests/experimental_resume/test_routing_local_backend_v2.py \
  tests/experimental_resume/test_routing_study_receipts_local_v2.py \
  --basetemp="$RUNTIME/tests" --junitxml="$ROUTING_V2_OUT/CPU_TESTS.xml"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/routing_local_engineering_v2.py" \
  prepare --output "$ROUTING_V2_OUT"
srun --ntasks=1 "$PY" -B -c 'import os, pathlib, xml.etree.ElementTree as E; from routing_local_backend_v2 import source_binding; from moe_exp.routing_control.study_v1 import save, file_sha; p=pathlib.Path(os.environ["ROUTING_V2_OUT"]); suites=E.parse(p/"CPU_TESTS.xml").getroot(); tests=sum(int(s.get("tests",0)) for s in suites); save(p/"CPU_CHECKS.json", {"schema":"routing-local-cpu-checks-v2", "status":"PASS", "tests_passed":tests, "junit_file_sha256":file_sha(p/"CPU_TESTS.xml"), "code_files":source_binding(), "job_id":os.environ["SLURM_JOB_ID"]})'
