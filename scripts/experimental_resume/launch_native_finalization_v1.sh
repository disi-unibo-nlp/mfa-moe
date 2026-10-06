#!/bin/bash
# User-authorized one-cycle launch. No scientific work on the login node.
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -f)" == *.leonardo.local ]] || exit 2
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24"
exec /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -c '
import native_finalization_operations_v1 as ops
ops.live()
ops.submit("preparation", [str(ops.SCRIPTS / "native_finalization_cpu_v1.sbatch"), "prepare"],
    {"scope": "authorized one diagnostic cycle; 32 original-prompt assignments", "source_files": ops.N.sources()})
'
