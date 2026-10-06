#!/bin/bash
# Explicitly named transport recovery; no reuse or deletion of the failed receipt.
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -f)" == *.leonardo.local ]] || exit 2
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24"
exec /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -c '
import native_finalization_operations_v1 as ops
ops.live()
ops.N.save(ops.N.DOC / "TRANSPORT_CORRECTION_v2.json", {
    "schema": "native-finalization-transport-correction-v2", "failed_job_id": "59416713",
    "observed_failure": "environment: line 19: _module_raw: command not found; exit 127 in 2 seconds",
    "correction": "Preserve BASH_FUNC__module_raw%% alongside the existing credential-free environment allowlist.",
    "affected_scientific_results": [], "scientific_manifest_existed_at_failure": False,
    "adapter_file_sha256": ops.N.U.file_sha(ops.SCRIPTS / "native_finalization_transport_v2.py")})
ops.submit("preparation-v2", [str(ops.SCRIPTS / "native_finalization_cpu_v1.sbatch"), "prepare"],
    {"scope": "same authorized cycle; recover failed module transport before scientific execution", "source_files": ops.N.sources()})
'
