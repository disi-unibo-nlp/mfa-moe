#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
umask 077
cv3_early_failure() {
  local cv3_exit=$?
  if [[ $cv3_exit -ne 0 && $# -eq 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    "$PY" -B -c 'import hashlib,json,os,pathlib,sys,time; c=json.loads(pathlib.Path(sys.argv[1]).read_text()); d={"schema":"verification-autodriver-shell-failure-v3","campaign_sha256":c["campaign_sha256"],"controller_card_sha256":c["sha256"],"controller_job_id":os.environ["SLURM_JOB_ID"],"exit_code":int(sys.argv[2]),"phase":"SHELL_OR_SRUN","observed_epoch":time.time(),"scientific_null":False}; d["sha256"]=hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest(); p=pathlib.Path(c["result_root"])/"automation/runs"/os.environ["SLURM_JOB_ID"]/"SHELL_FAILURE.json"; p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(".SHELL_FAILURE."+str(os.getpid())+".part"); t.write_text(json.dumps(d,sort_keys=True)+"\n"); os.chmod(t,0o600); os.replace(t,p)' "$1" "$cv3_exit" || true
  fi
  exit "$cv3_exit"
}
trap 'cv3_early_failure "$@"' EXIT
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_CLUSTER_NAME:-}" == leonardo && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
[[ $# == 1 ]] || exit 2
for cv3_name in ${!SLURM_CPU_BIND@} ${!SLURM_MEM_BIND@} ${!SRUN_@} ${!SBATCH_@}; do unset "$cv3_name"; done
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/cv3-autodriver-r3-$SLURM_JOB_ID
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=none --mem-bind=none "$PY" -B "$REPO/scripts/experimental_resume/verification_study_v3_autodriver_r3.py" run --card "$1"
