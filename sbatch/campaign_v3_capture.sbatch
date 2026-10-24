#!/bin/bash
#SBATCH --job-name=mfa-v3-capture
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "$(hostname)" == lrdn* ]] || exit 2
BUNDLE=${1:?Explicit frozen v3 bundle required}
SOURCE=${2:?Source required}
MANIFEST=${3:?Explicit immutable manifest required}
MODE=${4:-capture}
case "$SOURCE" in gpt|glm|qwen36|nemotron|gemma|qwen330b|qwen35) ;; *) exit 2;; esac
[[ "$MODE" == capture || "$MODE" == verify ]] || exit 2
[[ -f "$BUNDLE/campaign-v3.json" && -d "$BUNDLE/src/moe_exp" ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
RUN="$P/routing-study/results/measurement-study/campaign-v3/runtime/$SLURM_JOB_ID"
module load python/3.11.7 cuda/12.6
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" MPLCONFIGDIR="$RUN/cache/mpl"
export TORCH_HOME="$RUN/cache/torch" TRITON_CACHE_DIR="$RUN/cache/triton" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_CACHE="$P/cache/hf/hub" HF_HOME="$RUN/cache/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
# ZMQ IPC socket paths (base + "/" + uuid4) must fit sockaddr_un (107 chars); $RUN is far too deep
RPC="$P/r/$SLURM_JOB_ID"
mkdir -p "$RPC"
trap 'rm -rf "$RPC"' EXIT
export VLLM_RPC_BASE_PATH="$RPC"
export MEASUREMENT_BUNDLE="$BUNDLE"
export PYTHONPATH="$BUNDLE/src:$BUNDLE:$BUNDLE/scripts:$P/envs/vllm-cu129/lib/python3.11/site-packages"
export PATH="$P/envs/vllm-cu129/bin:$PATH"
if [[ "$SOURCE" == gpt ]]; then
  export LOCAL_KERNELS="kernels-community/gpt-oss-triton-kernels=$P/cache/hf/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots/0f351046d799bc4d2c9dbb2c7cf36753204b5916"
  # transformers loads gpt-oss natively as MXFP4 only with kernels>=0.16,<0.17 (the vLLM env ships
  # 0.11.7 and would silently dequantize); the overlay is gpt-only so it cannot shadow vLLM's packages.
  export PYTHONPATH="$BUNDLE/src:$BUNDLE:$BUNDLE/scripts:$P/envs/forward-kernels-0.16:$P/envs/vllm-cu129/lib/python3.11/site-packages"
fi
# Slurm's actual allocation end is authoritative; fail rather than run unguarded.
MEASUREMENT_JOB_END=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
export MEASUREMENT_DEADLINE_EPOCH=$(date -d "$MEASUREMENT_JOB_END" +%s)
cd "$BUNDLE"
args=("$SOURCE" --manifest "$MANIFEST" --bundle "$BUNDLE")
if [[ "$MODE" == verify ]]; then args+=(--verify-only); fi
srun "$P/envs/vllm-cu129/bin/python" -B -u -m measurement_study.campaign_v3_capture "${args[@]}"
