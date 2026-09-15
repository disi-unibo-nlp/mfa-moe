#!/bin/bash
#SBATCH --job-name=mfa-qwen-mtp-probe
#SBATCH --account=IscrC_MIOSR
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-mtp-probe-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-mtp-probe-%j.err

set -euo pipefail

P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
SCRATCH_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke
ENV_POINTER="$SCRATCH_ROOT/envs/qwen-fp-fa-dspy-current.path"
BASE_SITE="$P/envs/vllm-cu129/lib/python3.11/site-packages"

test -s "$ENV_POINTER"
ENV_ROOT="$(<"$ENV_POINTER")"
test -x "$ENV_ROOT/bin/python"
test -d "$BASE_SITE"

PY="$ENV_ROOT/bin/python"
VLLM="$P/envs/vllm-cu129/bin/vllm"
MODEL=Qwen/Qwen3.8-27B
PROGRAM_FILE="$P/repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
TRACES="$P/repo/results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b/math500/traces.jsonl"
HF_HOME="$SCRATCH_ROOT/hf"
HF_CACHE_DIR="$HF_HOME/hub"
RUNTIME_ROOT="$SCRATCH_ROOT/runtime"
RUN_ROOT="$SCRATCH_ROOT/mtp-probe/${SLURM_JOB_ID:-manual}"
SERVER_LOG="$RUN_ROOT/vllm.log"
SERVER_PID=""

mkdir -p "$RUN_ROOT" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/torch" "$RUNTIME_ROOT/triton" "$RUNTIME_ROOT/xdg"
module load python/3.11.7 cuda/12.6
export HF_HOME HF_HUB_CACHE="$HF_CACHE_DIR" HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_HOME="$RUNTIME_ROOT/torch" TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg" TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$P/envs/vllm-cu129/bin:$PATH"
export PYTHONPATH="$ENV_ROOT/lib/python3.11/site-packages:$BASE_SITE:$P/repo/src"

"$PY" -c 'import dspy, torch, vllm; from dspy.teleprompt import GEPA; print("judge_env_ready dspy={} vllm={}".format(dspy.__version__, getattr(vllm, "__version__", "unknown")))'

trap 'status=$?; if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true; fi; exit "$status"' EXIT INT TERM

test -s "$PROGRAM_FILE"
test -s "$TRACES"
test -d "$HF_CACHE_DIR/models--Qwen--Qwen3.8-27B"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Identical to the production serving contract, plus MTP speculative decoding.
VLLM_COMMAND=(
    srun --ntasks=1 --nodes=1 "$VLLM" serve "$MODEL"
    --served-model-name "$MODEL" --download-dir "$HF_CACHE_DIR"
    --host 127.0.0.1 --port 41800 --api-key local-vllm-key
    --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192
    --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90
    --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16
    --reasoning-parser qwen3 --language-model-only --tensor-parallel-size 2
    --attention-config '{"backend":"FLASH_ATTN"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
)
printf 'qwen_mtp_probe_command='; printf '%q ' "${VLLM_COMMAND[@]}"; printf '\n'
"${VLLM_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until curl --silent --fail --max-time 5 http://127.0.0.1:41800/health >/dev/null 2>&1; do
    if (( SECONDS >= deadline )) || ! kill -0 "$SERVER_PID" 2>/dev/null; then
        printf 'MTP_PROBE_SERVER_FAILED server_log=%s\n' "$SERVER_LOG" >&2
        printf -- '--- vllm.log tail ---\n' >&2
        tail -40 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done
printf 'native_vllm_ready model=%s backend=FLASH_ATTN tensor_parallel_size=2 mtp=on\n' "$MODEL"

grep -iE 'speculative|mtp|draft' "$SERVER_LOG" | head -20 || true

"$PY" -m moe_exp.correlation_pipeline.mtp_probe_run \
    --traces "$TRACES" \
    --judge-program "$PROGRAM_FILE" \
    --model "$MODEL" \
    --base-url http://127.0.0.1:41800/v1 \
    --api-key local-vllm-key \
    --batch-size 16 \
    --max-tokens 4096 --temperature 0.0 \
    --reasoning-effort medium
