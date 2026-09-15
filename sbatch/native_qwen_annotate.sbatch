#!/bin/bash
#SBATCH --job-name=mfa-qwen-b64-mtp3-label
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=07:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production/logs/%x-%j.err

set -euo pipefail

P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
PY_BASE="$P/envs/vllm-cu129/bin/python"
VLLM="$P/envs/vllm-cu129/bin/vllm"
ENV_POINTER=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-current.path
MODEL=Qwen/Qwen3.8-27B
PROGRAM_FILE="$REPO/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
GPT_TRACE_ROOT="$REPO/results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b"
GEMMA_TRACE_ROOT="$REPO/results/correlation_pipeline/gemma-nvfp4-nf4/reasoning-vllm-v1/sampling/generation/nvidia--Gemma-4-26B-A4B-NVFP4"
CACHE_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/hf
SOURCE=""
PART=""
PORT=""
OUTPUT_DIR=""

usage() {
    printf '%s\n' 'Usage: sbatch/native_qwen_annotate.sbatch --source gpt|gemma --part 0..3 --port PORT --output-dir ABSOLUTE_SCRATCH_PART'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE=${2:?}; shift 2 ;;
        --part) PART=${2:?}; shift 2 ;;
        --port) PORT=${2:?}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$SOURCE" == gpt || "$SOURCE" == gemma ]] || { echo '--source must be gpt or gemma' >&2; exit 2; }
[[ "$PART" =~ ^[0-3]$ ]] || { echo '--part must be 0..3' >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo '--port must be numeric' >&2; exit 2; }
if [[ "$SOURCE" == gpt ]]; then
    TRACE_ROOT="$GPT_TRACE_ROOT"
    EXPECTED_PORT=$((41800 + PART))
    EXPECTED_PREFIX=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production/gpt/part-
else
    TRACE_ROOT="$GEMMA_TRACE_ROOT"
    EXPECTED_PORT=$((41900 + PART))
    EXPECTED_PREFIX=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production/gemma/part-
fi
[[ "$PORT" -eq "$EXPECTED_PORT" ]] || { echo "port/source/part mismatch: expected $EXPECTED_PORT" >&2; exit 2; }
[[ "$OUTPUT_DIR" == "$EXPECTED_PREFIX$(printf '%02d' "$PART")" ]] || { echo "output-dir must be the immutable production part root" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'runtime execution requires Slurm' >&2; exit 2; }
test -s "$PROGRAM_FILE"
test -s "$TRACE_ROOT/math500/traces.jsonl"
test -s "$ENV_POINTER"
test -d "$CACHE_ROOT/hub/models--Qwen--Qwen3.8-27B"
ENV_ROOT=$(<"$ENV_POINTER")
test -x "$ENV_ROOT/bin/python"
test -x "$VLLM"

module load python/3.11.7 cuda/12.6
export HF_HOME="$CACHE_ROOT" HF_HUB_CACHE="$CACHE_ROOT/hub" TRANSFORMERS_CACHE="$CACHE_ROOT" HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_HOME="$OUTPUT_DIR/runtime/torch" TRITON_CACHE_DIR="$OUTPUT_DIR/runtime/triton"
export XDG_CACHE_HOME="$OUTPUT_DIR/runtime/xdg" TMPDIR="$OUTPUT_DIR/runtime/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$P/envs/vllm-cu129/bin:$PATH"
export PYTHONPATH="$ENV_ROOT/lib/python3.11/site-packages:$P/envs/vllm-cu129/lib/python3.11/site-packages:$REPO/src"
mkdir -p "$OUTPUT_DIR/runtime/torch" "$OUTPUT_DIR/runtime/triton" "$OUTPUT_DIR/runtime/xdg" "$OUTPUT_DIR/runtime/tmp" "$OUTPUT_DIR/logs"

"$ENV_ROOT/bin/python" -c 'import dspy, torch, vllm; assert torch.cuda.is_available() and torch.cuda.device_count() == 2; print("judge_env_ready dspy={} vllm={} cuda_devices={}".format(dspy.__version__, getattr(vllm, "__version__", "unknown"), torch.cuda.device_count()), flush=True)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

SERVER_LOG="$OUTPUT_DIR/logs/vllm.log"
SERVER_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

VLLM_COMMAND=(
    srun --ntasks=1 --nodes=1 "$VLLM" serve "$MODEL"
    --served-model-name "$MODEL" --download-dir "$CACHE_ROOT/hub"
    --host 127.0.0.1 --port "$PORT" --api-key local-vllm-key
    --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 8192
    --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90
    --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16
    --reasoning-parser qwen3 --language-model-only --tensor-parallel-size 2
    --attention-config '{"backend":"FLASH_ATTN"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)
printf 'production_command='; printf '%q ' "${VLLM_COMMAND[@]}"; printf '\n'
printf 'production_source=%s part=%s port=%s trace_root=%s output_dir=%s\n' "$SOURCE" "$PART" "$PORT" "$TRACE_ROOT" "$OUTPUT_DIR"
"${VLLM_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )) || ! kill -0 "$SERVER_PID" 2>/dev/null; then
        printf 'VLLM_SERVER_FAILED log=%s\n' "$SERVER_LOG" >&2
        tail -40 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done
printf 'native_vllm_ready model=%s backend=FLASH_ATTN tensor_parallel_size=2 mtp=3 batch_size=64 source=%s part=%s port=%s\n' "$MODEL" "$SOURCE" "$PART" "$PORT"

grep -iE 'speculative|mtp|flash|attention' "$SERVER_LOG" | head -30 || true

"$ENV_ROOT/bin/python" -m moe_exp.correlation_pipeline.annotation_batch \
    --trace-root "$TRACE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --part "$PART" --parts 4 --part-size 25000 --total 100000 \
    --judge-program "$PROGRAM_FILE" --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT/v1" --api-key local-vllm-key \
    --batch-size 64 --max-tokens 4096 --temperature 1.0 \
    --top-p 0.95 --top-k 20 --min-p 0.0 \
    --presence-penalty 0.0 --repetition-penalty 1.0 \
    --reasoning-effort medium
