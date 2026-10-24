#!/bin/bash
#SBATCH --job-name=mfa-qwen-label-resume
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err

# Label one self-contained 25k stratified part of the GPT-OSS, Gemma, Qwen3.6 or
# Nemotron benchmark generations with the usual Qwen3.8-27B judge (TP=2, MTP=3, batch 64,
# reasoning effort low). It generalises native_nemotron_annotate.sbatch: the
# trace roots come from
#   results/correlation_pipeline/<source root>/reasoning-vllm-v1/
#     sampling-100k-stratified/parts/part-XX/generation/<slug>
# (written by sample_stratified.py), each job only reads its own part, writes
# only its own
#   qwen38-mtp3-stratified/<source>/part-XX
# and resumes from that part's checkpoints. Ports are unique per source and
# part because two 2-GPU jobs can share one 4-GPU node.
set -euo pipefail

P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
CODE_ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/label-resume-v1/code
WATCHDOG_SCRIPT="$CODE_ROOT/qwen_annotation_watchdog.sh"
VLLM="$P/envs/vllm-cu129/bin/vllm"
ENV_POINTER=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-current.path
MODEL=Qwen/Qwen3.8-27B
PROGRAM_FILE="$REPO/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
CACHE_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/hf
STRAT_ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/label-resume-v1
SAMPLING_DIR=sampling-100k-stratified
SOURCE=""
PART=""
PORT=""
OUTPUT_DIR=""

usage() {
    printf '%s\n' 'Usage: sbatch/native_stratified_annotate.sbatch --source gpt|gemma|qwen36|nemotron|glm|qwen330b --part 0..3 --port PORT --output-dir ABSOLUTE_SCRATCH_PART'
    printf '%s\n' '  ports: gpt 43000, gemma 43100, qwen36 43200, nemotron 43300, glm 43400, qwen330b 43500 (+part); output-dir: <strat-root>/<source>/part-0X'
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
case "$SOURCE" in
    gpt)
        SOURCE_RESULTS=gpt-oss-20b
        SLUG=openai--gpt-oss-20b
        # v2: the first sample stratified gpt-oss by a reasoning length that vLLM
        # reports as a constant 0 for the harmony path, so its length quartiles
        # were decided by the trace digest tiebreak alone. sample_stratified.py
        # now derives the length from the saved token replay.
        SAMPLING_DIR=sampling-100k-stratified-v2
        PORT_BASE=43000 ;;
    gemma)
        SOURCE_RESULTS=gemma-nvfp4-nf4
        SLUG=nvidia--Gemma-4-26B-A4B-NVFP4
        PORT_BASE=43100 ;;
    qwen36)
        SOURCE_RESULTS=qwen36-35b-a3b-fp8
        SLUG=Qwen--Qwen3.6-35B-A3B-FP8
        PORT_BASE=43200 ;;
    nemotron)
        SOURCE_RESULTS=nemotron-nvfp4-dspark
        SLUG=nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
        PORT_BASE=43300 ;;
    glm)
        SOURCE_RESULTS=glm-4.7-flash
        SLUG=zai-org--GLM-4.7-Flash
        PORT_BASE=43400 ;;
    qwen330b)
        SOURCE_RESULTS=qwen3-30b-a3b
        SLUG=Qwen--Qwen3-30B-A3B
        PORT_BASE=43500 ;;
    *) echo '--source must be gpt, gemma, qwen36, nemotron, glm or qwen330b' >&2; exit 2 ;;
esac
[[ "$PART" =~ ^[0-3]$ ]] || { echo '--part must be 0..3' >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo '--port must be numeric' >&2; exit 2; }
PARTS_ROOT="$REPO/results/correlation_pipeline/$SOURCE_RESULTS/reasoning-vllm-v1/$SAMPLING_DIR/parts"
PRODUCTION_ROOT="$STRAT_ROOT/$SOURCE"
EXPECTED_PORT=$((PORT_BASE + PART))
EXPECTED_OUTPUT="$PRODUCTION_ROOT/part-$(printf '%02d' "$PART")"
TRACE_ROOT="$PARTS_ROOT/part-$(printf '%02d' "$PART")/generation/$SLUG"
[[ "$PORT" -eq "$EXPECTED_PORT" ]] || { echo "port mismatch: expected $EXPECTED_PORT for $SOURCE part $PART" >&2; exit 2; }
[[ "$OUTPUT_DIR" == "$EXPECTED_OUTPUT" ]] || { echo "output-dir must be $EXPECTED_OUTPUT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$SOURCE" == qwen330b && "$PART" =~ ^[01]$ ]] || { echo 'runtime execution requires Slurm' >&2; exit 2; }

test -s "$PROGRAM_FILE"
test -s "$PARTS_ROOT/part-$(printf '%02d' "$PART")/part_manifest.json"
test -s "$TRACE_ROOT/math500/traces.jsonl"
test -s "$TRACE_ROOT/aime24/traces.jsonl"
test -s "$TRACE_ROOT/aime25/traces.jsonl"
test -s "$TRACE_ROOT/olympiad/traces.jsonl"
test -s "$TRACE_ROOT/amc23/traces.jsonl"
test -s "$TRACE_ROOT/minerva/traces.jsonl"
test -s "$ENV_POINTER"
test -s "$CODE_ROOT/src/moe_exp/correlation_pipeline/annotation_batch.py"
test -x "$WATCHDOG_SCRIPT"
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
export PYTHONPATH="$CODE_ROOT/src:$ENV_ROOT/lib/python3.11/site-packages:$P/envs/vllm-cu129/lib/python3.11/site-packages"
mkdir -p "$OUTPUT_DIR/runtime/torch" "$OUTPUT_DIR/runtime/triton" "$OUTPUT_DIR/runtime/xdg" "$OUTPUT_DIR/runtime/tmp" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoints"

"$ENV_ROOT/bin/python" -c 'import dspy, torch, vllm; assert torch.cuda.is_available() and torch.cuda.device_count() == 2; print("judge_env_ready dspy={} vllm={} cuda_devices={}".format(dspy.__version__, getattr(vllm, "__version__", "unknown"), torch.cuda.device_count()), flush=True)'
"$ENV_ROOT/bin/python" -c 'import moe_exp.correlation_pipeline.annotation_batch as module; print("annotation_module", module.__file__, flush=True)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

RESUME_MAX_MODEL_LEN=$("$ENV_ROOT/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "ready"; print(d["max_model_len"])' "$OUTPUT_DIR/preflight.json")
printf 'resume_context_capacity=%s saved_checkpoints_verified=true\n' "$RESUME_MAX_MODEL_LEN"
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
    --max-model-len "$RESUME_MAX_MODEL_LEN" --max-num-seqs 64 --max-num-batched-tokens 8192
    --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90
    --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16
    --reasoning-parser qwen3 --language-model-only --tensor-parallel-size 2
    --attention-config '{"backend":"FLASH_ATTN"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)
printf 'production_source=%s part=%s port=%s code_root=%s trace_root=%s output_dir=%s\n' "$SOURCE" "$PART" "$PORT" "$CODE_ROOT" "$TRACE_ROOT" "$OUTPUT_DIR"
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

CLIENT_LOG="$OUTPUT_DIR/logs/annotation_client.log"
export MFA_FAULTHANDLER=1

client_command=(
    "$ENV_ROOT/bin/python" -m moe_exp.correlation_pipeline.annotation_batch
    --trace-root "$TRACE_ROOT"
    --output-dir "$OUTPUT_DIR"
    --part 0 --parts 1 --part-size 25000 --total 25000
    --judge-program "$PROGRAM_FILE" --model "$MODEL"
    --base-url "http://127.0.0.1:$PORT/v1" --api-key local-vllm-key
    --batch-size 64 --max-tokens 16384 --temperature 1.0
    --top-p 0.95 --top-k 20 --min-p 0.0
    --presence-penalty 0.0 --repetition-penalty 1.0
    --reasoning-effort low
)

"$WATCHDOG_SCRIPT" \
    --output-dir "$OUTPUT_DIR" \
    --client-log "$CLIENT_LOG" \
    --stall-seconds "${MFA_STALL_SECONDS:-1500}" \
    --poll-seconds "${MFA_POLL_SECONDS:-60}" \
    --max-restarts "${MFA_MAX_RESTARTS:-24}" \
    -- "${client_command[@]}"

test -s "$OUTPUT_DIR/annotations.json"
test -s "$OUTPUT_DIR/summary.json"
"$ENV_ROOT/bin/python" -c 'import json, sys; summary = json.load(open(sys.argv[1])); assert summary["status"] == "complete" and summary["completed"] == summary["expected"] == 25000, summary; print("final_summary", json.dumps(summary, sort_keys=True))' "$OUTPUT_DIR/summary.json"
