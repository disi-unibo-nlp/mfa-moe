#!/bin/bash
#SBATCH --job-name=mfa-qwen-fp-fa-benchmark
#SBATCH --account=IscrC_MIOSR
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-fp-fa-benchmark-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-fp-fa-benchmark-%j.err

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
GENERATION_MODEL=openai/gpt-oss-20b
SOURCE_ROOT="$P/repo/results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b"
PROGRAM_FILE="$P/repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
HF_HOME="$SCRATCH_ROOT/hf"
HF_CACHE_DIR="$HF_HOME/hub"
RUNTIME_ROOT="$SCRATCH_ROOT/runtime"
DATASETS=(math500 aime24 aime25 olympiad amc23 minerva)
SAMPLE_LIMIT=8
RUN_ROOT="$SCRATCH_ROOT/benchmarks/qwen3.8-27b-fullprecision-flash-attn/${SLURM_JOB_ID:-manual-$(date +%Y%m%dT%H%M%S)}"
SAMPLED_GENERATION="$RUN_ROOT/generation"
ANNOTATION_ROOT="$RUN_ROOT/annotations"
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
"$PY" -c 'import dspy, torch, vllm; from dspy.teleprompt import GEPA; print("judge_env_ready dspy={} gepa={} vllm={}".format(dspy.__version__, GEPA.__name__, getattr(vllm, "__version__", "unknown")))'

trap 'status=$?; if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true; fi; exit "$status"' EXIT INT TERM

test -s "$PROGRAM_FILE"
test -d "$HF_CACHE_DIR"
test -d "$HF_CACHE_DIR/models--Qwen--Qwen3.8-27B"
for dataset in "${DATASETS[@]}"; do
    test -s "$SOURCE_ROOT/$dataset/traces.jsonl"
done

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" --version
"$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; print(f"cuda_devices={torch.cuda.device_count()}")'
test -x "$VLLM"

# Copy eight evenly spaced real traces per dataset to a scratch input tree.
"$PY" - "$SOURCE_ROOT" "$SAMPLED_GENERATION" "${DATASETS[@]}" "$SAMPLE_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
destination_root = Path(sys.argv[2]) / "openai--gpt-oss-20b"
datasets = sys.argv[3:-1]
limit = int(sys.argv[-1])
if limit < 2:
    raise SystemExit("SAMPLE_LIMIT must be at least 2")

for dataset in datasets:
    source = source_root / dataset / "traces.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    if len(rows) < limit:
        raise SystemExit(f"{dataset}: {len(rows)} traces < requested {limit}")
    indices = [(i * (len(rows) - 1)) // (limit - 1) for i in range(limit)]
    if len(set(indices)) != limit:
        raise SystemExit(f"{dataset}: sample indices are not unique")
    target = destination_root / dataset
    target.mkdir(parents=True, exist_ok=True)
    with (target / "traces.jsonl").open("w") as handle:
        for index in indices:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    print(f"sample dataset={dataset} source_traces={len(rows)} selected_traces={len(indices)}")
PY

VLLM_COMMAND=(
    srun --ntasks=1 --nodes=1 "$VLLM" serve "$MODEL"
    --served-model-name "$MODEL" --download-dir "$HF_CACHE_DIR"
    --host 127.0.0.1 --port 41800 --api-key local-vllm-key
    --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192
    --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90
    --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16
    --reasoning-parser qwen3 --language-model-only --tensor-parallel-size 2
    --attention-config '{"backend":"FLASH_ATTN"}'
)
printf 'qwen_fp_flash_attn_command='; printf '%q ' "${VLLM_COMMAND[@]}"; printf '\n'
"${VLLM_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 900))
until curl --silent --fail --max-time 5 http://127.0.0.1:41800/health >/dev/null 2>&1; do
    if (( SECONDS >= deadline )) || ! kill -0 "$SERVER_PID" 2>/dev/null; then
        printf 'vLLM did not become ready; server_log=%s\n' "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
done
printf 'native_vllm_ready model=%s backend=FLASH_ATTN tensor_parallel_size=2 max_num_seqs=16\n' "$MODEL"

START_NS=$(date +%s%N)
set +e
"$PY" -m moe_exp.correlation_pipeline.annotate \
    --generation-dir "$SAMPLED_GENERATION" \
    --generation-model "$GENERATION_MODEL" \
    --datasets "${DATASETS[@]}" \
    --output-dir "$ANNOTATION_ROOT" \
    --judge-program "$PROGRAM_FILE" \
    --judge-model "$MODEL" \
    --base-url http://127.0.0.1:41800/v1 \
    --api-key local-vllm-key \
    --workers 16 --max-tokens 4096 --temperature 0.0 \
    --enable-thinking --reasoning-effort high
ANNOTATE_STATUS=$?
set -e
END_NS=$(date +%s%N)
if (( ANNOTATE_STATUS != 0 )); then
    exit "$ANNOTATE_STATUS"
fi

SUMMARY="$ANNOTATION_ROOT/openai--gpt-oss-20b/summary.json"
test -s "$SUMMARY"
"$PY" - "$SUMMARY" "$START_NS" "$END_NS" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
elapsed = (int(sys.argv[3]) - int(sys.argv[2])) / 1_000_000_000
traces = sum(item["traces"] for item in summary["datasets"])
units = sum(item["units"] for item in summary["datasets"])
if summary["status"] != "complete" or len(summary["datasets"]) != 6 or elapsed <= 0:
    raise SystemExit("invalid benchmark summary")
rate = units / elapsed
print(json.dumps({
    "benchmark": "qwen3.8-27b-fullprecision-flash-attn",
    "datasets": [item["dataset"] for item in summary["datasets"]],
    "traces": traces,
    "completed_sentence_requests": units,
    "requests": units,
    "responses": units,
    "elapsed_seconds": elapsed,
    "requests_per_second": rate,
    "sentences_per_second": rate,
    "sentences_per_minute": rate * 60,
}, sort_keys=True))
PY
printf 'SMOKE_OK benchmark=qwen3.8-27b-fullprecision-flash-attn summary=%s\n' "$SUMMARY"
