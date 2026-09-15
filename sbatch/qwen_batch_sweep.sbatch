#!/bin/bash
#SBATCH --job-name=mfa-qwen-batch-sweep
#SBATCH --account=IscrC_MIOSR
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-batch-sweep-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/qwen-batch-sweep-%j.err

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
GEN_ROOT="$P/repo/results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b"
TRACES_A="$GEN_ROOT/math500/traces.jsonl"
TRACES_B="$GEN_ROOT/amc23/traces.jsonl"
HF_HOME="$SCRATCH_ROOT/hf"
HF_CACHE_DIR="$HF_HOME/hub"
RUNTIME_ROOT="$SCRATCH_ROOT/runtime"
RUN_ROOT="$SCRATCH_ROOT/benchmarks/batch-sweep/${SLURM_JOB_ID:-manual}"
SCOREBOARD="$RUN_ROOT/scoreboard.jsonl"
SLICE_SIZE=256
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

"$PY" -c 'import dspy, torch, vllm; print("judge_env_ready dspy={} vllm={}".format(dspy.__version__, getattr(vllm, "__version__", "unknown")))'

test -s "$PROGRAM_FILE"; test -s "$TRACES_A"; test -s "$TRACES_B"
test -d "$HF_CACHE_DIR/models--Qwen--Qwen3.8-27B"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap 'status=$?; stop_server; exit "$status"' EXIT INT TERM

# $1=max_num_seqs  $2=mtp(on|off)  $3=server log path
start_server() {
    local max_num_seqs="$1" mtp="$2" log="$3"
    local command=(
        srun --ntasks=1 --nodes=1 "$VLLM" serve "$MODEL"
        --served-model-name "$MODEL" --download-dir "$HF_CACHE_DIR"
        --host 127.0.0.1 --port 41800 --api-key local-vllm-key
        --max-model-len 32768 --max-num-seqs "$max_num_seqs" --max-num-batched-tokens 8192
        --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90
        --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16
        --reasoning-parser qwen3 --language-model-only --tensor-parallel-size 2
        --attention-config '{"backend":"FLASH_ATTN"}'
    )
    if [[ "$mtp" == "on" ]]; then
        command+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
    fi
    printf 'server_start max_num_seqs=%s mtp=%s\n' "$max_num_seqs" "$mtp"
    printf 'server_command='; printf '%q ' "${command[@]}"; printf '\n'
    "${command[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1200))
    until curl --silent --fail --max-time 5 http://127.0.0.1:41800/health >/dev/null 2>&1; do
        if (( SECONDS >= deadline )) || ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf 'SWEEP_SERVER_FAILED log=%s\n' "$log" >&2
            tail -40 "$log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    printf 'native_vllm_ready model=%s backend=FLASH_ATTN tensor_parallel_size=2 max_num_seqs=%s mtp=%s\n' \
        "$MODEL" "$max_num_seqs" "$mtp"
}

run_group() {
    local max_num_seqs="$1" mtp="$2"; shift 2
    local batch_sizes=("$@")
    local log="$RUN_ROOT/vllm-mtp${mtp}-seqs${max_num_seqs}.log"
    start_server "$max_num_seqs" "$mtp" "$log"
    local mtp_flag=()
    [[ "$mtp" == "on" ]] && mtp_flag=(--mtp)
    "$PY" -m moe_exp.correlation_pipeline.batch_sweep_run \
        --traces "$TRACES_A" "$TRACES_B" \
        --slice-size "$SLICE_SIZE" \
        --judge-program "$PROGRAM_FILE" \
        --model "$MODEL" \
        --base-url http://127.0.0.1:41800/v1 \
        --api-key local-vllm-key \
        --scoreboard "$SCOREBOARD" \
        --max-num-seqs "$max_num_seqs" \
        --batch-sizes "${batch_sizes[@]}" \
        --concurrencies 1 2 \
        --max-tokens 4096 --temperature 0.0 \
        --reasoning-effort medium "${mtp_flag[@]}"
    grep -iE 'SpecDecoding metrics' "$log" | tail -3 || true
    stop_server
}

# Control group first so the label baseline exists before any other cell runs.
run_group 16 off 1 4 8 16
run_group 32 off 32
run_group 16 on  1 4 8 16
run_group 32 on  32

printf -- '--- scoreboard ---\n'
cat "$SCOREBOARD"

"$PY" - "$SCOREBOARD" <<'PY'
import json, sys
from pathlib import Path
from moe_exp.correlation_pipeline.batch_benchmark import SELECTION_RULE, select_winner

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print(f"cells_measured={len(rows)}")
print(f"selection_rule={SELECTION_RULE}")
winner = select_winner(rows)
if winner is None:
    print("SWEEP_RESULT=keep_single_request_mode")
else:
    print("SWEEP_RESULT=adopt " + json.dumps(winner, sort_keys=True))
mismatches = [r for r in rows if r.get("label_mismatch")]
print(f"cells_with_label_mismatch={len(mismatches)}")
PY

printf 'SWEEP_OK cells=%s scoreboard=%s\n' "$(wc -l <"$SCOREBOARD")" "$SCOREBOARD"
