#!/bin/bash
#SBATCH --job-name=dynamics-label
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
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/dynamics-label-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/dynamics-label-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit through Slurm only after workload approval" >&2; exit 2; }
source /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/routing_dynamics_env.sh
PORT=43842
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
srun --ntasks=1 --nodes=1 "$P/envs/vllm-cu129/bin/vllm" serve Qwen/Qwen3.8-27B \
    --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --served-model-name Qwen/Qwen3.8-27B --host 127.0.0.1 --port "$PORT" \
    --api-key local-vllm-key --download-dir "$P/cache/hf/hub" \
    --max-model-len 49152 --max-num-seqs 64 --max-num-batched-tokens 8192 \
    --enable-prefix-caching --generation-config vllm --gpu-memory-utilization 0.90 \
    --enforce-eager --dtype bfloat16 --kv-cache-dtype bfloat16 --reasoning-parser qwen3 \
    --language-model-only --tensor-parallel-size 2 --attention-config '{"backend":"FLASH_ATTN"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    >"$ROOT/judge-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 1200))
until curl --silent --fail --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
    (( SECONDS < deadline )) && kill -0 "$SERVER_PID" 2>/dev/null || exit 1
    sleep 2
done
"$PYTHON" scripts/run_routing_dynamics.py label --root "$ROOT" --base-url "http://127.0.0.1:$PORT/v1"
