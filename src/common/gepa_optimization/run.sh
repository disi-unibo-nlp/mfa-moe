#!/bin/bash
################################################################################
# GEPA Prompt Optimization — run script
#
# Usage:  ./run.sh [config_file]
# Example: ./run.sh config/gepa_config.yaml
#
# This script:
#   1. Reads model/server settings from the YAML config
#   2. Starts two vLLM servers (FOL model + judge model) in the background
#   3. Waits until both are healthy
#   4. Runs the Python optimization script
#   5. Shuts down the vLLM servers on exit
#
# Run from the gepa_optimization/ directory:
#   cd gepa_optimization
#   ./run.sh
################################################################################

set -e

CONFIG_FILE="${1:-config/gepa_config.yaml}"

[ ! -f "$CONFIG_FILE" ] && echo "ERROR: Config not found: $CONFIG_FILE" && exit 1

# Read vLLM-relevant fields from the YAML config
eval $(python3 -c "
import yaml, sys
with open('$CONFIG_FILE', 'r') as f:
    c = yaml.safe_load(f)
print(f\"FOL_MODEL={c.get('fol_model', '')}\")
print(f\"JUDGE_MODEL={c.get('judge_model', '')}\")
print(f\"FOL_PORT={c.get('fol_port', 8000)}\")
print(f\"JUDGE_PORT={c.get('judge_port', 8001)}\")
print(f\"FOL_MAX_LEN={c.get('fol_max_len', 2048)}\")
print(f\"FOL_GPU_MEM={c.get('fol_gpu_mem', 0.30)}\")
print(f\"JUDGE_MAX_LEN={c.get('judge_max_len', 4096)}\")
print(f\"JUDGE_GPU_MEM={c.get('judge_gpu_mem', 0.65)}\")
")

FOL_PID_FILE="/tmp/vllm_fol_gepa.pid"
JUDGE_PID_FILE="/tmp/vllm_judge_gepa.pid"

cleanup() {
    echo ""
    echo "Stopping vLLM servers..."
    [ -f "$FOL_PID_FILE" ] && kill "$(cat "$FOL_PID_FILE")" 2>/dev/null; rm -f "$FOL_PID_FILE"
    [ -f "$JUDGE_PID_FILE" ] && kill "$(cat "$JUDGE_PID_FILE")" 2>/dev/null; rm -f "$JUDGE_PID_FILE"
    echo "Done."
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local PORT=$1 NAME=$2
    echo "Waiting for $NAME on port $PORT (up to 5 min)..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "  $NAME is ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $NAME failed to become healthy after 5 minutes."
    echo "Check the log: output/vllm_fol.log or output/vllm_judge.log"
    return 1
}

mkdir -p output

echo "========================================"
echo "GEPA Optimization"
echo "========================================"
echo "Config      : $CONFIG_FILE"
echo "FOL model   : $FOL_MODEL  (port $FOL_PORT)"
echo "Judge model : $JUDGE_MODEL  (port $JUDGE_PORT)"
echo "========================================"

# ---- Start FOL server -------------------------------------------------------
echo ""
echo "[1/3] Starting FOL model server..."
vllm serve "$FOL_MODEL" \
    --port "$FOL_PORT" \
    --max-model-len "$FOL_MAX_LEN" \
    --max-num-seqs 4 \
    --gpu-memory-utilization "$FOL_GPU_MEM" \
    --dtype auto \
    --trust-remote-code \
    > output/vllm_fol.log 2>&1 &
echo $! > "$FOL_PID_FILE"

wait_for_server "$FOL_PORT" "FOL server" || exit 1

# ---- Start Judge server -----------------------------------------------------
echo ""
echo "[2/3] Starting judge model server..."
vllm serve "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --max-model-len "$JUDGE_MAX_LEN" \
    --max-num-seqs 4 \
    --gpu-memory-utilization "$JUDGE_GPU_MEM" \
    --dtype auto \
    --trust-remote-code \
    > output/vllm_judge.log 2>&1 &
echo $! > "$JUDGE_PID_FILE"

wait_for_server "$JUDGE_PORT" "Judge server" || exit 1

# ---- Run optimization -------------------------------------------------------
echo ""
echo "[3/3] Running GEPA optimization..."
echo "========================================"
python3 gepa_optimize_prompt.py --config "$CONFIG_FILE"
RESULT=$?

echo ""
echo "========================================"
if [ $RESULT -eq 0 ]; then
    echo "SUCCESS — results saved to output/gepa/"
else
    echo "FAILED (exit code $RESULT)"
    echo "vLLM logs: output/vllm_fol.log  output/vllm_judge.log"
fi
echo "========================================"

exit $RESULT
