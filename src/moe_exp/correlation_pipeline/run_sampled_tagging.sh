#!/usr/bin/env bash
set -euo pipefail

# All single-attempt problems and one solution per AIME/AMC problem, with error-rate-weighted sentence
# sampling. Reuse matching sentence checkpoints when this command is resumed.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"
RUN_DIR="results/correlation_pipeline/sentence-tagging-six-benchmarks-error-weighted-v2"
DATASETS=(math500 aime24 aime25 olympiad amc23 minerva)
REASONING_DIR="$RUN_DIR/reasoning-vllm-v1"
STAGE_LAUNCHER="$REPO_DIR/src/moe_exp/correlation_pipeline/run_docker.sh"
MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
export HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
mkdir -p "$RUN_DIR"
exec 9> "$RUN_DIR/pipeline.lock"
flock -n 9 || { echo "This sampled pipeline is already running" >&2; exit 1; }
exec >> "$RUN_DIR/pipeline.log" 2>&1

write_status() {
    python3 - "$RUN_DIR/status.json" "$1" "$2" "${3:-0}" <<'PY'
import datetime, json, os, sys
path, status, stage, code = sys.argv[1:]
value = {"status": status, "stage": stage, "exit_code": int(code),
         "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with open(path + ".tmp", "w") as output:
    json.dump(value, output, indent=2)
    output.write("\n")
os.replace(path + ".tmp", path)
PY
}
stage="sampling"
trap 'code=$?; if (( code != 0 )); then write_status failed "$stage" "$code"; fi' EXIT
write_status running "$stage"
docker run --rm --network none -e PYTHONPATH=/workspace/src \
    -v "$REPO_DIR:/workspace" -w /workspace moe-mfa-experiments:latest \
    python -m moe_exp.correlation_pipeline.sample_tagging \
    --analysis results/correlation_pipeline/reasoning-vllm-untagged-v1/analysis/unsloth--Qwen3.5-35B-A3B/correlations.json \
    --output-dir "$RUN_DIR" --datasets "${DATASETS[@]}" \
    --weight-by error-rate --sample-id 0 --max-sentences 100000

stage="tagging"
write_status running "$stage"
bash src/moe_exp/correlation_pipeline/run_all.sh \
    --skip-generate --skip-forward --skip-analyze --skip-sampling \
    --datasets "${DATASETS[@]}" --results-dir "$RUN_DIR" \
    --generation-model "$MODEL"

stage="class_forward"
write_status running "$stage"
bash "$STAGE_LAUNCHER" forward --datasets "${DATASETS[@]}" \
    --generation-dir "$RUN_DIR/generation" --generation-model "$MODEL" \
    --annotation-dir "$REASONING_DIR/annotations" \
    --output-dir "$REASONING_DIR/forward" --views class

stage="class_analysis"
write_status running "$stage"
bash "$STAGE_LAUNCHER" analyze --datasets "${DATASETS[@]}" \
    --forward-dir "$REASONING_DIR/forward" --output-dir "$REASONING_DIR/analysis" \
    --views class --skip-expert-identity --bootstrap-samples 500
write_status complete "$stage"
