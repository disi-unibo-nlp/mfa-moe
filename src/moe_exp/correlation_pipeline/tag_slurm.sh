#!/usr/bin/env bash
#SBATCH --job-name=correlation-tag
#SBATCH --output=slurm_logs/tag-%j.out
#SBATCH --error=slurm_logs/tag-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00

set -euo pipefail

usage() {
    cat <<'HELP'
Submit from the repository root after: mkdir -p slurm_logs
  sbatch src/moe_exp/correlation_pipeline/tag_slurm.sh \
    --model ORG/MODEL --generation-dir results/path/to/generation [options]

Tags existing traces only; no generation, sampling, extraction, or analysis.
--model is the model that GENERATED the traces, not the tagging judge.
Input: GENERATION_DIR/<model slug>/<dataset>/traces.jsonl
Output: RESULTS_DIR/reasoning-vllm-v1/annotations/<model slug>/<dataset>/
Existing matching annotation checkpoints are resumed automatically.

  --model NAME             Required saved generation model ID (or --generation-model)
  --generation-dir DIR     Required repo-relative directory containing model folders
  --datasets NAME...       Default: discover dataset folders containing traces.jsonl
  --results-dir DIR        Default: results/correlation_pipeline
  --judge-workers N        Default: 8
  --judge-tensor-parallel-size N  Judge tensor parallel GPU count (default: 1)
  --judge-model NAME       Override the default Qwen3.8-27B-NVFP4 judge
  --judge-reasoning-effort MODE  Judge thinking effort (default: high)
  --judge-program FILE     Override the frozen GEPA program (repo-relative)
  --limit N                Limit traces per dataset for a smoke test
  --dry-run                Validate inputs and print the plan without Docker calls

Images: IMAGE_NAME=moe-mfa-experiments:latest and
VLLM_IMAGE=vllm/vllm-openai:v0.29.0, as in the local pipeline.
Thinking effort defaults to high; judge MTP is disabled.
PHYS_DIR overrides the repository root (otherwise SLURM_SUBMIT_DIR in Slurm).
HF_CACHE_DIR defaults to /llms. JUDGE_PORT defaults to 41800; override it for
concurrent jobs on the same node. Docker and both images must exist on the node.
Override Slurm resources with sbatch options before the script path.
HELP
}

# Slurm executes a spool copy, so BASH_SOURCE cannot locate the repo in a job.
export PHYS_DIR="${PHYS_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
MODEL=""
INPUT_DIR=""
DATASETS=()
OPTIONS=()
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-high}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--generation-model|--generation-dir|--results-dir|--judge-workers|\
        --judge-tensor-parallel-size|--judge-model|--judge-reasoning-effort|--judge-program|--limit)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
                echo "$1 requires a value" >&2; exit 2;
            }
            case "$1" in
                --model|--generation-model) MODEL="$2" ;;
                --generation-dir) INPUT_DIR="$2" ;;
                --judge-reasoning-effort) JUDGE_REASONING_EFFORT="$2" ;;
                *) OPTIONS+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --datasets)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done
            [[ ${#DATASETS[@]} -gt 0 ]] || { echo "--datasets requires names" >&2; exit 2; }
            ;;
        --dry-run) OPTIONS+=(--dry-run); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" && -n "$INPUT_DIR" ]] || {
    echo "--model and --generation-dir are required" >&2; exit 2;
}
[[ "$INPUT_DIR" != /* && "/$INPUT_DIR/" != */../* ]] || {
    echo "--generation-dir must be below the repository root" >&2; exit 2;
}
cd "$PHYS_DIR"
MODEL_SLUG="$(python3 -c 'import re, sys; print(re.sub(r"[^A-Za-z0-9_.-]+", "--", sys.argv[1]).strip("-") or "local-llamacpp")' "$MODEL")"
MODEL_DIR="$INPUT_DIR/$MODEL_SLUG"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    for trace_file in "$MODEL_DIR"/*/traces.jsonl; do
        [[ -s "$trace_file" ]] || continue
        dataset_dir="${trace_file%/traces.jsonl}"
        DATASETS+=("${dataset_dir##*/}")
    done
fi
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "No saved traces under $MODEL_DIR" >&2; exit 2; }
for dataset in "${DATASETS[@]}"; do
    [[ "$dataset" != */* && "$dataset" != . && "$dataset" != .. && -s "$MODEL_DIR/$dataset/traces.jsonl" ]] || {
        echo "Missing or invalid dataset: $MODEL_DIR/$dataset/traces.jsonl" >&2; exit 2;
    }
done

# Docker selects host GPU IDs, whereas Slurm may remap CUDA_VISIBLE_DEVICES.
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
elif [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not expose allocated GPU IDs; set CUDA_VISIBLE_DEVICES to the allocated host GPU ID" >&2
    exit 2
fi
export JUDGE_SPECULATION=none
exec bash "$PHYS_DIR/src/moe_exp/correlation_pipeline/run_all.sh" \
    --skip-generate --skip-sampling --skip-forward --skip-analyze \
    --generation-model "$MODEL" --generation-dir "$INPUT_DIR" \
    --datasets "${DATASETS[@]}" --judge-reasoning-effort "$JUDGE_REASONING_EFFORT" "${OPTIONS[@]}"
