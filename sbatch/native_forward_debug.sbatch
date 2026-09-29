#!/bin/bash
#SBATCH --job-name=mfa-forward-debug
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/forward-debug-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/forward-debug-%j.err
set -euo pipefail
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
SOURCE="${1:?Usage: native_forward_debug.sbatch qwen330b|glm|qwen36 CPU_GATE [--plan-only]}"
CPU_GATE="${2:?Pass the successful CPU gate JSON}"
[[ "$SOURCE" == qwen330b || "$SOURCE" == glm || "$SOURCE" == qwen36 ]] || exit 2
VALIDATION=forward-validation-v1
[[ "$SOURCE" != qwen36 ]] || VALIDATION=forward-validation-v2
PANEL="$REPO/results/correlation_pipeline/$VALIDATION/panels/$SOURCE"
RUN="$REPO/results/correlation_pipeline/$VALIDATION/runs/$SOURCE-${SLURM_JOB_ID:-unsubmitted}"
ARGV=("$P/envs/vllm-cu129/bin/python" -m moe_exp.correlation_pipeline.forward_debug
      run --source "$SOURCE" --panel-root "$PANEL" --output-root "$RUN" --cpu-gate "$CPU_GATE")
EXTRA=()
PLAN_ONLY=false
LOADER_ONLY=false
for argument in "${@:3}"; do
    case "$argument" in
        --plan-only) PLAN_ONLY=true ;;
        --loader-only) LOADER_ONLY=true ;;
        *) EXTRA+=("$argument") ;;
    esac
done
ARGV+=("${EXTRA[@]}")
if $PLAN_ONLY; then
    if $LOADER_ONLY; then
        printf '%q ' srun --ntasks=1 "$P/envs/vllm-cu129/bin/python" -m pytest \
            tests/test_qwen_quantized.py -k cuda
    else
        printf '%q ' srun --ntasks=1 --nodes=1 "${ARGV[@]}"
    fi
    printf '\n'
    exit 0
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit through Slurm after approval" >&2; exit 2; }
[[ "$(id -un)" == lmolfett && "$(hostname)" == lrdn* ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
[[ -f "$PANEL/panel.json" && -f "$CPU_GATE" ]] || exit 2
module load python/3.11.7 cuda/12.6
umask 077
mkdir -p "$RUN/runtime/tmp" "$RUN/runtime/xdg" "$RUN/runtime/hf"
export PYTHONPATH="$REPO/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$RUN/runtime/hf" HF_HUB_CACHE="$P/cache/hf/hub"
export XDG_CACHE_HOME="$RUN/runtime/xdg" TMPDIR="$RUN/runtime/tmp"
export TORCH_HOME="$RUN/runtime/torch" TRITON_CACHE_DIR="$RUN/runtime/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/runtime/inductor" DSPY_CACHEDIR="$RUN/runtime/dspy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$P/envs/vllm-cu129/bin:$PATH"
cd "$REPO"
"$P/envs/correlation-client-3.11/bin/python" -c \
    'import json,sys; from moe_exp.correlation_pipeline.provenance import code_provenance; g=json.load(open(sys.argv[1])); assert g["status"] == "passed" and g["code"] == code_provenance()' "$CPU_GATE"
if [[ "$SOURCE" == qwen36 ]]; then
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    srun --ntasks=1 --nodes=1 "$P/envs/vllm-cu129/bin/python" -m pytest -x -q -p no:cacheprovider \
        tests/test_qwen_quantized.py -k cuda --basetemp "$RUN/runtime/tmp/pytest" \
        --junitxml "$RUN/loader-tests.xml"
    "$P/envs/correlation-client-3.11/bin/python" -c \
        'import json,sys,xml.etree.ElementTree as E; from pathlib import Path; p=Path(sys.argv[1]); cases=E.parse(p/"loader-tests.xml").findall(".//testcase"); assert len(cases)==5 and all(c.find("skipped") is None and c.find("failure") is None and c.find("error") is None for c in cases); r={"status":"complete","passed_tests":len(cases),"stage":"tiny_cuda_quantized_loader"}; (p/"loader-acceptance.json").write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r))' "$RUN"
fi
if $LOADER_ONLY; then
    [[ "$SOURCE" == qwen36 ]] || exit 2
    exit 0
fi
srun --ntasks=1 --nodes=1 "${ARGV[@]}"
