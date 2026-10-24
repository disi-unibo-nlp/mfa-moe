#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=240G --time=16:00:00 --no-requeue --job-name=capture-routers
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Capture the routers of one shard from its frozen manifest. A manifest whose shard is "verify" runs the 2-trace check
# that comes first. The model is the manifest's source:
#   sbatch scripts/capture/capture_routers.sbatch MANIFEST OUT_DIR
# The paper's manifests: /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/{A,B}/SHARD.json and MODEL/verify.json.
# A repeat goes to a new tree: OUT_DIR = ROOT/verify for the check, then ROOT/0, ROOT/1 ... for the shards (see src/capture/README.md).
# After "drained" (exit 75, the last 30 minutes are not used) or a failure, the same command resumes.
: "${1:?MANIFEST}" "${2:?OUT_DIR}"
test -f "$1" || { echo "no such manifest: $1" >&2; exit 1; }
source scripts/slurm/env.sh || exit 1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 VLLM_USE_V2_MODEL_RUNNER=0
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
srun python -B -u -m capture.capture_routers "$1" "$2" --deadline-epoch $END
