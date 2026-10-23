#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=240G --time=16:00:00 --no-requeue --job-name=capture
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Capture the routers of one shard of one model from its frozen manifest, or ("verify") run the 2-trace check that comes first:
#   sbatch scripts/capture/capture.sbatch MODEL MANIFEST OUTPUT_DIR [verify]
# The paper's manifests: /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/{A,B}/SHARD.json and MODEL/verify.json.
# A repeat goes to a new tree: OUTPUT_DIR = ROOT/verify for the check, then ROOT/0, ROOT/1 ... for the shards (see src/capture/PORTED.md).
# After "drained" (exit 75, the last 30 minutes are not used) or a failure, the same command resumes.
[ -z "${4:-}" ] || [ "$4" = verify ] || { echo "4th argument must be the word verify, got: $4" >&2; exit 2; }
source scripts/slurm/env.sh || exit 1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 VLLM_USE_V2_MODEL_RUNNER=0
if [ "$1" = gpt ]; then  # transformers loads gpt-oss as MXFP4 only with kernels 0.16 (the vLLM env has 0.11 and would dequantize)
  export LOCAL_KERNELS=kernels-community/gpt-oss-triton-kernels=$MFA/cache/hf/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots/0f351046d799bc4d2c9dbb2c7cf36753204b5916
  export PYTHONPATH=$PYTHONPATH:$MFA/envs/forward-kernels-0.16
fi
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
srun python -B -u -m capture.run "$1" "$2" "$3" --deadline-epoch $END ${4:+--verify-only}
