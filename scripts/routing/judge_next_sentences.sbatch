#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=04:00:00 --job-name=pairs-judge
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# One shard of the next-sentence judge of one model (a rerun resumes); scripts/routing/label_next_sentences.sh submits them:
#   sbatch scripts/routing/judge_next_sentences.sbatch STUDY_DIR HF_MODEL I/N [SOURCE_CLASSES]
source scripts/slurm/env.sh || exit 1
# As scripts/slurm/run_module_gpu.sbatch (the LFM2.5 labels): a few BLAS threads next to the engine. The renderer child finds DSPy
# in the pinned copy the study names (labeling.renderer); the judge-client environment is not put on PYTHONPATH, where
# its openai and pydantic would shadow the vLLM environment's.
T=$((SLURM_CPUS_PER_TASK < 4 ? SLURM_CPUS_PER_TASK : 4))
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
python -m routing.next_sentence_labels judge --study "${1:?STUDY_DIR}" --model "${2:?HF_MODEL}" --shard "${3:?I/N}" ${4:+--source-classes "$4"}
