# Where the judge client environment (DSPy and GEPA) lives: its folder is named after the hash of its lock file.
# Sourced from the repo root by build_judge_client_env.sh, which builds it, and by scripts/slurm/env.sh, which exports it as $DSPY.
DSPY_LOCK=requirements/qwen-fullprecision-smoke.lock
DSPY=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-$(sha256sum $DSPY_LOCK | cut -c1-16)
