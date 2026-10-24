#!/bin/bash
# Start the vLLM server of one model on this job's port ($PORT, set by scripts/slurm/env.sh); you do not run it by hand,
# a job calls start_server HF_MODEL from env.sh:
#   scripts/serve/serve_model.sh HF_MODEL        e.g. openai/gpt-oss-20b
# The flags of every model come from one table, src/generation/serve_flags.py.
set -euo pipefail
flags=$(python -m generation.serve_flags "${1:?HF_MODEL}")
mapfile -t args <<< "$flags"
exec vllm serve "${args[@]}"
