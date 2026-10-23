#!/bin/bash
# Class occupancy, dwell, transitions and expert nominations of the seven campaign-v3 models, from the saved sentences.
# ROOT must be new; it receives all outputs:
#   scripts/routing/dynamics.sh ROOT
sbatch -J dynamics scripts/slurm/cpu.sh routing.dynamics report/research-campaign-v3-20261014-r5-r2 gpt nemotron qwen35 qwen36 gemma glm qwen330b --root $1
