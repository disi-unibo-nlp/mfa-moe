#!/usr/bin/env bash
# Regenerate one matched baseline; preserve and compare both saved guided runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# The incompatible baseline copies must have been archived before this script.
# Resume is safe; compatible completed baselines and guided runs are skipped.
TEMPLATE_DATE=2026-09-22 MODEL_PROFILE=oss STRENGTH=2 MAX_NUM_SEQS=25 \
OUTPUT_ROOT=results/moe_identity_guiding/oss_strength_2 \
bash src/moe_exp/moe_identity_guiding/run_global.sh all full

# Reuses the new baseline after checking every rendered prompt and token ID.
TEMPLATE_DATE=2026-09-22 MODEL_PROFILE=oss STRENGTH=1 MAX_NUM_SEQS=25 \
OUTPUT_ROOT=results/moe_margin_guiding/oss_strength_1 \
bash src/moe_exp/moe_margin_guiding/run_global.sh all full
