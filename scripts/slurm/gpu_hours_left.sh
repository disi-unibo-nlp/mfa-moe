#!/bin/bash
# GPU hours left on the account quota, read from CINECA's saldo (login node, no sbatch):
#   scripts/slurm/gpu_hours_left.sh
set -euo pipefail
BILLING_PER_GPU_HOUR=8  # on the Booster one GPU bills 8 local hours per hour (a 2-GPU, 16-core job bills 16 per hour)
/cineca/bin/saldo -b | awk -v per_gpu=$BILLING_PER_GPU_HOUR '
  $4 ~ /^[0-9]+$/ {  # account start end total localConsumed totConsumed percent monthTotal monthConsumed
    left = $4 - $6; found = 1
    printf "%s: %d billing hours left of %d (= %d GPU hours at %d per GPU), budget ends %s-%s-%s\n",
           $1, left, $4, left / per_gpu, per_gpu, substr($3, 1, 4), substr($3, 5, 2), substr($3, 7, 2)
    printf "%s this month: %d billing hours consumed against a monthly share of %d\n", $1, $9, $8
  }
  END { if (!found) { print "saldo -b printed no account row" > "/dev/stderr"; exit 1 } }'
