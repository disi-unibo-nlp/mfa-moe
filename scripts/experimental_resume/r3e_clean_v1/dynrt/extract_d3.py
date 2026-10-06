"""D3 stage 1: sentence tables and histograms for the replication models (same code path as D1).

Thin wrapper over `dynrt.extract.run` (frozen D1 code) that accepts any campaign-v3 model.
CLI: python -m dynrt.extract_d3 MODEL COHORT [--workers W]
"""
from __future__ import annotations

import argparse
import os
import sys

from .common import RESULTS, cohort_dir
from .extract import run

MODELS = ("gemma", "glm", "nemotron", "qwen330b", "qwen35")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("cohort", choices=("A", "B"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args(argv)
    checks = run(a.model, a.cohort, cohort_dir(a.model, a.cohort, RESULTS), workers=a.workers, limit=None,
                 crosscheck=True)
    # `unknown_label` records (judge output outside the 7 classes) are dropped exactly as v3an drops them; the
    # comparisons against v3an's alignment_audit / class_token_map (the *_mismatch checks) must still be zero.
    failed = [c for c in checks["failed_checks"] if c != "unknown_label"]
    print(f"benign unknown_label records: {checks['unknown_label']}; failed checks: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
