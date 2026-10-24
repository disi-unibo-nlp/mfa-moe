#!/usr/bin/env python3
"""Run a frozen, hash-verified CPU reporting bundle with importable workers."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

def main():
    bundle = Path(__file__).resolve().parent
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm required")
    for rel, expected in json.loads((bundle / "SHA256.json").read_text()).items():
        if hashlib.sha256((bundle / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Changed reporting code: " + rel)
    for path, expected in json.loads((bundle / "campaign.json").read_text())["runtime_code_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Changed shared reducer: " + path)
    sys.path.insert(0, str(bundle))
    source = sys.argv[1]
    if source == "all":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        module = "measurement_study.production_report"
    else:
        sys.argv = [sys.argv[0], source, "--workers", "6", *sys.argv[2:]]
        module = "measurement_study.production_supplement"
    importlib.import_module(module).main()

if __name__ == "__main__":
    main()
