"""CPU Slurm entrypoint for sealed 1,024-token dense trajectory receipts.

The input must already join generation receipts to arm-blind offline labels and
semantic reader votes. This program performs no labeling or GPU inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moe_exp.routing_control.trajectory_analysis_v1 import analyze, verify_seal, digest

OWNED = Path("/leonardo_work/IscrC_MIOSR/lmolfett")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("full trajectory analysis requires a CPU Slurm allocation")
    if not args.input.resolve().is_relative_to(OWNED) or not args.out_dir.resolve().is_relative_to(OWNED):
        raise ValueError("trajectory artifacts must stay in owned LEONARDO work storage")
    value = verify_seal(json.loads(args.input.read_text()))
    result = analyze(value)
    module_dir = REPO / "src/moe_exp/routing_control"
    code_files = [Path(__file__).resolve(), module_dir / "trajectory_analysis_v1.py",
                  module_dir / "analysis.py", module_dir / "design.py"]
    code_hashes = {str(path): file_sha(path) for path in code_files}
    code_digest = digest(code_hashes)
    binding = {"schema": "dense-trajectory-analysis-binding-v1",
               "input_sha256": value["sha256"], "input_file_sha256": file_sha(args.input),
               "code_files": code_hashes, "code_digest": code_digest}
    binding["sha256"] = digest(binding)
    directory = args.out_dir / ("dense-trajectory-v1-" + value["sha256"][:12] + "-" +
                                code_digest[:12])
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (("BINDING.json", binding), ("RESULT.json", result)):
        path = directory / name
        if path.exists():
            old = json.loads(path.read_text())
            if old != payload:
                raise ValueError("same-bound trajectory artifact differs: " + str(path))
        else:
            pending = directory / ("." + name + ".pending-" + os.environ["SLURM_JOB_ID"])
            with pending.open("w") as stream:
                json.dump(payload, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, path)
    print(json.dumps({"status": "COMPLETE", "result": str(directory / "RESULT.json"),
                      "sha256": result["sha256"], "assignments": result["assignments"]}))


if __name__ == "__main__":
    main()
