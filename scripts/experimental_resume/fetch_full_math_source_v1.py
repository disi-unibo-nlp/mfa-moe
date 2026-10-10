"""Bounded public full-MATH test fetch, pinned before any outcome generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
DATASET = "EleutherAI/hendrycks_math"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
MAX_BYTES = 16 << 20
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report/routing-study-v1/full-math-source"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, total = [], 0
    for subject in SUBJECTS:
        path = OUT / (subject + ".parquet")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{subject}/test-00000-of-00001.parquet"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
                raise ValueError("unexpected parquet response or bounded transfer exceeded")
            with path.open("xb") as stream: stream.write(data)
        data = path.read_bytes()
        total += len(data)
        if total > MAX_BYTES: raise ValueError("bounded source transfer exceeded")
        records.append({"subject": subject, "path": str(path), "url": url, "bytes": len(data),
                        "file_sha256": hashlib.sha256(data).hexdigest()})
    body = {"schema": "routing-study-full-math-source-v1", "dataset": DATASET,
            "revision": REVISION, "split": "test", "files": records, "bytes": total,
            "transfer_ceiling_bytes": MAX_BYTES, "contains_generated_outcomes": False}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    value = {**body, "sha256": hashlib.sha256(canonical).hexdigest()}
    path = OUT / "SOURCE.json"
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError("source receipt changed")
    else:
        with path.open("x") as stream: json.dump(value, stream, indent=1); stream.write("\n")
    if json.loads(path.read_text()) != value: raise ValueError("source receipt readback failed")
    print(json.dumps({"source": str(path), "revision": REVISION, "files": len(records), "bytes": total}))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
