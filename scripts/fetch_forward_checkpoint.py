"""Fetch the approved Qwen forward snapshot; never alter an existing cached file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

MODEL = "unsloth/Qwen3.6-35B-A3B"
CACHE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub")
REPORT = Path(__file__).resolve().parents[1] / "results/correlation_pipeline/forward-validation-v2/checkpoint.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, record):
    if path.stat().st_size != record["size"]:
        raise RuntimeError(f"Incorrect checkpoint size: {path}")
    expected = (record.get("lfs") or {}).get("sha256")
    actual = sha256(path)
    if expected:
        if actual != expected:
            raise RuntimeError(f"Incorrect checkpoint SHA256: {path}")
    else:
        content = path.read_bytes()
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        if hashlib.sha1(blob).hexdigest() != record["blobId"]:
            raise RuntimeError(f"Incorrect checkpoint Git blob hash: {path}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-https", action="store_true",
                        help="Requires approval to use direct HTTPS instead of the data mover")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    report_path = args.report
    model_id = args.model
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["model"] != model_id:
            raise ValueError("Checkpoint report belongs to another model")
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}?blobs=true", timeout=30) as stream:
            meta = json.load(stream)
        files = [r for r in meta["siblings"] if r["rfilename"].endswith(
            (".safetensors", ".json", ".jinja", "merges.txt")
        ) and "/" not in r["rfilename"]]
        report = {"model": model_id, "revision": meta["sha"], "files": files, "status": "planned"}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+"\n")
    snapshot = CACHE / ("models--" + model_id.replace("/", "--")) / "snapshots" / report["revision"]
    print(json.dumps({"model": model_id, "revision": report["revision"], "snapshot": str(snapshot),
                      "bytes": sum(f["size"] for f in report["files"]), "files": len(report["files"])}), flush=True)
    if not args.download_https:
        return
    snapshot.mkdir(parents=True, exist_ok=True)
    for record in report["files"]:
        name = record["rfilename"]
        target = snapshot / name
        expected = (record.get("lfs") or {}).get("sha256")
        if target.exists():
            record["download_sha256"] = verify(target, record)
            print(f"REUSED {name}", flush=True)
            continue
        partial = target.with_name(name + ".partial")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", "--connect-timeout", "30", "--max-time", "1800",
                        f"https://huggingface.co/{model_id}/resolve/{report['revision']}/{name}",
                        "--output", str(partial)], check=True)
        actual = verify(partial, record)
        record["download_sha256"] = actual
        os.replace(partial, target)
        print(f"VERIFIED {name} {record['size']}", flush=True)
        report_path.write_text(json.dumps(report, indent=2)+"\n")
    report["status"] = "complete"
    report["snapshot"] = str(snapshot)
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    assert json.loads(report_path.read_text()) == report
    print("CHECKPOINT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
