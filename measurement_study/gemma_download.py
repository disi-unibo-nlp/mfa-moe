"""Sequential, resumable HTTPS recovery of the manifest-pinned Gemma checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPOSITORY = "nvidia/Gemma-4-26B-A4B-NVFP4"
REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
DESTINATION = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/native-replay/Gemma-4-26B-A4B-NVFP4") / REVISION

def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    h = hashlib.sha256() if item.get("sha256") else hashlib.sha1()
    if not item.get("sha256"):
        h.update(f"blob {item['bytes']}\0".encode())
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest() == (item.get("sha256") or item["git_blob_id"])

def fetch(item, destination, run=subprocess.run):
    path = destination / item["name"]
    partial = path.with_name(path.name + ".partial")
    if verify(path, item):
        return "reused_verified"
    if path.exists():
        raise ValueError("Existing published file failed verification: " + item["name"])
    if partial.exists() and partial.stat().st_size > item["bytes"]:
        raise ValueError("Oversized partial: " + item["name"])
    if not (partial.exists() and partial.stat().st_size == item["bytes"]):
        run(["curl", "--fail", "--location", "--silent", "--show-error",
             "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2",
             "--connect-timeout", "30", "--max-time", "1800",
             "--continue-at", "-", "--output", str(partial), item["url"]], check=True)
    if not verify(partial, item):
        raise ValueError("Downloaded partial failed size/hash verification: " + item["name"])
    partial.replace(path)
    return "downloaded_verified"

def validate_index(destination, items):
    byname = {x["name"]: x for x in items}
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    expected = {n for n in byname if n.endswith(".safetensors")}
    if not shards or shards != expected:
        raise ValueError("Shard index does not cover the complete manifest")
    for name in shards:
        if not verify(destination / name, byname[name]):
            raise ValueError("Missing or corrupt indexed shard: " + name)
    return sorted(shards)

def main():
    p = argparse.ArgumentParser(); p.add_argument("manifest", type=Path); p.add_argument("--metadata-only", action="store_true"); a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    assert m["repository"] == REPOSITORY and m["revision"] == REVISION
    assert Path(m["destination"]) == DESTINATION
    assert len(m["files"]) == 10 and sum(x["bytes"] for x in m["files"]) == 18825668663
    names = [x["name"] for x in m["files"]]
    assert len(set(names)) == len(names)
    for x in m["files"]:
        assert Path(x["name"]).name == x["name"]
        assert x["url"] == f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{x['name']}"
        assert x.get("sha256") or x.get("git_blob_id")
    os.umask(0o077); DESTINATION.mkdir(parents=True, exist_ok=True)
    records = []
    # Metadata first, still strictly one transfer at a time.
    for x in sorted(m["files"], key=lambda x: x["name"].endswith(".safetensors")):
        if a.metadata_only and x["name"].endswith(".safetensors"): continue
        print(json.dumps({"file": x["name"], "status": "verifying_or_resuming"}), flush=True)
        status = fetch(x, DESTINATION)
        records.append(dict(name=x["name"], bytes=x["bytes"], status=status))
        print(json.dumps(records[-1]), flush=True)
    receipt = dict(status="metadata_verified" if a.metadata_only else "verified", repository=REPOSITORY, revision=REVISION,
                   transport="direct HTTPS with TLS validation", manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                   historical_generation_revision_status="unverified", files=records, verified_at=time.time())
    if not a.metadata_only: receipt["indexed_shards"] = validate_index(DESTINATION, m["files"])
    target = DESTINATION / ("metadata-receipt.json" if a.metadata_only else "transfer-receipt.json")
    tmp = target.with_suffix(".partial"); tmp.write_text(json.dumps(receipt, indent=2)+"\n"); tmp.replace(target)

if __name__ == "__main__": main()
