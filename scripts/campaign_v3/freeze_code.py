"""Freeze the reviewed campaign-v3 code into an immutable snapshot that every job runs from.

Copies src/, scripts/campaign_v3/ and sbatch/ of the live checkout into
<campaign>/code/t1-<sha16>/, writes MANIFEST.json (per-file sha256 plus the tree hash)
and makes the copy read-only. Re-running on an unchanged tree returns the same snapshot;
a changed tree gets a new directory, never an overwrite.

    python freeze_code.py [--repo REPO] [--campaign DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
CAMPAIGN = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3")
TREES = ("src", "scripts/campaign_v3", "sbatch")


def tree_files(repo: Path) -> dict[str, str]:
    files = {}
    for tree in TREES:
        for path in sorted((repo / tree).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def freeze(repo: Path, campaign: Path) -> dict:
    files = tree_files(repo)
    tree_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = campaign / "code" / f"t1-{tree_sha[:16]}"
    manifest = dict(schema_version=1, source=str(repo), tree_sha256=tree_sha, trees=list(TREES), files=files)
    if target.exists():
        if json.loads((target / "MANIFEST.json").read_text()) != manifest or tree_files(target) != files:
            raise SystemExit(f"existing snapshot {target} differs; refusing to reuse it")
        return dict(snapshot=str(target), tree_sha256=tree_sha, reused=True)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    for relative in files:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    if tree_files(staging) != files:
        raise SystemExit("source tree changed while it was being copied; retry")
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    os.replace(staging, target)
    for path in sorted(target.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    target.chmod(target.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return dict(snapshot=str(target), tree_sha256=tree_sha, reused=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--campaign", type=Path, default=CAMPAIGN)
    args = ap.parse_args(argv)
    print(json.dumps(freeze(args.repo, args.campaign)))


if __name__ == "__main__":
    main()
