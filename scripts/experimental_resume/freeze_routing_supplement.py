"""Bind the saved-result routing-profile supplement to immutable reporting code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/paper/resume-v1')


def run():
    source = REPO/'scripts/experimental_resume/routing_supplement.py'
    data = source.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    directory = ROOT/('code-routing-supplement-'+sha[:16])
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory/source.name
    if driver.exists() and driver.read_bytes() != data:
        raise ValueError('immutable supplement driver changed')
    if not driver.exists():
        driver.write_bytes(data)
    body = {'schema': 'routing-supplement-code-v1', 'driver': str(driver),
            'driver_sha256': sha}
    path = ROOT/'CODE_ROUTING_SUPPLEMENT.json'
    if path.exists() and json.loads(path.read_text()) != body:
        raise ValueError('different supplement code already frozen')
    if not path.exists():
        path.write_text(json.dumps(body, indent=1)+'\n')
    print(json.dumps({'path': str(path), 'driver_sha256': sha}))


if __name__ == '__main__':
    run()
