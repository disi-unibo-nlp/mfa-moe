"""Bounded metadata-only correction of the TargetSet constructor; no new fixtures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.counterfactual import digest, sealed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = sealed(REPO / 'report/experimental-resume-v1/COUNTERFACTUAL_QUALIFICATION_MANIFEST_v1.json')
    originals = [REPO / 'scripts/experimental_resume/run_counterfactual_qualification.py',
        REPO / 'src/moe_exp/routing_control/counterfactual.py',
        REPO / 'src/moe_exp/routing_control/receipts.py']
    code_sha = digest({p.name: sha(p) for p in originals})
    directory = S / 'addenda/counterfactual' / code_sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    files = {p: value for p, value in old['code_files'].items() if '/addenda/counterfactual/' not in p}
    for source, name in zip(originals, ('run_counterfactual_qualification.py', 'counterfactual_helpers.py', 'receipts.py')):
        target = directory / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('immutable correction code differs')
        if not target.exists():
            target.write_bytes(source.read_bytes())
            target.chmod(0o400)
        files[str(target)] = sha(target)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('qualified worker changed')
    body = {k: v for k, v in old.items() if k != 'sha256'}
    body.update(driver=str(directory / originals[0].name), driver_sha256=sha(originals[0]), code_files=files,
        superseded_manifest_sha256=old['sha256'], correction='TargetSet requires explicit CUSTOM scope and source; caught by CPU policy-table smoke before any GPU submission; fixtures, arms, price and criteria unchanged')
    result = {**body, 'sha256': digest(body)}
    target = REPO / 'report/experimental-resume-v1/COUNTERFACTUAL_QUALIFICATION_MANIFEST_v2.json'
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError('v2 correction already differs')
    if not target.exists():
        target.write_text(json.dumps(result, indent=1) + '\n')
    directory.chmod(0o500)
    print(json.dumps({'manifest': str(target), 'sha256': result['sha256'], 'driver': body['driver'],
        'driver_sha256': body['driver_sha256'], 'requests': result['requests'], 'price': result['price']}))


if __name__ == '__main__':
    main()
