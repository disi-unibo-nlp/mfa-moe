"""Freeze serial/eager qualification after the observed simultaneous-layout failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.counterfactual import digest, sealed


def main():
    doc = REPO / 'report/experimental-resume-v1'
    old = sealed(doc / 'SCORE_API_CALIBRATION_MANIFEST_v3.json')
    failed_path = S / 'runs/routing-control-v1/score-api-calibration-b4aa1c878c25fe32/QUALIFICATION.json'
    failed = sealed(failed_path)
    if failed['pass'] or failed['completed'] != 80:
        raise ValueError('v3 failure must be preserved before a new execution profile')
    source = REPO / 'scripts/experimental_resume/run_score_api_calibration_v4_serial.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = S / 'addenda/score-api-serial' / source_sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError('serial wrapper snapshot changed')
    if not target.exists():
        target.write_bytes(source.read_bytes())
        target.chmod(0o400)
    body = {k: v for k, v in old.items() if k != 'sha256'}
    body.update(schema='routing-score-api-calibration-manifest-v4-serial',
        entry_driver=str(target), entry_driver_sha256=source_sha,
        engine_overrides={'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0},
        preserved_v3_failure=str(failed_path), preserved_v3_failure_sha256=failed['sha256'],
        source_manifest_sha256=old['sha256'],
        execution_change='One runnable request at a time, eager execution, identical frozen 80 fixtures and criteria; original MARLIN regime, no candidate/sample additions',
        code_files={**old['code_files'], str(target): source_sha})
    body['price'] = {**old['price'], 'prior_job_id': 59196354,
        'prior_total_allocation_seconds': 823, 'projected_seconds': 1.5 * 823 + 900 + 180,
        'rule': '1.5x measured entire v3 allocation plus 900s serial/full-vocabulary processing reserve and 180s safety; 45min2A100 ceiling covers2354.5s projection',
        'wall_minutes': 45, 'two_A100_allocation_GPU_hours': 1.5}
    value = {**body, 'sha256': digest(body)}
    path = doc / 'SCORE_API_CALIBRATION_MANIFEST_v4_serial.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('serial calibration manifest already differs')
    if not path.exists():
        path.write_text(json.dumps(value, indent=1) + '\n')
    directory.chmod(0o500)
    print(json.dumps({'manifest': str(path), 'sha256': value['sha256'],
        'entry_driver': str(target), 'entry_driver_sha256': source_sha, 'price': body['price']}))


if __name__ == '__main__':
    main()
