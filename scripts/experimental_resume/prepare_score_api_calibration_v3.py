"""Bounded metadata-only scorer amendment; no new examples or model execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
DOC = REPO / 'report/experimental-resume-v1'
BASE = S / 'code/s1-9a61e32f48c04c24'
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.counterfactual import digest, prediction_input, sealed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = sealed(DOC / 'COUNTERFACTUAL_QUALIFICATION_MANIFEST_v2.json')
    failed_path = S / 'runs/routing-control-v1/counterfactual-qualification-b7c4c5273ab6fdc7/QUALIFICATION.json'
    failed = sealed(failed_path)
    if failed['pass'] or failed['completed'] != 148 or not failed['recovery_pass']:
        raise ValueError('preserved failure does not match scorer amendment')
    policy_path = S / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067/policy-table.json'
    policy = sealed(policy_path)
    originals = [REPO / 'scripts/experimental_resume/run_score_api_calibration_v3.py',
        REPO / 'src/moe_exp/routing_control/counterfactual.py',
        REPO / 'src/moe_exp/routing_control/score_calibration.py',
        REPO / 'src/moe_exp/routing_control/receipts.py']
    code_sha = digest({p.name: sha(p) for p in originals})
    directory = S / 'addenda/score-api-calibration' / code_sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    files = {p: value for p, value in old['code_files'].items() if '/addenda/counterfactual/' not in p}
    for source, name in zip(originals, ('run_score_api_calibration_v3.py',
        'counterfactual_helpers.py', 'score_calibration.py', 'receipts.py')):
        target = directory / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('immutable calibration source differs')
        if not target.exists():
            target.write_bytes(source.read_bytes())
            target.chmod(0o400)
        files[str(target)] = sha(target)
    files[str(policy_path)] = sha(policy_path)
    if any(sha(p) != expected for p, expected in files.items()):
        raise ValueError('bound helper/worker changed')
    prefill = sum(4 * len(prediction_input(row, j)[0]) + len(prediction_input(row, j)[0]) + 1
        for row in old['rows'] for j in old['positions'])
    body = {'schema': 'routing-score-api-calibration-manifest-v3', 'stage': 'engineering',
        'base_tree_sha256': old['base_tree_sha256'], 'family_freeze_sha256': old['family_freeze_sha256'],
        'worker_binding': old['worker_binding'], 'worker_overlay': old['worker_overlay'],
        'driver': str(directory / originals[0].name), 'driver_sha256': sha(originals[0]),
        'code_files': files, 'source_manifest_sha256': old['sha256'],
        'preserved_failed_result': str(failed_path), 'preserved_failed_result_sha256': failed['sha256'],
        'rows': old['rows'], 'positions': old['positions'], 'requests': 80,
        'policy_table': str(policy_path), 'policy_table_sha256': policy['sha256'],
        'calibration_ratio': 1.25, 'calibration_epsilon': 1e-6,
        'criteria': 'max cross-API A/B mean and p99 absolute logprob difference <= 1.25 * larger designated/full-vocabulary repeat difference + 1e-6, identical prediction prefix; all requests inactive native k8 with both-rank identity/weight checks',
        'groups': 'four prefix groups of 16 with designated_A/full_vocab_A/designated_B/full_vocab_B interleaved at each position, then 16 appended-token diagnostic controls',
        'price': {'maximum_prefill_tokens_without_reuse': prefill, 'maximum_generated_tokens': 80,
            'full_vocabulary_reference_requests': 32, 'prior_job_id': 59189255,
            'prior_total_allocation_seconds': 815, 'projected_seconds': 1.5 * 815 + 600 + 180,
            'rule': '1.5x entire prior 148-request allocation including load/recompute/shutdown, plus 600s for 32 full-vocabulary references and 180s safety; this 80-request workload has fewer prefills',
            'wall_minutes': 45, 'two_A100_allocation_GPU_hours': 1.5},
        'interpretation': 'new prospective numerical API calibration; prior fixed absolute teacher-force failure stays failed; no semantic eligibility, expert ranking or engine equivalence inference'}
    result = {**body, 'sha256': digest(body)}
    manifest_path = DOC / 'SCORE_API_CALIBRATION_MANIFEST_v3.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != result:
        raise ValueError('v3 calibration manifest already differs')
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(result, indent=1) + '\n')
    directory.chmod(0o500)
    # Exercise all known constructors and request assignments before a cold load.
    sys.path.insert(0, str(Path(old['worker_overlay'])))
    sys.path.insert(0, str(BASE))
    from moe_steer import engine, policies as P
    table = P.PolicyTable.from_sealed(policy)
    if table.digest() != policy['sha256'] or table.hooked_layers() != (28,):
        raise ValueError('pilot policy table invalid')
    spec = importlib.util.spec_from_file_location('score_driver_cpu_smoke', body['driver'])
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assignments = driver.cases(result)
    if len({meta['uid'] for _, meta in assignments}) != 80:
        raise ValueError('invalid assignment UIDs')
    if engine.code_tree_sha256() != old['base_tree_sha256']:
        raise ValueError('base engine tree changed')
    print(json.dumps({'manifest': str(manifest_path), 'sha256': result['sha256'],
        'driver': body['driver'], 'driver_sha256': body['driver_sha256'],
        'constructor_preflight': 'PASS', 'requests': len(assignments), 'price': body['price']}))


if __name__ == '__main__':
    main()
