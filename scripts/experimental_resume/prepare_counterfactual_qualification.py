"""Freeze CPU-native fixtures and complete price; never perform inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.counterfactual import digest, sealed, prediction_input, pricing


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sealed(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('immutable preparation differs: ' + str(path))
    if not path.exists():
        path.write_text(json.dumps(value, indent=1) + '\n')
    return value


def trace_at(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location['path']).open('rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('native raw trace changed')
    return json.loads(raw)


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native fixture extraction requires CPU Slurm')
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    scout_path = S / 'runs/routing-control-v1/dense-discovery/CANDIDATE_PREFIX_SCOUT_v1.json'
    pilot_path = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
    price_path = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_FULL_STAGE_PRICE_v1.json'
    prep_path = S / 'runs/ordered-qualification-v1/PREPARED.cpu-recovery-v2.json'
    qual_path = S / 'runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json'
    family_path = REPO / 'report/experimental-resume-v1/family-freeze.json'
    scout, pilot, old_price, worker, qual, families = map(sealed,
        (scout_path, pilot_path, price_path, prep_path, qual_path, family_path))
    if not qual['pass'] or pilot['qualified_worker_sha256'] != qual['sha256']:
        raise ValueError('passing worker qualification does not match pilot')
    allowed = set(families['new_parent_pools']['parent_pools']['discovery'])
    selected = {r['uid'] for r in pilot['rows']}
    sources = sorted([r for r in scout['records'] if r['uid'] in selected], key=lambda r: r['uid'])
    if len(sources) != 4 or len({r['family'] for r in sources}) != 4 or not {r['family'] for r in sources} <= allowed:
        raise ValueError('exact four fixed engineering discovery prefixes required')
    attempts_path = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
    attempts = pd.read_parquet(attempts_path, columns=['attempt_id', 'source_location', 'trace_sha256'])
    by_attempt = {str(r['attempt_id']): r for r in attempts.to_dict('records')}
    rows = []
    for r in sources:
        source = by_attempt[r['attempt_id']]
        trace = trace_at(source['source_location'])
        if trace_digest(TraceRecord(**trace)) != source['trace_sha256'] or source['trace_sha256'] != r['trace_sha256']:
            raise ValueError('native trace binding changed')
        replay = trace['metadata']['token_replay']
        ids = replay['completion_token_ids']
        n = len(r['prefix_ids'])
        if ids[:n] != r['prefix_ids'] or replay['prompt_token_ids'] != r['prompt_ids'] or len(ids) < n + 4:
            raise ValueError('native prefix or four-token continuation mismatch')
        suffix = ids[n:n + 4]
        if 248069 in r['prefix_ids'] or 248069 in suffix:
            raise ValueError('fixed engineering window is already reasoning-closed')
        rows.append({k: r[k] for k in ('uid', 'family', 'question', 'prompt_ids', 'prefix_ids', 'trace_sha256')} |
            {'native_suffix_ids': suffix})

    original_driver = Path(__file__).with_name('run_counterfactual_qualification.py')
    helper = REPO / 'src/moe_exp/routing_control/counterfactual.py'
    receipts = REPO / 'src/moe_exp/routing_control/receipts.py'
    code_digest = digest({p.name: sha(p) for p in (original_driver, helper, receipts)})
    frozen_dir = S / 'addenda/counterfactual' / code_digest[:16]
    frozen_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for original, name in ((original_driver, original_driver.name), (helper, 'counterfactual_helpers.py'),
                           (receipts, 'receipts.py')):
        destination = frozen_dir / name
        if destination.exists() and destination.read_bytes() != original.read_bytes():
            raise ValueError('frozen loss-screen code differs')
        if not destination.exists():
            destination.write_bytes(original.read_bytes())
            destination.chmod(0o400)
        files[str(destination)] = sha(destination)
    files.update(worker['files'])
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('source or qualified worker changed')

    random_pairs = [[120, 139], [133, 255], [5, 43], [24, 196]]
    arms = [
        {'name': 'native', 'operator': 'none', 'ids': [9, 189]},
        {'name': 'native_repeat', 'operator': 'none', 'ids': [9, 189]},
        {'name': 'target_bias0.5', 'operator': 'bias', 'magnitude': .5, 'ids': [9, 189]},
        {'name': 'target_bias1', 'operator': 'bias', 'magnitude': 1., 'ids': [9, 189]},
        {'name': 'target_force_positive', 'operator': 'force_positive', 'ids': [9, 189]},
        {'name': 'target_force_negative', 'operator': 'force_negative', 'ids': [9, 189]},
        {'name': 'random_force_positive', 'operator': 'force_positive', 'ids': 'by_prefix'},
        {'name': 'random_force_negative', 'operator': 'force_negative', 'ids': 'by_prefix'},
    ]
    score_prefill = sum(len(prediction_input(r, j)[0]) * len(arms) for r in rows for j in range(4))
    tf_prefill = sum(len(prediction_input(r, j)[0]) + 1 for r in rows for j in range(4))
    closed_prefill = 4 * (len(rows[0]['prompt_ids']) + len(rows[0]['prefix_ids']) + 1)
    requests = 4 * 4 * 8 + 16 + 4
    price = pricing(score_prefill + tf_prefill + closed_prefill, requests,
        pilot['expected_prefill_tokens'], pilot['maximum_decode_tokens'],
        old_price['pilot_fixed_in_driver_seconds'], old_price['pilot_batch_seconds'],
        old_price['pilot_outside_driver_seconds_including_shutdown'])
    body = {'schema': 'routing-counterfactual-qualification-v1', 'stage': 'engineering',
        'base_tree_sha256': pilot['base_tree_sha256'],
        'family_freeze_sha256': families['sha256'], 'prefix_scout_sha256': scout['sha256'],
        'pilot_manifest_sha256': pilot['sha256'], 'worker_qualification_sha256': qual['sha256'],
        'worker_binding': str(prep_path), 'worker_overlay': worker['overlay'],
        'driver': str(frozen_dir / original_driver.name), 'driver_sha256': sha(original_driver),
        'code_files': files, 'rows': rows, 'arms': arms, 'random_pairs': random_pairs,
        'positions': [0, 1, 2, 3], 'requests': requests, 'native_top_k': 8,
        'shared_expert': 'unchanged', 'logprobs_mode': 'raw_logprobs',
        'preemption_calls': [2], 'teacher_force_alignment_mean_tolerance': 1e-4,
        'teacher_force_alignment_max_tolerance': 1e-3, 'price': price,
        'population': 'four fixed discovery engineering prefixes; semantic eligibility unestablished',
        'interpretation': 'single-position local loss proxy, no semantic steering or correctness result'}
    manifest = write_sealed(REPO / 'report/experimental-resume-v1/COUNTERFACTUAL_QUALIFICATION_MANIFEST_v1.json', body)
    frozen_dir.chmod(0o500)
    print(json.dumps({'manifest_sha256': manifest['sha256'], 'requests': requests, 'price': price,
        'driver': body['driver'], 'status': 'PREPARED_ENGINEERING_NOT_SEMANTIC'}))


if __name__ == '__main__':
    main()
