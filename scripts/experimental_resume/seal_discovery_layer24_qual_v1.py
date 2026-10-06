"""Seal separate layer24+28 hook/dose qualification before new intervention."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_discovery_feasibility_v1 as discovery

REPO = discovery.REPO
REPORT = discovery.REPORT
OUT = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_MANIFEST_v1.json'
PRICE = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_PRICE_v1.json'


def write_once(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('existing sealed qualification differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1) + '\n')
    return value


def main():
    pilot, amendment, dictionary = (base.sealed(p) for p in
                                    (discovery.PILOT, discovery.AMENDMENT, discovery.DICTIONARY))
    worker_prep, worker_qual = base.sealed(base.WORKER_PREP), base.sealed(base.WORKER_QUAL)
    overlay = Path(worker_prep['overlay'])
    codes = {str(Path(discovery.__file__)): base.file_sha(discovery.__file__),
             str(Path(base.__file__)): base.file_sha(base.__file__)}
    codes.update({str(overlay / name): base.file_sha(overlay / name)
                  for name in discovery.OVERLAY_FILES})
    body = {
        'schema': 'routing-discovery-feasibility-generation-v1',
        'stage': 'layer24_engineering_qual',
        'driver_sha256': base.file_sha(base.__file__),
        'entry_driver_sha256': base.file_sha(discovery.__file__),
        'code_files': codes,
        'source_pilot_sha256': pilot['sha256'],
        'amendment_sha256': amendment['sha256'],
        'action_dictionary_sha256': dictionary['sha256'],
        'base_tree_sha256': pilot['base_tree_sha256'],
        'family_freeze_sha256': amendment['family_freeze_sha256'],
        'qualified_worker_sha256': worker_qual['sha256'],
        'profile': {'max_num_seqs': 8, 'enforce_eager': False,
                    'VLLM_BATCH_INVARIANT': 0, 'batch_size': 8},
        'rows': pilot['rows'],
        'actions': discovery.action_spec(dictionary),
        'arms': [{'name': 'active', 'policy': 'condition', 'role': 'conditional'},
                 {'name': 'sentinel', 'policy': 'zero', 'role': 'native'}],
        'random_set_by_transition_family_seed': {
            'approach_to_commit': {r['family']: [i % 4, (i + 1) % 4]
                                   for i, r in enumerate(pilot['rows'])}},
        'seeds': [0, 1], 'max_tokens': 256,
        'scope': 'Engineering-only new layer24 action hook and dose on four invalid-semantic old candidate prefixes; layer28 remains present in same table; no semantic claim',
    }
    body.update(discovery.workload_counts(body))
    manifest = write_once(OUT, body)
    load, shutdown, factor = 685.0234088897705, 196., 1.25
    stress_decode, stress_prefill = 50., 1000.
    estimate = load + shutdown + factor * (body['expected_prefill_tokens'] / stress_prefill +
                                            body['maximum_decode_tokens'] / stress_decode)
    price_body = {
        'schema': 'routing-discovery-layer24-hook-qual-price-v1',
        'manifest_sha256': manifest['sha256'],
        'expected_requests': body['expected_requests'],
        'expected_batches': body['expected_batches'],
        'expected_prefill_tokens': body['expected_prefill_tokens'],
        'maximum_decode_tokens': body['maximum_decode_tokens'],
        'maximum_context_tokens': body['maximum_context_tokens'],
        'source_neighbor_profile_manifest': 'CAUSAL_BATCHED_NEIGHBOR_QUAL_MANIFEST_v1.json',
        'stress_aggregate_decode_tokens_per_second': stress_decode,
        'stress_prefill_tokens_per_second': stress_prefill,
        'repeat_factor': factor, 'cold_load_seconds': load,
        'shutdown_seconds': shutdown,
        'estimated_complete_wall_seconds': estimate,
        'requested_wall_seconds': 3600,
        'gpus': 2, 'gpu_hour_ceiling': 2,
        'status': 'engineering price and CPU preflight only; do not submit until ownership reconciliation and accepted neighbor result',
    }
    price = write_once(PRICE, price_body)
    print(json.dumps({'manifest': str(OUT), 'sha256': manifest['sha256'],
                      'price': str(PRICE), 'price_sha256': price['sha256'],
                      'requests': body['expected_requests'],
                      'batches': body['expected_batches'],
                      'estimate_seconds': estimate}))


if __name__ == '__main__':
    main()
