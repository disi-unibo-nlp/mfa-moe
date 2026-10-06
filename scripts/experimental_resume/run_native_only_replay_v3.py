"""Qualified native-request replay under the pilot's exact hooked policy table.

Uses the immutable six-arm runner's batch, telemetry, crash and output logic;
the separate sealed manifest binds this wrapper and contains only two zero
policy REQUEST arms per frozen family/seed.  The unused target/random policies
retain the pilot's layer-28 hooks, which the plugin requires.  This diagnoses native variability before
attributing it to edited neighbors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_boundary_micro_screen as base

ORIGINAL_BUILD_POLICY_TABLE = base.build_policy_table


def validate_replay(manifest, driver_path):
    from moe_steer import policies as P
    if manifest.get('schema') != 'routing-native-only-replay-v3':
        raise ValueError('wrong native replay schema')
    if manifest.get('driver_sha256') != base.file_sha(driver_path):
        raise ValueError('base runner changed')
    if manifest.get('replay_driver_sha256') != base.file_sha(__file__):
        raise ValueError('native replay wrapper changed')
    if manifest.get('base_tree_sha256') != base.REQUIRED_BASE_TREE:
        raise ValueError('base sampler tree changed')
    pilot = base.sealed(Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/'
                             'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'))
    if manifest.get('source_pilot_sha256') != pilot['sha256']:
        raise ValueError('pilot comparison source changed')
    if manifest.get('rows') != pilot['rows'] or manifest.get('seeds') != [0, 1]:
        raise ValueError('replay must use exact four pilot prefixes and seeds')
    arms = manifest.get('arms')
    if arms != [{'name': 'native_A', 'policy': 'zero', 'role': 'native'},
                {'name': 'native_B', 'policy': 'zero', 'role': 'native'}]:
        raise ValueError('replay has non-native arm or changed arm order')
    if manifest.get('actions') != pilot['actions'] or manifest.get('max_tokens') != 256:
        raise ValueError('replay must retain exact pilot hook table and 256-token cap')
    source_checks = ('family_freeze_sha256', 'prefix_scout_sha256', 'prepared_sha256',
                     'qualified_worker_sha256')
    if any(manifest.get(key) != pilot[key] for key in source_checks):
        raise ValueError('frozen family/prefix/worker provenance changed')
    worker_prep = base.sealed(base.WORKER_PREP)
    worker_qual = base.sealed(base.WORKER_QUAL)
    if (not worker_qual['pass'] or worker_qual['sha256'] != manifest['qualified_worker_sha256'] or
            worker_qual['worker_code_digest'] != worker_prep['sha256']):
        raise ValueError('ordered worker qualification differs')
    if any(base.file_sha(path) != expected for path, expected in worker_prep['files'].items()):
        raise ValueError('qualified overlay changed')
    codes = manifest.get('code_files', {})
    required = (Path(__file__), Path(driver_path),
                Path(worker_prep['overlay']) / 'moe_exp/routing_control/worker_adapter.py',
                Path(worker_prep['overlay']) / 'moe_exp/routing_control/ordered_vllm.py')
    if any(codes.get(str(path)) != base.file_sha(path) for path in required):
        raise ValueError('native replay code hashes differ')
    if any(base.file_sha(path) != expected for path, expected in codes.items()):
        raise ValueError('bound source code differs')
    rows = manifest['rows']
    requests = len(rows) * len(manifest['seeds']) * len(arms)
    prefill = sum((len(row['prompt_ids']) + len(row['prefix_ids'])) * 4 for row in rows)
    decode = requests * 256
    context = max(len(row['prompt_ids']) + len(row['prefix_ids']) + 256 for row in rows)
    if any(manifest.get(field) != value for field, value in (
        ('expected_requests', requests), ('expected_prefill_tokens', prefill),
        ('maximum_decode_tokens', decode), ('maximum_context_tokens', context))):
        raise ValueError('replay stage resource count differs')
    table = ORIGINAL_BUILD_POLICY_TABLE(manifest['actions'])
    if table.policies[0].name != 'zero' or table.hooked_layers() != (28,):
        raise ValueError('pilot target hook table was not constructed')
    pilot_table = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
                       'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
                       'micro-screen-qual4-ac4c9651e71fe067/policy-table.json')
    import json
    if table.sealed() != json.loads(pilot_table.read_text()):
        raise ValueError('native replay policy table differs from mixed pilot')
    return rows, manifest['actions'], arms


def pilot_table(actions):
    table = ORIGINAL_BUILD_POLICY_TABLE(actions)
    if table.hooked_layers() != (28,):
        raise ValueError('pilot hook layer changed')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    base.validate_manifest = validate_replay
    base.build_policy_table = pilot_table
    base.run(args)


if __name__ == '__main__':
    main()
