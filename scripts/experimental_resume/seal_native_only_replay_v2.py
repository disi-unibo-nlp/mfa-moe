"""Freeze the 16-request native-only replay against the sealed four-family pilot."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_native_only_replay_v2 as replay

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
PILOT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_NATIVE_ONLY_REPLAY_MANIFEST_v2.json'


def main():
    pilot = base.sealed(PILOT)
    prep = base.sealed(base.WORKER_PREP)
    wrapper = Path(replay.__file__)
    driver = Path(base.__file__)
    overlay = Path(prep['overlay'])
    codes = [wrapper, driver, overlay / 'moe_exp/routing_control/design.py',
             overlay / 'moe_exp/routing_control/worker_adapter.py',
             overlay / 'moe_exp/routing_control/ordered_vllm.py']
    rows = pilot['rows']
    body = {'schema': 'routing-native-only-replay-v2',
            'base_tree_sha256': pilot['base_tree_sha256'],
            'driver_sha256': base.file_sha(driver),
            'replay_driver_sha256': base.file_sha(wrapper),
            'code_files': {str(path): base.file_sha(path) for path in codes},
            'source_pilot_sha256': pilot['sha256'],
            'family_freeze_sha256': pilot['family_freeze_sha256'],
            'prefix_scout_sha256': pilot['prefix_scout_sha256'],
            'prepared_sha256': pilot['prepared_sha256'],
            'qualified_worker_sha256': pilot['qualified_worker_sha256'],
            'rows': rows, 'actions': pilot['actions'],
            'arms': [{'name': 'native_A', 'policy': 'zero', 'role': 'native'},
                     {'name': 'native_B', 'policy': 'zero', 'role': 'native'}],
            'seeds': [0, 1], 'max_tokens': 256,
            'expected_requests': 16,
            'expected_prefill_tokens': sum((len(r['prompt_ids']) + len(r['prefix_ids'])) * 4
                                           for r in rows),
            'maximum_decode_tokens': 16 * 256,
            'maximum_context_tokens': max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256
                                          for r in rows),
            'purpose': 'diagnose native duplicate variation without edited neighbors under exact pilot hook table; all assigned requests use zero policy'}
    manifest = {**body, 'sha256': base.digest(body)}
    if OUT.exists():
        if base.sealed(OUT) != manifest:
            raise ValueError('sealed native replay manifest changed')
    else:
        OUT.write_text(json.dumps(manifest, indent=1) + '\n')
    replay.validate_replay(manifest, driver)
    print(json.dumps({'manifest': str(OUT), 'sha256': manifest['sha256'],
                      'requests': manifest['expected_requests'],
                      'prefill_tokens': manifest['expected_prefill_tokens'],
                      'maximum_decode_tokens': manifest['maximum_decode_tokens']}))


if __name__ == '__main__':
    main()
