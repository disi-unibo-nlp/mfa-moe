"""Immediate first-token route comparison before continuation paths diverge."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from build_micro_blind_frame import digest, file_sha, sealed

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
RUN = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_TOKEN_ZERO_AUDIT_v1.json'
TARGET = (9, 189)


def main():
    manifest = sealed(MANIFEST)
    binding = sealed(RUN / 'BINDING.json')
    batch = sealed(RUN / 'batch-000.json')
    npz = RUN / 'batch-000.npz'
    if (binding['manifest_sha256'] != manifest['sha256'] or
            batch['manifest_sha256'] != manifest['sha256'] or
            batch['binding_sha256'] != binding['sha256'] or
            file_sha(npz) != batch['array_sha256'] or len(batch['outputs']) != 48):
        raise ValueError('frozen pilot receipts differ')
    records = []
    with np.load(npz, allow_pickle=False) as arrays:
        for output in batch['outputs']:
            routed = arrays[output['uid']]
            if (output['error'] or not output['tokens'] or
                    routed.shape != (len(output['tokens']), 40, 8)):
                raise ValueError('missing routed first-token data')
            experts = sorted(int(x) for x in routed[0, 28])
            records.append({'prefix_uid': output['prefix_uid'], 'seed': output['seed'],
                            'arm': output['arm'], 'policy': output['policy'],
                            'prompt_sha256': output['prompt_sha256'],
                            'first_emitted_token': output['tokens'][0],
                            'layer28_first_token_unordered_topk': experts,
                            'target_selected': any(x in TARGET for x in experts),
                            'target_ids_selected': [x for x in TARGET if x in experts],
                            'action_dose_both_ranks_equal':
                            output['action_dose'].get('0') == output['action_dose'].get('1'),
                            'action_dose': output['action_dose']})
    groups = defaultdict(dict)
    for record in records:
        groups[(record['prefix_uid'], record['seed'])][record['arm']] = record
    if len(groups) != 8 or any(len(arms) != 6 for arms in groups.values()):
        raise ValueError('pilot pair and arm count differs')
    comparisons = []
    for (prefix_uid, seed), arms in sorted(groups.items()):
        base = arms['native']
        duplicate = arms['native_duplicate']
        if len({r['prompt_sha256'] for r in arms.values()}) != 1:
            raise ValueError('arms have different same-prefix prompts')
        comparisons.append({'prefix_uid': prefix_uid, 'seed': seed,
                            'native_first_token_equal':
                            base['first_emitted_token'] == duplicate['first_emitted_token'],
                            'native_topk_equal':
                            base['layer28_first_token_unordered_topk'] ==
                            duplicate['layer28_first_token_unordered_topk'],
                            'native_target_mask_equal':
                            base['target_selected'] == duplicate['target_selected'],
                            'first_token_target_selected_by_arm':
                            {arm: value['target_selected'] for arm, value in arms.items()},
                            'first_token_topk_by_arm':
                            {arm: value['layer28_first_token_unordered_topk']
                             for arm, value in arms.items()},
                            'target_plus1_vs_native':
                            int(arms['target_bias1']['target_selected']) -
                            int(base['target_selected']),
                            'target_plus1_vs_random_plus1':
                            int(arms['target_bias1']['target_selected']) -
                            int(arms['random_bias1']['target_selected'])})
    by_arm = {arm['name']: {'selected_count': sum(
        x['first_token_target_selected_by_arm'][arm['name']] for x in comparisons),
        'pairs': len(comparisons)} for arm in manifest['arms']}
    body = {'schema': 'routing-micro-screen-first-token-audit-v1',
            'manifest_sha256': manifest['sha256'],
            'batch_receipt_sha256': batch['sha256'],
            'routed_array_sha256': batch['array_sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'records': records, 'pairs': comparisons,
            'first_token_target_selected_by_arm': by_arm,
            'native_first_token_equal_pairs': sum(x['native_first_token_equal'] for x in comparisons),
            'native_topk_equal_pairs': sum(x['native_topk_equal'] for x in comparisons),
            'native_target_mask_equal_pairs': sum(x['native_target_mask_equal'] for x in comparisons),
            'plus1_target_vs_native_positive_pairs': sum(x['target_plus1_vs_native'] > 0 for x in comparisons),
            'plus1_target_vs_random_plus1_positive_pairs': sum(
                x['target_plus1_vs_random_plus1'] > 0 for x in comparisons),
            'interpretation': 'immediate same-prefix gate selection before emitted-token divergence; native variability and semantic effect remain separate'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing token-zero audit differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'selected': by_arm,
                      'native_first_token_equal_pairs': body['native_first_token_equal_pairs'],
                      'native_topk_equal_pairs': body['native_topk_equal_pairs'],
                      'native_target_mask_equal_pairs': body['native_target_mask_equal_pairs'],
                      'plus1_target_vs_native_positive_pairs': body['plus1_target_vs_native_positive_pairs'],
                      'plus1_target_vs_random_plus1_positive_pairs':
                      body['plus1_target_vs_random_plus1_positive_pairs']}))


if __name__ == '__main__':
    main()
