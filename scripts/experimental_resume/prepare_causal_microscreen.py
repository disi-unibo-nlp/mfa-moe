"""Freeze a six-arm, 12-family same-prefix micro-screen proposal on discovery.

This prepares inputs and resource bounds only. The GPU runner code digest and
context-matched complete runtime price must be filled before launch.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SCOUT = R / 'steering-v1/runs/routing-control-v1/dense-discovery/CANDIDATE_PREFIX_SCOUT_v1.json'
SOURCE = R / 'steering-v1/runs/routing-control-v1/dense-verify-source-v2/RESULT.json'
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
QUAL = R / 'steering-v1/runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_PREPARED_v0.2.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if digest({k: v for k, v in value.items() if k != 'sha256'}) != value.get('sha256'):
        raise ValueError('input seal differs: ' + str(path))
    return value


def build_random_assignment(rows):
    slots = [(r['family'], seed) for r in rows for seed in (0, 1)]
    slots.sort(key=lambda item: digest(['micro-random-assign-v1', *item]))
    assigned = {item: index % 4 for index, item in enumerate(slots)}
    if Counter(assigned.values()) != Counter({i: 6 for i in range(4)}):
        raise ValueError('four random sets do not balance over 24 assignments')
    return {r['family']: [assigned[r['family'], 0], assigned[r['family'], 1]] for r in rows}


def main():
    scout, source, families = sealed(SCOUT), sealed(SOURCE), sealed(FAMILY)
    qual = json.loads(QUAL.read_text())
    if (scout['schema'] != 'candidate-prefix-scout-v1' or len(scout['records']) != 12 or
        source['schema'] != 'dense-verify-native-source-shortlist-v2' or
        source['support']['within_family_matched_positive_families'] != 48 or
        not qual.get('pass')):
        raise ValueError('micro-screen source qualification differs')
    proposed = source['proposal']
    if not proposed or not proposed['action_shape_valid'] or proposed['band_layers'] != 1:
        raise ValueError('no valid dense Verify class action source')
    experts = proposed['experts']
    if [(x['layer'], x['expert']) for x in experts] != [(28, 189), (28, 9)]:
        raise ValueError('source expert set changed; new manifest version required')
    random_pairs = [
        [experts[0]['first_four_matched_random_experts'][i],
         experts[1]['first_four_matched_random_experts'][i]] for i in range(4)]
    if random_pairs != [[139, 120], [255, 133], [43, 5], [196, 24]]:
        raise ValueError('exposure-matched random pool changed')
    if any(x['exposure_matched_random_candidates_plusminus_10_percent'] < 4 for x in experts):
        raise ValueError('insufficient random expert support')
    discovery = set(families['new_parent_pools']['parent_pools']['discovery'])
    representative = families['new_parent_pools']['representative_questions']
    rows = []
    for record in scout['records']:
        for key in ('prompt_ids', 'prefix_ids'):
            if digest(record[key]) != record[key + '_sha256'] or not record[key] or not all(type(t) is int and t >= 0 for t in record[key]):
                raise ValueError('exact native token ID replay digest changed')
        if (record['family'] not in discovery or
            record['question'] != representative[record['family']] or
            len(record['prefix_ids']) > 8192):
            raise ValueError('scout row violates frozen discovery enrollment')
        rows.append({key: record[key] for key in ('uid', 'question', 'family', 'prompt_ids',
            'prefix_ids', 'prompt_ids_sha256', 'prefix_ids_sha256', 'tokenizer_sha256',
            'trace_sha256', 'generation_messages_sha256', 'prefix_text_sha256',
            'attempt_id', 'sentence_index')})
    if len({r['family'] for r in rows}) != 12 or len({r['uid'] for r in rows}) != 12:
        raise ValueError('prefix scout lacks 12 independent discovery families')
    actions = []
    for name, ids in [('target', [189, 9]), *[(f'random{i}', pair) for i, pair in enumerate(random_pairs)]]:
        for label, bias in [('0.5', 0.5), ('1', 1.0)]:
            actions.append({'name': f'{name}_bias{label}',
                            'transition': 'candidate_to_verify',
                            'experts': [[28, ids]], 'bias': bias})
    arms = [
        {'name': 'native', 'policy': 'zero', 'role': 'native'},
        {'name': 'native_duplicate', 'policy': 'zero', 'role': 'native_duplicate_isolation'},
        {'name': 'target_bias0.5', 'policy': 'target_bias0.5', 'role': 'target'},
        {'name': 'target_bias1', 'policy': 'target_bias1', 'role': 'target'},
        {'name': 'random_bias0.5', 'policy': 'random_selector_bias0.5', 'role': 'matched_random'},
        {'name': 'random_bias1', 'policy': 'random_selector_bias1', 'role': 'matched_random'},
    ]
    assignments = build_random_assignment(rows)
    prefill = sum(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 2 * len(arms)
    decode = len(rows) * 2 * len(arms) * 256
    if prefill != 376392 or decode != 36864:
        raise ValueError('frozen complete token price changed')
    payload = {'schema': 'causal-microscreen-prepared-v2',
               'status': 'PREPARED_HOLD_GPU_RUNNER_DIGEST_AND_CONTEXT_PRICE',
               'population': '12 fixed discovery families with two-reader accepted candidate starts; not validation families',
               'family_freeze_sha256': families['sha256'],
               'prefix_scout_sha256': scout['sha256'],
               'expert_source_sha256': source['sha256'],
               'qualified_worker_receipt_sha256': qual.get('sha256'),
               'worker_code_digest': qual['worker_code_digest'],
               'base_tree_sha256': qual['base_tree'],
               'driver_sha256': None,
               'manifest_builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'rows': rows, 'actions': actions, 'arms': arms,
               'target_action_source': 'dense_verify_class_native_paired_routing',
               'random_action_source': 'native_exposure_matched_random',
               'random_set_by_family_seed': assignments,
               'random_selector_rule': 'for random arm, use random{set_for_family_and_seed}_bias{dose}; set index from map',
               'seeds': [0, 1], 'max_tokens': 256,
               'native_top_k': 8, 'shared_expert': 'unchanged',
               'pulse': {'relative_token_start': 0, 'relative_token_end_exclusive': 256,
                         'stop_at_reasoning_closure': True},
               'price': {'requests': len(rows) * len(arms) * 2,
                         'maximum_prefill_tokens_without_cache_reuse': prefill,
                         'maximum_decode_tokens': decode,
                         'longest_context_tokens': max(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows),
                         'proposed_slurm': {'GPUs': 2, 'A100_time_limit_minutes': 45,
                                            'allocation_ceiling_GPU_hours': 1.5},
                         'timing_reference': 'ordered-worker qualification job 59108983: 17 requests, 37472 prefill, 16512 decode, 1010 s allocation on 2 A100',
                         'gate': 'context-matched pricing and runner qualification required; load/prefill/decode/grading/retry/shutdown counted'},
               'science_gate': 'first-stage target expert selection and dose; same-prefix semantic ratings; native duplicate/neighbor isolation; cap, closure, failures retained',
               'interpretation': 'dense Verify class source is observational; future target rating is never passed to worker; this exploratory micro-screen does not establish accuracy/token utility'}
    value = {**payload, 'sha256': digest(payload)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing prepared manifest differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'], 'requests': payload['price']['requests'],
                      'prefill': prefill, 'decode': decode, 'random_assignments': dict(Counter(v for x in assignments.values() for v in x))}))


if __name__ == '__main__':
    main()
