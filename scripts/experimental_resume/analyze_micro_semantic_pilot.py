"""Join sealed pretreatment and arm-blind Qwen ratings to all pilot assignments."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from build_micro_blind_frame import digest, sealed
import rate_micro_blind_semantics_v2_2 as rating

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
RATINGS = ROOT / 'runs/routing-control-v1/micro-screen-semantic-v22-1e56570b050828e3'
BLIND = ROOT / 'runs/routing-control-v1/micro-screen-blind-qual4-ac4c9651e71fe067'
FIRST = ROOT / 'runs/routing-control-v1/micro-screen-first-stage-qual4-ac4c9651e71fe067/FIRST_STAGE.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_SEMANTIC_PILOT_v1.json'


def main():
    frame = sealed(rating.FRAME)
    mapping = sealed(BLIND / 'ARM_MAP.json')
    binding = sealed(RATINGS / 'BINDING.json')
    summary = sealed(RATINGS / 'SUMMARY.json')
    first = sealed(FIRST)
    if (binding['frame_sha256'] != frame['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            summary['ratings'] != 104 or summary['rows'] != 48 or
            summary['valid_stopped_ratings'] > 104 or
            mapping['manifest_sha256'] != frame['generation_manifest_sha256'] or
            first['manifest_sha256'] != mapping['manifest_sha256']):
        raise ValueError('rating/generation/first-stage population differs')
    batches = []
    for reader in (0, 1):
        batches.append(sealed(RATINGS / 'batches' / f'start-reader{reader}.json'))
    for start in range(0, 48, rating.BATCH):
        for reader in (0, 1):
            batches.append(sealed(RATINGS / 'batches' / f'target-{start:03d}-reader{reader}.json'))
    if [part['sha256'] for part in batches] != summary['batch_sha256']:
        raise ValueError('rating summary batch seal list differs')
    grouped = defaultdict(dict)
    for part in batches:
        for record in part['records']:
            key = (part['task'], record['rating_id'])
            if part['reader'] in grouped[key]:
                raise ValueError('duplicate reader rating')
            grouped[key][part['reader']] = record
    start_ids = {key for key, _ in rating.start_groups(frame['records'])}
    target_ids = {row['blind_id'] for row in frame['records']}
    if ({key for task, key in grouped if task == 'start'} != start_ids or
            {key for task, key in grouped if task == 'target'} != target_ids or
            any(set(readers) != {0, 1} for readers in grouped.values())):
        raise ValueError('reader coverage differs')
    frame_by_id = {row['blind_id']: row for row in frame['records']}
    first_by_assignment = {(x['prefix_uid'], x['seed'], x['arm']): x
                           for x in first['records']}
    records = []
    for assignment in mapping['records']:
        blind_id = assignment['blind_id']
        if blind_id not in frame_by_id:
            raise ValueError('ungradeable assigned row requires explicit ITT status handling')
        source = frame_by_id[blind_id]['reader_input']
        start_key = digest([source['problem'], source['full_emitted_prefix'],
                            source['triggering_sentence']])[:32]
        start_readers = grouped[('start', start_key)]
        target_readers = grouped[('target', blind_id)]
        def status(readers, task):
            values = []
            for index in (0, 1):
                record = readers[index]
                valid = record['finish_reason'] == 'stop' and record['rating'] is not None
                values.append(record['rating'][task] if valid else None)
            return {'readers': values, 'both_valid': all(value is not None for value in values),
                    'both_positive': all(value is True for value in values),
                    'agreement': values[0] == values[1] if None not in values else None}
        start = status(start_readers, 'start')
        target = status(target_readers, 'target')
        key = (assignment['prefix_uid'], assignment['seed'], assignment['arm'])
        if key not in first_by_assignment:
            raise ValueError('first-stage assigned case missing')
        records.append({'blind_id': blind_id,
                        'prefix_uid': assignment['prefix_uid'],
                        'family': assignment['family'], 'seed': assignment['seed'],
                        'arm': assignment['arm'], 'role': assignment['role'],
                        'measurement_status': assignment['measurement_status'],
                        'start': start, 'target': target,
                        'trajectory_complete_both_readers': start['both_positive'] and
                        target['both_positive'],
                        'first_stage_target_fraction':
                        first_by_assignment[key]['target_selected_fraction'],
                        'first_stage_target_hits':
                        first_by_assignment[key]['target_selected_tokens'],
                        'reasoning_tokens': first_by_assignment[key]['reasoning_tokens']})
    if len(records) != 48:
        raise ValueError('not all 48 assigned requests retained')
    by_arm = {}
    for arm in sorted({x['arm'] for x in records}):
        subset = [x for x in records if x['arm'] == arm]
        by_arm[arm] = {'assigned': len(subset),
                       'start_both_positive': sum(x['start']['both_positive'] for x in subset),
                       'target_both_positive': sum(x['target']['both_positive'] for x in subset),
                       'trajectory_complete_both_positive': sum(
                           x['trajectory_complete_both_readers'] for x in subset),
                       'target_both_valid': sum(x['target']['both_valid'] for x in subset),
                       'target_reader_disagreements': sum(
                           x['target']['agreement'] is False for x in subset)}
    pairs = defaultdict(dict)
    for record in records:
        pairs[(record['prefix_uid'], record['seed'])][record['arm']] = record
    deltas = []
    for (prefix_uid, seed), arms in sorted(pairs.items()):
        if len(arms) != 6:
            raise ValueError('paired six-arm assignment differs')
        outcome = lambda name: int(arms[name]['trajectory_complete_both_readers'])
        deltas.append({'prefix_uid': prefix_uid, 'seed': seed,
                       'target_plus1_minus_native': outcome('target_bias1') - outcome('native'),
                       'target_plus1_minus_random_plus1': outcome('target_bias1') - outcome('random_bias1'),
                       'native_duplicate_minus_native': outcome('native_duplicate') - outcome('native')})
    body = {'schema': 'routing-micro-screen-semantic-pilot-v1',
            'frame_sha256': frame['sha256'], 'arm_map_sha256': mapping['sha256'],
            'rating_summary_sha256': summary['sha256'],
            'first_stage_sha256': first['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'start_unique_prefixes': 4,
            'start_both_positive_unique_prefixes': sum(
                all(grouped[('start', key)][index]['finish_reason'] == 'stop' and
                    grouped[('start', key)][index]['rating'] == {'start': True}
                    for index in (0, 1)) for key in start_ids),
            'valid_stopped_ratings': summary['valid_stopped_ratings'],
            'assigned': 48, 'records': records, 'by_arm': by_arm,
            'paired_deltas': deltas,
            'interpretation': 'discovery-only four-family exploratory pilot; two Qwen ratings are LLM audits, and source eligibility was rated without assigned continuations. No confirmatory semantic effect claim.'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing pilot semantic analysis differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'start_both_positive_unique_prefixes': body['start_both_positive_unique_prefixes'],
                      'by_arm': by_arm, 'paired_deltas': deltas}))


if __name__ == '__main__':
    main()
