"""Join eligible blind ratings to sealed arm assignments and run the predeclared ITT analysis.

The input is the sealed ARM_MAP.json (assignment identities, doses and operational
status), the sealed BLIND_FRAME.json and the arm-blind rating output directory.
The primary endpoint is the conservative both-reader-positive immediate semantic
transition; the at-least-one-reader endpoint is a sensitivity analysis.  Every
assigned request remains in the intention-to-treat denominator, including
generation failures, missing routed arrays, zero-token nonfires and unparsed
reader attempts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket

from moe_exp.routing_control.analysis import paired_itt


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BATCH = 12
ALLOWLIST = frozenset({'transition', 'problem', 'full_emitted_prefix',
                       'triggering_sentence', 'continuation'})
KINDS = {
    'routing-eligible-micro-serial-v3': {
        'kind': 'positive',
        'pairs': (('target_bias0.5', 'native'), ('target_bias0.5', 'random_bias0.5'),
                  ('target_bias1', 'native'), ('target_bias1', 'random_bias1')),
    },
    'routing-eligible-deactivation-serial-v4': {
        'kind': 'deactivation',
        'pairs': (('target_bias_minus1', 'native'), ('target_bias_minus1', 'random_bias_minus1'),
                  ('target_force_off', 'native'), ('target_force_off', 'random_force_off')),
    },
}
NOISE_PAIR = (('native_duplicate', 'native'),)
METRICS = ('semantic_success', 'tokens')
OBSERVED_STATUSES = frozenset({'complete', 'nonfire', 'early_finish', 'cap',
                              'failure', 'unscored'})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({key: item for key, item in value.items()
                                      if key != 'sha256'}):
        raise ValueError('invalid sealed JSON: ' + str(path))
    return value


def _atomic_json(path, value):
    path = Path(path)
    suffix = os.environ.get('SLURM_JOB_ID', 'local')
    part = path.with_name(path.name + '.part-' + suffix)
    part.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(part, path)


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    path = Path(path)
    if path.exists():
        if sealed(path) != value:
            raise ValueError('bound analysis output differs: ' + str(path))
    else:
        _atomic_json(path, value)
    return value


def execution_positions(manifest):
    """Recover every (family, seed, arm) run position from the sealed schedule."""
    arms = [arm['name'] for arm in manifest['arms']]
    if len(arms) != 6 or len(set(arms)) != 6:
        raise ValueError('six-arm schedule required')
    schedules = manifest.get('arm_order_by_family_seed')
    if not isinstance(schedules, dict) or len(schedules) != len(manifest['rows']):
        raise ValueError('sealed arm-order schedule is incomplete')
    seeds = manifest['seeds']
    table = {}
    for row in manifest['rows']:
        family = row['family']
        order_by_seed = schedules.get(family)
        if not isinstance(order_by_seed, list) or len(order_by_seed) != len(seeds):
            raise ValueError('sealed arm-order schedule has the wrong seed count')
        for seed_index, order in enumerate(order_by_seed):
            if len(order) != len(arms) or sorted(order) != sorted(arms):
                raise ValueError('sealed arm-order schedule is not a permutation')
            for position, arm in enumerate(order):
                table[(family, seeds[seed_index], arm)] = position
    return table


def primary_pairs(manifest):
    if manifest['schema'] not in KINDS:
        raise ValueError('unsupported eligible manifest schema')
    pairs = KINDS[manifest['schema']]['pairs']
    arm_names = {arm['name'] for arm in manifest['arms']}
    if any(left not in arm_names or right not in arm_names for pair in pairs for left, right in (pair,)):
        raise ValueError('predeclared contrast arm is missing from the manifest')
    return pairs


def validate_inputs(manifest, arm_map, frame, binding, summary, batches):
    if manifest['schema'] not in KINDS:
        raise ValueError('unsupported eligible manifest schema')
    if arm_map.get('schema') != 'eligible-immediate-arm-map-v1':
        raise ValueError('unexpected arm-map schema')
    if frame.get('schema') != 'eligible-immediate-blind-frame-v1':
        raise ValueError('unexpected blind-frame schema')
    if binding.get('schema') != 'eligible-immediate-blind-rating-binding-v1':
        raise ValueError('unexpected rating-binding schema')
    if summary.get('schema') != 'eligible-immediate-rating-summary-v1':
        raise ValueError('unexpected rating-summary schema')
    if (arm_map.get('manifest_sha256') != manifest['sha256'] or
            frame.get('generation_manifest_sha256') != manifest['sha256'] or
            arm_map.get('generation_summary_sha256') != frame.get('generation_summary_sha256')):
        raise ValueError('generation bindings differ')
    if (binding.get('frame_sha256') != frame['sha256'] or
            summary.get('binding_sha256') != binding['sha256'] or
            summary.get('frame_sha256') != frame['sha256']):
        raise ValueError('rating bindings differ')
    if set(frame.get('reader_input_allowlist', [])) != ALLOWLIST:
        raise ValueError('blind frame allowlist differs')
    if frame.get('continuation_max_tokens') != 256:
        raise ValueError('blind frame continuation cap differs')
    frame_rows = frame.get('records', [])
    if not 1 <= len(frame_rows) <= manifest['expected_requests']:
        raise ValueError('blind frame has an invalid number of gradeable rows')
    if any(set(row) != {'blind_id', 'reader_input'} or
           set(row['reader_input']) != ALLOWLIST for row in frame_rows):
        raise ValueError('blind frame row exceeds the reader allowlist')
    map_rows = arm_map.get('records', [])
    if len(map_rows) != manifest['expected_requests']:
        raise ValueError('arm map does not retain every assigned request')
    if len({row['uid'] for row in map_rows}) != len(map_rows):
        raise ValueError('arm map has duplicate assigned UIDs')
    if len({row['blind_id'] for row in map_rows}) != len(map_rows):
        raise ValueError('arm map has duplicate blind IDs')
    if summary.get('rows') != len(frame_rows) or summary.get('ratings') != 2 * len(frame_rows):
        raise ValueError('rating summary count differs from the blind frame')
    expected_batches = [(start, reader) for start in range(0, len(frame_rows), BATCH)
                        for reader in (0, 1)]
    if len(batches) != len(expected_batches) or len(summary.get('batch_sha256', [])) != len(batches):
        raise ValueError('rating batch count differs')
    return frame_rows, map_rows, expected_batches


def join_records(manifest, arm_map, frame, binding, summary, batches):
    frame_rows, map_rows, expected_batches = validate_inputs(
        manifest, arm_map, frame, binding, summary, batches)
    positions = execution_positions(manifest)
    frame_ids = {row['blind_id'] for row in frame_rows}
    rating_by_id = {}
    for batch, (start, reader) in zip(batches, expected_batches, strict=True):
        if (batch.get('schema') != 'eligible-immediate-rating-batch-v1' or
                batch.get('start') != start or batch.get('reader') != reader):
            raise ValueError('rating batch identity differs')
        expected_ids = [row['blind_id'] for row in frame_rows[start:start + BATCH]]
        records = batch.get('records', [])
        if [row['blind_id'] for row in records] != expected_ids:
            raise ValueError('rating batch does not match the frozen blind block')
        for record in records:
            if record['blind_id'] in rating_by_id and reader in rating_by_id[record['blind_id']]:
                raise ValueError('duplicate reader rating')
            rating_by_id.setdefault(record['blind_id'], {})[reader] = record
    if any(set(readers) != {0, 1} for readers in rating_by_id.values()):
        raise ValueError('both readers must rate every gradeable row')
    if set(rating_by_id) != frame_ids:
        raise ValueError('rating coverage differs from the blind frame')
    joined = []
    for row in map_rows:
        required = {'uid', 'prefix_uid', 'family', 'transition', 'canonical_question',
                    'seed', 'arm', 'role', 'blind_id', 'measurement_status',
                    'emitted_tokens'}
        if not required <= set(row):
            raise ValueError('arm-map row is missing a required field')
        key = (row['family'], row['seed'], row['arm'])
        if key not in positions:
            raise ValueError('assigned arm is absent from the sealed schedule')
        gradeable = row['measurement_status'] == 'gradeable'
        if gradeable != (row['blind_id'] in frame_ids):
            raise ValueError('gradeable status and blind frame disagree')
        readers = rating_by_id.get(row['blind_id'], {}) if gradeable else {}
        values = {}
        for reader in (0, 1):
            record = readers.get(reader)
            valid = (record is not None and record.get('rating') is not None and
                     record.get('finish_reason') == 'stop')
            values[reader] = record.get('rating') if valid else None
            values[str(reader) + '_valid'] = valid
        both_valid = all(values[reader] is not None for reader in (0, 1))
        both_positive = both_valid and values[0] is True and values[1] is True
        at_least_one = any(values[reader] is True for reader in (0, 1) if values[reader] is not None)
        agreement = (values[0] == values[1]) if both_valid else None
        if row['measurement_status'] != 'gradeable':
            status = 'nonfire' if row['measurement_status'] == 'zero_emitted_tokens' else 'failure'
        elif both_valid:
            status = 'complete'
        else:
            status = 'unscored'
        if status not in OBSERVED_STATUSES:
            raise ValueError('unknown operational status')
        joined.append({
            **row,
            'execution_position': positions[key],
            'reader_ratings': values,
            'both_readers_valid': both_valid,
            'both_readers_positive': both_positive,
            'at_least_one_reader_positive': at_least_one,
            'reader_agreement': agreement,
            'operational_status': status,
            'tokens': int(row.get('emitted_tokens') or 0),
        })
    if len(joined) != len(map_rows):
        raise ValueError('assigned rows were lost during the join')
    return joined


def _paired(records, pairs, endpoint_key, n_boot, seed):
    assigned, observed = [], []
    for record in records:
        assigned.append({'uid': record['uid'], 'question': record['canonical_question'],
                         'family': record['family'], 'arm': record['arm'], 'seed': record['seed']})
        observed.append({
            'uid': record['uid'], 'question': record['canonical_question'],
            'family': record['family'], 'arm': record['arm'], 'seed': record['seed'],
            'status': record['operational_status'], 'correct': 0,
            'semantic_success': int(record[endpoint_key]), 'tokens': record['tokens']})
    return paired_itt(assigned, observed, reference_arm='native', n_boot=n_boot, seed=seed,
                      comparison_pairs=list(pairs), metrics=METRICS)


def _arm_summary(records):
    result = {}
    for arm in sorted({record['arm'] for record in records}):
        subset = [record for record in records if record['arm'] == arm]
        result[arm] = {
            'assigned': len(subset),
            'gradeable': sum(record['measurement_status'] == 'gradeable' for record in subset),
            'both_readers_valid': sum(record['both_readers_valid'] for record in subset),
            'both_readers_positive': sum(record['both_readers_positive'] for record in subset),
            'at_least_one_reader_positive': sum(
                record['at_least_one_reader_positive'] for record in subset),
            'reader_disagreements': sum(record['reader_agreement'] is False for record in subset),
            'emitted_tokens_sum': sum(record['tokens'] for record in subset),
            'statuses': dict(Counter(record['operational_status'] for record in subset)),
        }
    return result


def _dose_summary(records, kind):
    result = {}
    for arm in sorted({record['arm'] for record in records}):
        subset = [record for record in records if record['arm'] == arm]
        if kind == 'positive':
            present = [record for record in subset if record.get('action_dose')]
            # The receipt mixes token counts, expert hits, mass and L1 distance.
            # A scalar mean across those units has no scientific interpretation.
            # Typed first-stage summaries are sealed in the separate routing audit.
            result[arm] = {'records_with_action_dose': len(present)}
        else:
            present = [record for record in subset if record.get('base_hook_dose')]
            entries = [entry for record in present
                       for entry in record['base_hook_dose'].values()
                       if isinstance(entry, dict)]

            def _sum_field(entry, field):
                value = entry.get(field)
                return value if isinstance(value, int) and not isinstance(value, bool) else 0

            result[arm] = {'records_with_base_hook_dose': len(present),
                           'cpu_pulse_rows': sum(_sum_field(entry, 'cpu_pulse_rows')
                                                 for entry in entries),
                           'cpu_active_rows': sum(_sum_field(entry, 'cpu_active_rows')
                                                  for entry in entries),
                           'preemptions': sum(_sum_field(entry, 'preemptions')
                                              for entry in entries)}
    return result


def _position_sensitivity(records, pairs, endpoint_key):
    by_position = defaultdict(list)
    for record in records:
        by_position[record['execution_position']].append(int(record[endpoint_key]))
    position_mean = {position: sum(values) / len(values)
                     for position, values in by_position.items()}
    grouped = defaultdict(dict)
    for record in records:
        grouped[(record['canonical_question'], record['seed'])][record['arm']] = record
    result = {}
    for target, reference in pairs:
        raw, adjusted, per_position = [], [], defaultdict(list)
        for arms in grouped.values():
            if target not in arms or reference not in arms:
                raise ValueError('paired contrast is incomplete')
            left, right = arms[target], arms[reference]
            raw.append(int(left[endpoint_key]) - int(right[endpoint_key]))
            adjusted.append((int(left[endpoint_key]) - position_mean[left['execution_position']]) -
                            (int(right[endpoint_key]) - position_mean[right['execution_position']]))
            per_position[left['execution_position']].append(
                int(left[endpoint_key]) - int(right[endpoint_key]))
        result[target + '_minus_' + reference] = {
            'raw_paired_mean': sum(raw) / len(raw),
            'position_adjusted_paired_mean': sum(adjusted) / len(adjusted),
            'pairs': len(raw),
            'by_target_position': {str(position): {'n': len(values),
                                                   'mean': sum(values) / len(values)}
                                   for position, values in sorted(per_position.items())},
        }
    return result


def run_analysis(manifest, arm_map, frame, binding, summary, batches, *,
                 n_boot=5000, seed=20261002, driver_sha=''):
    joined = join_records(manifest, arm_map, frame, binding, summary, batches)
    pairs = primary_pairs(manifest)
    pooled = _paired(joined, pairs, 'both_readers_positive', n_boot, seed)
    sensitivity = _paired(joined, pairs, 'at_least_one_reader_positive', n_boot, seed)
    noise = _paired(joined, NOISE_PAIR, 'both_readers_positive', n_boot, seed)
    strata = {}
    for transition in sorted({record['transition'] for record in joined}):
        subset = [record for record in joined if record['transition'] == transition]
        strata[transition] = _paired(subset, pairs, 'both_readers_positive', n_boot, seed)
    body = {
        'schema': 'eligible-immediate-semantic-analysis-v1',
        'kind': KINDS[manifest['schema']]['kind'],
        'manifest_sha256': manifest['sha256'],
        'arm_map_sha256': arm_map['sha256'],
        'frame_sha256': frame['sha256'],
        'generation_summary_sha256': arm_map['generation_summary_sha256'],
        'rating_binding_sha256': binding['sha256'],
        'rating_summary_sha256': summary['sha256'],
        'rating_batch_sha256': [batch['sha256'] for batch in batches],
        'driver_sha256': driver_sha,
        'population': {
            'assigned': len(joined),
            'gradeable': sum(record['measurement_status'] == 'gradeable' for record in joined),
            'both_readers_valid': sum(record['both_readers_valid'] for record in joined),
            'both_readers_positive': sum(record['both_readers_positive'] for record in joined),
            'at_least_one_reader_positive': sum(
                record['at_least_one_reader_positive'] for record in joined),
            'by_transition': dict(Counter(record['transition'] for record in joined)),
        },
        'by_arm': _arm_summary(joined),
        'primary': {'endpoint': 'both_readers_positive_immediate_semantic_transition',
                    'itt_unscored_counted_as_negative': True,
                    'result': pooled},
        'sensitivity': {'endpoint': 'at_least_one_reader_positive',
                        'itt_unscored_counted_as_negative': True,
                        'result': sensitivity},
        'transition_strata': strata,
        'native_duplicate_noise': noise,
        'execution_position_sensitivity': _position_sensitivity(
            joined, pairs, 'both_readers_positive'),
        'dose': _dose_summary(joined, KINDS[manifest['schema']]['kind']),
        'interpretation': (
            'Exploratory 13-family, 256-token same-prefix discovery screen only. '
            'The two reader ratings are LLM audits, not human truth. Every assigned '
            'cell is retained in the intention-to-treat denominator. This analysis '
            'does not establish an ordered 1,024-token trajectory, answer accuracy, '
            '16k token utility or engine equivalence.'),
    }
    return body


def _load_batches(ratings, frame_rows, summary):
    batches = []
    index = 0
    for start in range(0, len(frame_rows), BATCH):
        for reader in (0, 1):
            path = Path(ratings) / 'batches' / f'target-{start:03d}-reader{reader}.json'
            batch = sealed(path)
            if batch['sha256'] != summary['batch_sha256'][index]:
                raise ValueError('rating batch seal differs from its summary')
            batches.append(batch)
            index += 1
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm-map', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261002)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('eligible semantic analysis requires CPU Slurm')
    if not args.out.resolve().is_relative_to(REPO):
        raise ValueError('analysis output must remain in the repository')
    manifest = sealed(args.manifest)
    arm_map = sealed(args.arm_map)
    frame = sealed(args.frame)
    binding = sealed(args.ratings / 'BINDING.json')
    summary = sealed(args.ratings / 'SUMMARY.json')
    if binding.get('frame_file_sha256') != file_sha(args.frame):
        raise ValueError('rating binding does not match the exact blind-frame bytes')
    batches = _load_batches(args.ratings, frame['records'], summary)
    body = run_analysis(manifest, arm_map, frame, binding, summary, batches,
                        n_boot=args.n_boot, seed=args.seed,
                        driver_sha=file_sha(__file__))
    value = write_once(args.out, body)
    print(json.dumps({'status': 'COMPLETE_ELIGIBLE_SEMANTIC_ANALYSIS',
                      'out': str(args.out), 'sha256': value['sha256'],
                      'assigned': body['population']['assigned'],
                      'both_readers_positive': body['population']['both_readers_positive'],
                      'gradeable': body['population']['gradeable']}), flush=True)


if __name__ == '__main__':
    main()
