"""Join sealed arm-blind Qwen votes to every 1024-token mechanism assignment.

Primary ITT is both valid readers positive, including every assigned failure,
closure and cap. Uncertainty resamples whole parent families jointly across
the frozen pooled and transition-specific contrasts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

import run_boundary_micro_screen as base
import build_mechanism_blind_frame_v1 as builder
import rate_mechanism_semantics_v1 as rating

PAIRS = [('target_bias1', 'native'), ('target_bias1', 'random_bias1'),
         ('native_duplicate', 'native'), ('target_bias1', 'native_average')]
SCOPES = ('all', 'candidate_to_verify', 'approach_to_commit')
BOOTSTRAPS = 5000
BOOT_SEED = 20261003


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reader_votes(frame, price, ratings_out):
    """Verify every committed reader and physical attempt before joining arms."""
    rating.validate(frame, price)
    stage = base.sealed(ratings_out / 'STAGE_SUMMARY.json')
    require(stage['schema'] == 'mechanism-1024-reader-stage-summary-v1' and
            stage['status'] == 'COMPLETE_ARM_BLIND_LLM_AUDIT' and
            stage['frame_sha256'] == frame['sha256'] and
            stage['price_sha256'] == price['sha256'] and
            stage['ratings'] == 2 * len(frame['records']) and
            len(stage['shard_summary_sha256s']) == len(price['shards']),
            'two-reader stage is unsealed or incomplete')
    by_id = defaultdict(dict)
    observed_summaries = []
    for index, shard in enumerate(price['shards']):
        out = ratings_out / f'shard-{index:03d}'
        binding = base.sealed(out / 'BINDING.json')
        summary = base.sealed(out / 'SUMMARY.json')
        require(binding['schema'] == 'mechanism-1024-reader-binding-v1' and
                binding['frame_sha256'] == frame['sha256'] and
                binding['price_sha256'] == price['sha256'] and
                binding['driver_sha256'] == base.file_sha(rating.__file__) and
                binding['shard_index'] == index and
                summary['binding_sha256'] == binding['sha256'] and
                summary['sha256'] == stage['shard_summary_sha256s'][index] and
                summary['rows'] == shard['end'] - shard['start'] and
                summary['ratings'] == 2 * summary['rows'],
                'reader shard seal or binding differs')
        observed_summaries.append(summary['sha256'])
        hashes = []
        for start, block in rating.shard_batches(frame, price, index):
            for reader in (0, 1):
                batch = rating.committed(out, start, reader, block, binding['sha256'])
                require(batch is not None, 'missing committed reader batch')
                hashes.append(batch['sha256'])
                for record in batch['records']:
                    blind_id = record['blind_id']
                    require(reader not in by_id[blind_id], 'duplicate reader vote')
                    by_id[blind_id][reader] = record
        require(hashes == summary['batch_sha256s'],
                'reader shard summary omits committed batch')
    require(observed_summaries == stage['shard_summary_sha256s'] and
            set(by_id) == {r['blind_id'] for r in frame['records']} and
            all(set(votes) == {0, 1} for votes in by_id.values()),
            'reader stage lacks exact two-vote coverage')
    return stage, by_id


def join(manifest, arm_map, frame, votes):
    require(arm_map['schema'] == 'mechanism-1024-arm-map-v1' and
            frame['schema'] == 'mechanism-1024-blind-frame-v1' and
            arm_map['manifest_sha256'] == manifest['sha256'] and
            frame['generation_manifest_sha256'] == manifest['sha256'] and
            arm_map['stage_completion_sha256'] == frame['stage_completion_sha256'] and
            arm_map['rubric_sha256'] == frame['rubric_sha256'] ==
            base.file_sha(rating.RUBRIC) and
            arm_map['builder_sha256'] == frame['builder_sha256'] ==
            base.file_sha(builder.__file__), 'blind frame or arm map differs')
    plan = builder.expected_assignments(manifest)
    map_rows = arm_map['records']
    require(len(map_rows) == len(plan) == frame['assigned'] and
            [r['uid'] for r in map_rows] == [r['uid'] for r in plan],
            'arm map lacks ordered assigned population')
    frame_ids = {r['blind_id'] for r in frame['records']}
    require(len(frame_ids) == len(frame['records']) and
            frame_ids == set(votes), 'blind frame or reader IDs differ')
    records = []
    for expected, row in zip(plan, map_rows, strict=True):
        require(all(row.get(key) == expected[key] for key in expected) and
                (row['measurement_status'] == 'gradeable') ==
                (row['blind_id'] in frame_ids),
                'arm map identity or gradeable status differs')
        assigned_votes = votes.get(row['blind_id'], {})
        reader_rows = []
        for reader in (0, 1):
            result = assigned_votes.get(reader)
            valid = result is not None and result['finish_reason'] == 'stop' and \
                isinstance(result['rating'], dict) and \
                type(result['rating'].get('target')) is bool
            reader_rows.append({'reader': reader,
                                'valid_stop': valid,
                                'target': result['rating']['target'] if valid else None,
                                'finish_reason': result['finish_reason'] if result else None,
                                'generated_tokens': result['generated_tokens'] if result else None})
        valid_pair = all(r['valid_stop'] for r in reader_rows)
        positive = valid_pair and all(r['target'] is True for r in reader_rows)
        if row['measurement_status'] in ('generation_error', 'missing_routed_array'):
            status = 'failure'
        elif row['measurement_status'] == 'zero_reasoning_tokens':
            status = 'nonfire'
        elif not valid_pair:
            status = 'unscored'
        elif row['hit_1024_cap']:
            status = 'cap'
        elif row['closed_reasoning']:
            status = 'early_finish'
        else:
            status = 'complete'
        records.append({key: row[key] for key in
                        ('uid', 'blind_id', 'prefix_uid', 'family', 'transition',
                         'question', 'seed', 'arm', 'execution_position',
                         'measurement_status', 'emitted_tokens', 'reasoning_tokens',
                         'closed_reasoning', 'hit_1024_cap', 'finish', 'stop_reason',
                         'error', 'routed_present')} |
                       {'reader_votes': reader_rows, 'reader_pair_valid': valid_pair,
                        'reader_agreement': (reader_rows[0]['target'] == reader_rows[1]['target'])
                        if valid_pair else None,
                        'both_positive': positive,
                        'at_least_one_positive': any(r['target'] is True for r in reader_rows),
                        'operational_status': status})
    return records


def _cell_values(records, endpoint):
    by_cell = defaultdict(dict)
    for row in records:
        key = row['prefix_uid'], row['seed']
        require(row['arm'] not in by_cell[key], 'duplicate arm within assigned start/seed')
        by_cell[key][row['arm']] = row
    starts = defaultdict(list)
    for (prefix_uid, seed), arms in by_cell.items():
        require(set(arms) == {'native', 'native_duplicate', 'target_bias1',
                              'random_bias1'}, 'incomplete four-arm assigned cell')
        family = arms['native']['family']
        transition = arms['native']['transition']
        require(all(r['family'] == family and r['transition'] == transition and
                    r['seed'] == seed and r['prefix_uid'] == prefix_uid
                    for r in arms.values()), 'four-arm cell identity differs')
        values = {arm: float(row[endpoint]) for arm, row in arms.items()}
        values['native_average'] = (values['native'] + values['native_duplicate']) / 2
        starts[prefix_uid].append((family, transition, seed, values, arms))
    output = []
    for prefix_uid, cells in starts.items():
        require(len(cells) == 2 and {c[2] for c in cells} == {0, 1},
                'accepted start lacks both seeds')
        family, transition = cells[0][:2]
        require(all(c[:2] == (family, transition) for c in cells),
                'start changes family or transition across seeds')
        means = {name: sum(c[3][name] for c in cells) / 2 for name in
                 ('native', 'native_duplicate', 'target_bias1', 'random_bias1',
                  'native_average')}
        output.append({'prefix_uid': prefix_uid, 'family': family,
                       'transition': transition, 'means': means})
    return output


def _quantile(sorted_values, p):
    position = p * (len(sorted_values) - 1)
    low = int(position)
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[min(low + 1, len(sorted_values) - 1)] * fraction


def clustered_contrasts(records, endpoint='both_positive', n_boot=BOOTSTRAPS,
                         seed=BOOT_SEED):
    require(n_boot >= 100, 'insufficient family bootstrap repetitions')
    starts = _cell_values(records, endpoint)
    rng = random.Random(seed)
    results = []
    multiplicity = len(PAIRS) * len(SCOPES)
    for scope in SCOPES:
        subset = [row for row in starts if scope == 'all' or row['transition'] == scope]
        families = sorted({r['family'] for r in subset})
        require(len(families) >= 2, 'fewer than two independent families in analysis scope')
        members = {f: [r for r in subset if r['family'] == f] for f in families}
        draws = [[families[rng.randrange(len(families))]
                  for _ in families] for _ in range(n_boot)]
        for left, right in PAIRS:
            diffs = {f: [r['means'][left] - r['means'][right] for r in members[f]]
                     for f in families}
            estimate = sum(sum(values) for values in diffs.values()) / len(subset)
            bootstrap = []
            for sampled in draws:
                total = sum(sum(diffs[f]) for f in sampled)
                denominator = sum(len(diffs[f]) for f in sampled)
                bootstrap.append(total / denominator)
            bootstrap.sort()
            alpha = .05 / multiplicity
            results.append({'scope': scope, 'arm': left, 'reference': right,
                            'estimate': estimate,
                            'simultaneous_ci95': [_quantile(bootstrap, alpha / 2),
                                                   _quantile(bootstrap, 1 - alpha / 2)],
                            'accepted_starts': len(subset), 'families': len(families)})
    return {'endpoint': endpoint, 'contrasts': results,
            'resampling': 'paired accepted starts, whole parent-family clusters; starts weighted equally',
            'simultaneous_method': 'Bonferroni percentile bootstrap approximation',
            'simultaneous_scope': 'twelve contrasts within this endpoint only',
            'multiplicity': multiplicity, 'replicates': n_boot, 'seed': seed,
            'registered_128_family_claim': False}


def arm_summary(records):
    return {arm: {'assigned': len(rows),
                  'gradeable': sum(r['measurement_status'] == 'gradeable' for r in rows),
                  'reader_pair_valid': sum(r['reader_pair_valid'] for r in rows),
                  'both_positive': sum(r['both_positive'] for r in rows),
                  'reader_disagreement': sum(r['reader_agreement'] is False for r in rows),
                  'reasoning_closure': sum(r['closed_reasoning'] for r in rows),
                  'cap_1024': sum(r['hit_1024_cap'] for r in rows),
                  'emitted_tokens': sum(r['emitted_tokens'] for r in rows),
                  'statuses': dict(Counter(r['operational_status'] for r in rows))}
            for arm, rows in ((arm, [r for r in records if r['arm'] == arm])
                              for arm in ('native', 'native_duplicate',
                                          'target_bias1', 'random_bias1'))}


def position_sensitivity(records):
    groups = defaultdict(list)
    by_cell = defaultdict(dict)
    for row in records:
        groups[row['arm'], row['execution_position']].append(row)
        by_cell[row['prefix_uid'], row['seed']][row['arm']] = row
    rates = {f'{arm}|{position}': {'assigned': len(rows),
                                  'both_positive': sum(r['both_positive'] for r in rows),
                                  'rate': sum(r['both_positive'] for r in rows) / len(rows)}
             for (arm, position), rows in sorted(groups.items())}
    paired = {}
    for position in range(4):
        cells = [arms for arms in by_cell.values()
                 if arms['target_bias1']['execution_position'] == position]
        paired[str(position)] = {'cells': len(cells),
                                 'target_minus_native': sum(
                                     int(a['target_bias1']['both_positive']) -
                                     int(a['native']['both_positive']) for a in cells) /
                                     len(cells) if cells else None}
    return {'arm_by_execution_position': rates,
            'target_minus_native_by_target_position': paired,
            'interpretation': 'Descriptive position sensitivity; no independent position allocation claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm-map', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--price', type=Path, required=True)
    parser.add_argument('--ratings-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest, arm_map, frame, price = map(base.sealed,
        (args.manifest, args.arm_map, args.frame, args.price))
    stage, votes = reader_votes(frame, price, args.ratings_out)
    records = join(manifest, arm_map, frame, votes)
    require(len(records) == manifest['expected_requests'] and
            len({r['uid'] for r in records}) == len(records),
            'ITT lost assigned outcomes')
    body = {'schema': 'mechanism-1024-semantic-itt-v1',
            'manifest_sha256': manifest['sha256'],
            'arm_map_sha256': arm_map['sha256'],
            'blind_frame_sha256': frame['sha256'],
            'reader_price_sha256': price['sha256'],
            'reader_stage_summary_sha256': stage['sha256'],
            'analysis_driver_sha256': base.file_sha(__file__),
            'rubric_sha256': base.file_sha(rating.RUBRIC),
            'stage_label': manifest['stage_label'],
            'registered_128_family_feasibility':
                manifest['registered_128_family_feasibility'],
            'assigned': len(records),
            'accepted_families': len({r['family'] for r in records}),
            'accepted_starts': len({r['prefix_uid'] for r in records}),
            'arm_summary': arm_summary(records),
            'primary_itt': clustered_contrasts(records),
            'at_least_one_reader_sensitivity': clustered_contrasts(
                records, endpoint='at_least_one_positive', seed=BOOT_SEED + 1),
            'position_sensitivity': position_sensitivity(records),
            'records': records,
            'claim_limit': manifest['claim_limit'] +
                ' Semantic ratings are Qwen3.8 audits, not human truth; '
                'bootstrap intervals are approximate and do not establish '
                'a second transition or action-order effect.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    value = rating.save(args.out, body, existing_ok=True)
    print(json.dumps({'status': 'COMPLETE_ALL_ASSIGNED_ITT',
                      'analysis_sha256': value['sha256'],
                      'assigned': value['assigned'],
                      'families': value['accepted_families'],
                      'stage_label': value['stage_label']}), flush=True)


if __name__ == '__main__':
    main()
