"""Family-clustered, all-assigned semantic analysis of one frozen overnight design.

Every arm, seed and prefix stays in the analysis. Designs and horizons are never
pooled. The primary family covers every frozen contrast/scope jointly across
both-reader observed semantic success and emitted continuation-token count.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

import run_boundary_micro_screen as base
import build_overnight_blind_frame_v1 as builder
import rate_overnight_semantics_v1 as rating

require = rating.require
BOOTSTRAPS = 50000


def scopes_for(manifest):
    declared = manifest['analysis_scopes']
    transitions = {r['transition'] for r in manifest['rows']}
    require(declared[0] == 'all' and len(declared) == len(set(declared)) and
            set(declared[1:]) == transitions, 'analysis scopes differ')
    if len(transitions) == 1:
        return ['all'], {next(iter(transitions)): 'all'}
    return declared, {}


def reader_votes(frame, price, ratings_out):
    rating.validate(frame, price)
    stage = base.sealed(ratings_out / 'STAGE_SUMMARY.json')
    require(stage['schema'] == 'overnight-semantic-reader-stage-summary-v1' and
            stage['status'] == 'COMPLETE_ARM_BLIND_LLM_AUDIT' and
            stage['frame_sha256'] == frame['sha256'] and stage['price_sha256'] == price['sha256'] and
            stage['ratings'] == 2 * len(frame['records']) and
            len(stage['shard_summary_sha256s']) == len(price['shards']), 'reader stage incomplete or rebound')
    votes = defaultdict(dict)
    for index, shard in enumerate(price['shards']):
        directory = ratings_out / f'shard-{index:03d}'
        binding = base.sealed(directory / 'BINDING.json')
        summary = base.sealed(directory / 'SUMMARY.json')
        require(binding['schema'] == 'overnight-semantic-reader-binding-v1' and
                binding['frame_sha256'] == frame['sha256'] and binding['price_sha256'] == price['sha256'] and
                binding['driver_sha256'] == base.file_sha(rating.__file__) and binding['shard_index'] == index and
                summary['binding_sha256'] == binding['sha256'] and
                summary['sha256'] == stage['shard_summary_sha256s'][index] and
                summary['rows'] == shard['end'] - shard['start'] and
                summary['ratings'] == 2 * summary['rows'], 'reader shard changed')
        hashes = []
        for start, block in rating.shard_batches(frame, price, index):
            for reader in (0, 1):
                batch = rating.committed(directory, start, reader, block, binding['sha256'])
                require(batch is not None, 'missing committed reader batch')
                hashes.append(batch['sha256'])
                for record in batch['records']:
                    blind_id = record['blind_id']
                    require(reader not in votes[blind_id], 'duplicate reader vote')
                    require(record['rating'] == rating.parse_rating(record['raw_completion']),
                            'saved reader parse differs from raw response')
                    votes[blind_id][reader] = record
        require(hashes == summary['batch_sha256s'], 'summary omits committed rating batch')
    require(set(votes) == {r['blind_id'] for r in frame['records']} and
            all(set(v) == {0, 1} for v in votes.values()), 'not exactly two votes per blind continuation')
    return stage, votes


def join(manifest, arm_map, frame, votes):
    require(manifest['horizon'] in (256, 1024) and
            arm_map['schema'] == 'overnight-semantic-arm-map-v1' and
            frame['schema'] == 'overnight-semantic-blind-frame-v1' and
            arm_map['manifest_sha256'] == frame['generation_manifest_sha256'] == manifest['sha256'] and
            arm_map['stage_completion_sha256'] == frame['stage_completion_sha256'] and
            arm_map['source_frame_sha256'] == frame['source_frame_sha256'] and
            arm_map['continuation_max_tokens'] == frame['continuation_max_tokens'] == manifest['horizon'] and
            arm_map['builder_sha256'] == frame['builder_sha256'] == base.file_sha(builder.__file__) and
            arm_map['analysis_driver_sha256'] == frame['analysis_driver_sha256'] == base.file_sha(__file__) and
            arm_map['rubric_sha256'] == frame['rubric_sha256'] == base.file_sha(rating.RUBRIC),
            'frame, arm map, horizon or manifest changed')
    plan = builder.expected_assignments(manifest)
    require(len(arm_map['records']) == len(plan) == frame['assigned'], 'assigned population changed')
    frame_ids = {r['blind_id'] for r in frame['records']}
    require(len(frame_ids) == len(frame['records']) and frame_ids == set(votes), 'blind reader IDs differ')
    records = []
    for assigned, row in zip(plan, arm_map['records'], strict=True):
        require(all(row.get(k) == v for k, v in assigned.items()) and
                (row['measurement_status'] == 'gradeable') == (row['blind_id'] in frame_ids),
                'assignment identity or gradeability changed')
        rv = []
        for reader in (0, 1):
            r = votes.get(row['blind_id'], {}).get(reader)
            valid = (r is not None and r['finish_reason'] == 'stop' and
                     isinstance(r['rating'], dict) and set(r['rating']) == {'target'} and
                     type(r['rating']['target']) is bool)
            rv.append({'reader': reader, 'valid_stop': valid,
                       'target': r['rating']['target'] if valid else None,
                       'finish_reason': r['finish_reason'] if r else None,
                       'generated_tokens': r['generated_tokens'] if r else None})
        valid_pair = all(r['valid_stop'] for r in rv)
        both = valid_pair and all(r['target'] for r in rv)
        status = ('failure' if row['measurement_status'] in ('generation_error', 'missing_routed_array')
                  else 'empty_reasoning' if row['measurement_status'] == 'zero_reasoning_tokens'
                  else 'unscored' if not valid_pair else 'cap' if row['hit_cap']
                  else 'early_finish' if row['closed_reasoning'] else 'complete')
        records.append({**row, 'reader_votes': rv, 'reader_pair_valid': valid_pair,
                        'reader_agreement': rv[0]['target'] == rv[1]['target'] if valid_pair else None,
                        'both_positive': both, 'at_least_one_positive': any(r['target'] is True for r in rv),
                        'measurement_unknown': not valid_pair,
                        'operational_status': status})
    require(len({r['uid'] for r in records}) == len(records), 'duplicate assignment after join')
    return records


def cells(manifest, records, endpoints):
    names = [a['name'] for a in manifest['arms']]
    by_cell = defaultdict(dict)
    for row in records:
        key = row['prefix_uid'], row['seed']
        require(row['arm'] not in by_cell[key], 'duplicate assigned arm')
        by_cell[key][row['arm']] = row
    output = []
    for native in manifest['rows']:
        per_seed = []
        for seed in manifest['seeds']:
            block = by_cell.pop((native['uid'], seed), {})
            require(set(block) == set(names), 'missing assigned arm or seed')
            require(all(row['family'] == native['family'] and row['transition'] == native['transition']
                        for row in block.values()), 'arm changes prefix family or transition')
            per_seed.append(block)
        means = {endpoint: {arm: float(np.mean([b[arm][endpoint] for b in per_seed]))
                             for arm in names} for endpoint in endpoints}
        output.append({'prefix_uid': native['uid'], 'family': native['family'],
                       'transition': native['transition'], 'means': means})
    require(not by_cell, 'unassigned prefix or seed in analysis')
    return output


def clustered_contrasts(manifest, records, endpoints=('both_positive', 'emitted_tokens'),
                        n_boot=BOOTSTRAPS):
    require(n_boot >= 100 and manifest['analysis_seed'] == 20261004, 'invalid frozen bootstrap settings')
    pairs = manifest['planned_contrasts']
    names = {a['name'] for a in manifest['arms']}
    scopes, aliases = scopes_for(manifest)
    require(pairs and len({tuple(p) for p in pairs}) == len(pairs) and
            all(len(p) == 2 and p[0] != p[1] and set(p) <= names for p in pairs), 'invalid frozen contrasts')
    rows = cells(manifest, records, endpoints)
    multiplicity = len(pairs) * len(scopes) * len(endpoints)
    tail = .05 / (2 * multiplicity)
    results = []
    for index, scope in enumerate(scopes):
        subset = [r for r in rows if scope == 'all' or r['transition'] == scope]
        families = sorted({r['family'] for r in subset})
        require(len(families) >= 2, 'fewer than two families in frozen scope')
        grouped = {family: [r for r in subset if r['family'] == family] for family in families}
        rng = np.random.default_rng(manifest['analysis_seed'] + index)
        draws = rng.multinomial(len(families), [1 / len(families)] * len(families), size=n_boot)
        sizes = np.asarray([len(grouped[f]) for f in families], dtype=float)
        denominators = draws @ sizes
        for endpoint in endpoints:
            for left, right in pairs:
                sums = np.asarray([sum(r['means'][endpoint][left] - r['means'][endpoint][right]
                                       for r in grouped[f]) for f in families], dtype=float)
                values = (draws @ sums) / denominators
                lo, hi = map(float, np.quantile(values, [tail, 1 - tail]))
                estimate = float(sums.sum() / len(subset))
                results.append({'scope': scope, 'endpoint': endpoint, 'arm': left, 'reference': right,
                    'estimate': estimate, 'simultaneous_ci95': [lo, hi],
                    'families': len(families), 'assigned_starts': len(subset),
                    'precision_status': 'DEGENERATE_BOOTSTRAP_NO_EQUIVALENCE_CLAIM' if lo == hi
                                        else 'SMALL_DISCOVERY_FAMILY_BOOTSTRAP_APPROXIMATION'})
    return {'contrasts': results, 'replicates': n_boot, 'seed': manifest['analysis_seed'],
            'multiplicity': multiplicity, 'endpoints': list(endpoints), 'scopes': scopes, 'scope_aliases': aliases,
            'simultaneous_method': 'Bonferroni percentile family-cluster bootstrap approximation',
            'resampling': 'whole families, paired arms, seeds averaged within start, starts weighted equally',
            'scope_limit': 'This design and its frozen horizon only; other overnight designs require separate multiplicity accounting.',
            'uncertainty_limit': 'Small discovery samples and degenerate binary outcomes can make bootstrap coverage unreliable; no equivalence or noninferiority claim.'}


def missing_sensitivity(manifest, records):
    rows = cells(manifest, records, ('both_positive', 'measurement_unknown'))
    contrasts = []
    for scope in scopes_for(manifest)[0]:
        subset = [r for r in rows if scope == 'all' or r['transition'] == scope]
        for left, right in manifest['planned_contrasts']:
            lower = sum(r['means']['both_positive'][left] - r['means']['both_positive'][right] -
                        r['means']['measurement_unknown'][right] for r in subset) / len(subset)
            upper = sum(r['means']['both_positive'][left] + r['means']['measurement_unknown'][left] -
                        r['means']['both_positive'][right] for r in subset) / len(subset)
            contrasts.append({'scope': scope, 'arm': left, 'reference': right,
                              'unknown_outcome_identification_bounds': [float(lower), float(upper)]})
    return {'contrasts': contrasts, 'confidence_interval': False,
            'definition': 'All assignments lacking a valid reader pair, including failed generation, may be 0 or 1 adversarially by arm. These identification bounds are descriptive, not confidence intervals.'}


def arm_summary(manifest, records):
    output = {}
    for arm in (a['name'] for a in manifest['arms']):
        rows = [r for r in records if r['arm'] == arm]
        output[arm] = {'assigned': len(rows), 'gradeable': sum(r['measurement_status'] == 'gradeable' for r in rows),
            'reader_pair_valid': sum(r['reader_pair_valid'] for r in rows),
            'both_positive': sum(r['both_positive'] for r in rows),
            'at_least_one_positive': sum(r['at_least_one_positive'] for r in rows),
            'reader_disagreement': sum(r['reader_agreement'] is False for r in rows),
            'reasoning_closed': sum(r['closed_reasoning'] for r in rows),
            'hit_cap': sum(r['hit_cap'] for r in rows), 'emitted_tokens': sum(r['emitted_tokens'] for r in rows),
            'statuses': dict(Counter(r['operational_status'] for r in rows))}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'arm-map', 'frame', 'price', 'ratings-out', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'), '50k bootstrap requires CPU Slurm step')
    manifest, arm_map, frame, price = map(base.sealed, (args.manifest, args.arm_map, args.frame, args.price))
    require(manifest['bootstrap_replicates'] == BOOTSTRAPS, 'frozen 50k bootstrap changed')
    stage, votes = reader_votes(frame, price, args.ratings_out)
    records = join(manifest, arm_map, frame, votes)
    require(len(records) == manifest['expected_requests'], 'ITT omitted assigned requests')
    args.out.mkdir(parents=True, exist_ok=True)
    assigned = rating.save(args.out / 'ASSIGNED_RESULTS.json', {
        'schema': 'overnight-semantic-assigned-results-v1', 'manifest_sha256': manifest['sha256'],
        'arm_map_sha256': arm_map['sha256'], 'frame_sha256': frame['sha256'],
        'reader_stage_sha256': stage['sha256'], 'records': records}, existing_ok=True)
    body = {'schema': 'overnight-semantic-itt-v1', 'manifest_sha256': manifest['sha256'],
        'arm_map_sha256': arm_map['sha256'], 'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
        'reader_stage_sha256': stage['sha256'], 'assigned_results_sha256': assigned['sha256'],
        'analysis_driver_sha256': base.file_sha(__file__), 'rubric_sha256': base.file_sha(rating.RUBRIC),
        'horizon': manifest['horizon'], 'assigned': len(records), 'families': len({r['family'] for r in records}),
        'starts': len(manifest['rows']), 'arm_summary': arm_summary(manifest, records),
        'primary': clustered_contrasts(manifest, records),
        'either_reader_sensitivity': clustered_contrasts(manifest, records, ('at_least_one_positive',)),
        'missing_measurement_sensitivity': missing_sensitivity(manifest, records),
        'execution_position_descriptive': {
            f'{arm}|{position}': {'assigned': len(rows), 'both_positive': sum(r['both_positive'] for r in rows)}
            for arm in (a['name'] for a in manifest['arms'])
            for position in range(len(manifest['arms']))
            if (rows := [r for r in records if r['arm'] == arm and r['execution_position'] == position])},
        'interpretation': 'Exploratory discovery-only same-prefix causal comparison with arm-blind LLM outcomes. All assigned errors/caps/closures/invalid ratings retained. No independent confirmation, human validation, original-prompt accuracy, token-saving utility, equivalence or optimal-trajectory claim.'}
    result = rating.save(args.out / 'ANALYSIS.json', body, existing_ok=True)
    print(json.dumps({'status': 'COMPLETE_DISCOVERY_ITT_ANALYSIS', 'sha256': result['sha256'],
                      'assigned': result['assigned'], 'families': result['families'], 'horizon': result['horizon']}), flush=True)


if __name__ == '__main__':
    main()
