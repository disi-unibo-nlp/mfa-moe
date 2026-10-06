"""Assigned-population target engagement from sealed, already generated routes.

No model loads, semantic labels, outcomes-based enrollment, or unsaved gate
weights enter this independently frozen secondary measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import generated_dense_pipeline_v1 as source

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
DEFINITIONS = DOC / 'ROUTING_FIRST_STAGE_DEFINITIONS_v1.md'
PLAN = DOC / 'ROUTING_FIRST_STAGE_PLAN_v1.json'
BOOTSTRAPS = 50000
SEED = 20261004
require = source.require


def code_files():
    names = ('analyze_routing_first_stage_v1.py', 'analyze_routing_first_stage_v1.sbatch',
             'submit_routing_first_stage_v1.py', 'submit_routing_first_stage_v1.sbatch',
             'generated_dense_pipeline_v1.py', 'submit_generated_dense_chain_v1.py',
             'dispatch_generated_dense_v1.py', 'dispatch_overnight_readers_v1.py',
             'freeze_mechanism_extension_220_v1.py')
    paths = [Path(__file__).with_name(name) for name in names]
    return {str(path): source.file_sha(path) for path in paths}


def validate_plan():
    plan = source.sealed(PLAN)
    require(plan['schema'] == 'routing-first-stage-plan-v1' and
            plan['code_files'] == code_files() and
            plan['definitions_sha256'] == source.file_sha(DEFINITIONS) and
            plan['bootstrap_replicates'] == BOOTSTRAPS and plan['analysis_seed'] == SEED,
            'frozen first-stage source or definitions changed')
    return plan


def output_path(manifest, parent, plan):
    return Path(parent) / ('routing-first-stage-v1-' + manifest['sha256'][:16] + '-' + plan['sha256'][:12])


def frozen_contrasts(manifest):
    """Exact semantic contrast inventory; no new promising comparisons."""
    transitions = list(dict.fromkeys(r['transition'] for r in manifest['rows']))
    aliases = {transitions[0]: 'all'} if len(transitions) == 1 else {}
    if manifest['schema'] == 'overnight-routing-manifest-v2':
        original = manifest['planned_contrasts_scoped']
    else:
        original = [{'scope': scope, 'left': left, 'right': right}
                    for scope in manifest['analysis_scopes']
                    for left, right in manifest['planned_contrasts']]
    result = []
    for item in original:
        require(set(item) == {'scope', 'left', 'right'} and item['left'] != item['right'],
                'malformed frozen contrast')
        value = {**item, 'scope': aliases.get(item['scope'], item['scope'])}
        require(value['scope'] == 'all' or value['scope'] in transitions, 'unknown contrast scope')
        if value not in result:
            result.append(value)
    require(result, 'no frozen contrast family')
    return result


def target_spec(meta, actions, horizon):
    """Resolve the LEFT request's exact scheduled targets, including its seed."""
    if not meta['policies']:
        return None
    require(len(meta['policies']) == len(meta['slots']) and meta['slots'] in ([0], [0, 512]),
            'unsupported intended pulse geometry')
    pulses = []
    for slot, name in zip(meta['slots'], meta['policies'], strict=True):
        action = actions[name]
        require(action['transition'] == meta['transition'], 'target action transition differs')
        targets = [(layer, expert) for layer, experts in action['experts'] for expert in experts]
        require(targets and len(set(targets)) == len(targets) and
                all(type(layer) is int and type(expert) is int and 0 <= layer < 40 and 0 <= expert < 256
                    for layer, expert in targets), 'malformed selected layer/expert IDs')
        end = min(slot + 256, horizon)
        require(end > slot, 'pulse lies outside the frozen horizon')
        pulses.append({'slot': slot, 'end': end, 'action': name, 'targets': [list(x) for x in targets]})
    return {'pulses': pulses, 'intended_opportunities': sum(
            (p['end'] - p['slot']) * len(p['targets']) for p in pulses)}


def measure(row, routes, spec):
    """Membership inclusion; closing-token prediction is still an active row."""
    tokens = row['tokens']
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), 'invalid emitted IDs')
    closure = tokens.index(source.THINK_END) if source.THINK_END in tokens else None
    observed_end = min(len(tokens), closure + 1) if closure is not None else len(tokens)
    intended = spec['intended_opportunities']
    executed = sum(max(0, min(p['end'], observed_end) - p['slot']) * len(p['targets'])
                   for p in spec['pulses'])
    valid = not row['error'] and row['finish'] in ('stop', 'length')
    observed_hits = None
    if routes is not None:
        require(routes.shape == (len(tokens), 40, 8) and np.issubdtype(routes.dtype, np.integer) and
                ((routes >= 0) & (routes < 256)).all() and
                (np.diff(np.sort(routes, axis=-1), axis=-1) > 0).all(), 'invalid top-eight route IDs')
        observed_hits = 0
        for pulse in spec['pulses']:
            end = min(pulse['end'], observed_end)
            if end <= pulse['slot']:
                continue
            for layer, expert in pulse['targets']:
                observed_hits += int(np.count_nonzero(np.any(
                    routes[pulse['slot']:end, layer, :] == expert, axis=1)))
    known = valid and (observed_hits is not None or executed == 0)
    value = (observed_hits or 0) / intended if known else None
    bounds = [value, value] if known else [0., executed / intended if valid else 1.]
    return {'intended_opportunities': intended, 'observed_executed_opportunities': executed,
            'observed_selected_expert_hits': observed_hits, 'endpoint': value,
            'endpoint_bounds': bounds, 'point_identified': known,
            'conditional_inclusion': observed_hits / executed if known and executed else None,
            'exposure_fraction': executed / intended if valid else None,
            'no_pulse_exposure': executed == 0 if valid else None,
            'missing_kind': None if known else 'generation_failure' if not valid else 'missing_route_array',
            'observed_closure_index': closure}


def pair_inventory(manifest, expected):
    actions = {a['name']: a for a in manifest['actions']}
    cells = {(r['prefix_uid'], r['seed'], r['arm']): r for r in expected}
    require(len(cells) == len(expected), 'duplicate assignment cell')
    pairs, wanted = [], defaultdict(dict)
    for pair in frozen_contrasts(manifest):
        records = []
        supported = True
        for prefix in manifest['rows']:
            if pair['scope'] != 'all' and prefix['transition'] != pair['scope']:
                continue
            for seed in manifest['seeds']:
                left = cells[(prefix['uid'], seed, pair['left'])]
                right = cells[(prefix['uid'], seed, pair['right'])]
                require(all(left[k] == right[k] for k in
                            ('prefix_uid', 'family', 'transition', 'seed', 'prompt_sha256', 'prompt_len')),
                        'contrast is not an exact prefix/seed pair')
                spec = target_spec(left, actions, manifest['horizon'])
                if spec is None:
                    supported = False
                    continue
                key = source.digest(spec)
                wanted[left['uid']][key] = wanted[right['uid']][key] = spec
                records.append({'prefix_uid': prefix['uid'], 'family': prefix['family'],
                                'transition': prefix['transition'], 'seed': seed,
                                'left_uid': left['uid'], 'right_uid': right['uid'], 'target_spec_sha256': key})
        pairs.append({**pair, 'supported': supported,
                      'unsupported_reason': None if supported else 'Left arm has no selected expert set.',
                      'records': records})
    return pairs, wanted


def measurements(found, arrays, wanted):
    """Read each archive once, rechecking its immutable byte digest."""
    grouped, result, rows = defaultdict(list), {}, {r['uid']: r for r in found}
    for uid in wanted:
        grouped[arrays[uid]['path']].append(uid)
    for path, uids in grouped.items():
        expected_hashes = {arrays[uid]['sha256'] for uid in uids}
        require(len(expected_hashes) == 1 and source.file_sha(path) in expected_hashes, 'changed route archive')
        with np.load(path, allow_pickle=False) as archive:
            for uid in uids:
                route = archive[uid] if arrays[uid]['available'] else None
                result[uid] = {key: measure(rows[uid], route, spec) for key, spec in wanted[uid].items()}
    return result


def infer(pairs, values, *, n_boot=BOOTSTRAPS, seed=SEED):
    """Pair before clustering; seeds, then starts, then equal family weight."""
    require(n_boot >= 100, 'insufficient bootstrap replicates')
    multiplicity = sum(p['supported'] for p in pairs)
    output, assigned = [], []
    tail = .05 / (2 * max(1, multiplicity))
    for index, pair in enumerate(pairs):
        identity = {key: pair[key] for key in ('scope', 'left', 'right')}
        if not pair['supported']:
            output.append({**identity, 'status': 'UNSUPPORTED_NO_LEFT_TARGET', 'estimate': None,
                           'simultaneous_ci95': None, 'reason': pair['unsupported_reason']})
            continue
        by_prefix = defaultdict(list)
        for row in pair['records']:
            key = row['target_spec_sha256']
            left, right = values[row['left_uid']][key], values[row['right_uid']][key]
            bounds = [left['endpoint_bounds'][0] - right['endpoint_bounds'][1],
                      left['endpoint_bounds'][1] - right['endpoint_bounds'][0]]
            record = {**identity, **row, 'left_measurement': left, 'right_measurement': right,
                      'difference_bounds': bounds}
            assigned.append(record)
            by_prefix[(row['family'], row['prefix_uid'])].append(record)
        by_family = defaultdict(list)
        for (family, prefix), block in by_prefix.items():
            require(sorted(r['seed'] for r in block) == [0, 1], 'paired assigned seeds incomplete')
            by_family[family].append(np.mean([r['difference_bounds'] for r in block], axis=0))
        families = sorted(by_family)
        family_bounds = np.asarray([np.mean(by_family[f], axis=0) for f in families], dtype=float)
        unknown = sum(not r['left_measurement']['point_identified'] or
                      not r['right_measurement']['point_identified'] for block in by_prefix.values() for r in block)
        bounds = family_bounds.mean(axis=0).tolist() if families else [None, None]
        identified = bool(families) and unknown == 0
        interval = None
        if len(families) >= 2:
            rng = np.random.default_rng(seed + index)
            draws = rng.multinomial(len(families), [1 / len(families)] * len(families), size=n_boot)
            boot = draws @ family_bounds / len(families)
            interval = [float(np.quantile(boot[:, 0], tail)), float(np.quantile(boot[:, 1], 1 - tail))]
        estimate = bounds[0] if identified else None
        output.append({**identity, 'status': 'POINT_IDENTIFIED' if identified else 'PARTIALLY_IDENTIFIED',
            'estimate': estimate, 'identification_bounds': bounds,
            'simultaneous_ci95': interval if identified else None,
            'simultaneous_uncertainty_envelope95': interval if not identified else None,
            'families': len(families), 'assigned_starts': len(by_prefix), 'paired_seed_cells': len(pair['records']),
            'unknown_paired_seed_cells': unknown,
            'precision_status': 'INSUFFICIENT_FAMILIES' if len(families) < 2 else
                'DEGENERATE_BOOTSTRAP_NO_EQUIVALENCE_CLAIM' if interval[0] == interval[1] else
                'FAMILY_BOOTSTRAP_APPROXIMATION'})
    return {'contrasts': output, 'replicates': n_boot, 'analysis_seed': seed, 'multiplicity': multiplicity,
            'endpoint': 'selected_expert_inclusion_per_intended_opportunity',
            'method': 'Paired family-cluster Bonferroni percentile bootstrap approximation; bounds envelope when missing.',
            'weighting': 'Seeds averaged within prefix; prefixes within family; equal family weight.',
            'scope': 'One separately corrected secondary family in this stage; no cross-stage pooled claim.'}, assigned


def operational_summary(manifest, found):
    actions = {a['name']: a for a in manifest['actions']}
    groups = defaultdict(list)
    for row in found:
        groups[(row['transition'], row['arm'])].append(row)
    result = []
    for (transition, arm), block in sorted(groups.items()):
        exposures = [measure(row, None, spec)['observed_executed_opportunities']
                     for row in block if (spec := target_spec(row, actions, manifest['horizon'])) is not None
                     and not row['error'] and row['finish'] in ('stop', 'length')]
        result.append({'transition': transition, 'arm': arm, 'assigned': len(block),
            'generation_errors': sum(bool(r['error']) for r in block),
            'missing_route_arrays': sum(not r['routed_present'] for r in block),
            'observed_reasoning_closures': sum(source.THINK_END in r['tokens'] for r in block),
            'natural_stops': sum(not r['error'] and r['finish'] == 'stop' for r in block),
            'length_stops': sum(not r['error'] and r['finish'] == 'length' for r in block),
            'emitted_tokens': sum(len(r['tokens']) for r in block),
            'valid_target_exposure_cells': len(exposures),
            'target_nonfires': sum(x == 0 for x in exposures) if exposures else None,
            'finish_counts': dict(Counter(str(r['finish']) for r in block)),
            'native_arm': all(r['role'] == 'native' for r in block)})
    return result


def telemetry_summary(found):
    """Keep ranks distinct; these sparse same-hidden-state weights are saved."""
    groups, coverage = {}, []
    for row in found:
        payload = row.get('action_dose', {})
        coverage.append({'uid': row['uid'], 'arm': row['arm'], 'role': row['role'],
                         'ranks_present': sorted(payload), 'missing_telemetry_ranks': row.get('missing_telemetry_ranks', []),
                         'action_rows_and_segments_by_rank': {rank: {'rows': value.get('rows'),
                             'segments': value.get('segments')} for rank, value in payload.items()}})
        for rank, value in payload.items():
            for action, layers in (value.get('dose') or {}).items():
                for layer, counters in layers.items():
                    key = row['transition'], row['arm'], rank, action, layer
                    item = groups.setdefault(key, {'transition': key[0], 'arm': key[1], 'rank': rank,
                        'action': action, 'layer': int(layer), 'observed_requests': 0, 'totals': defaultdict(float)})
                    item['observed_requests'] += 1
                    for name, number in counters.items():
                        require(isinstance(number, (int, float)) and np.isfinite(number) and number >= 0,
                                'invalid saved routing telemetry')
                        item['totals'][name] += number
    result = []
    for item in groups.values():
        active = item['totals'].get('active_rows', 0)
        l1 = item['totals'].get('weight_l1')
        item['totals'] = dict(item['totals'])
        item['mean_sparse_weight_l1_per_active_row'] = l1 / active if active and l1 is not None else None
        result.append(item)
    return {'records': result, 'assignment_coverage': coverage,
            'scope': 'Saved per-action, per-layer, per-TP-rank same-hidden-state native-versus-edited sparse top-eight weights.',
            'independence': 'TP ranks are redundant computational measurements, never independent samples or summed ranks.',
            'unavailable': ['per-token native gate-weight arrays', 'full gate-distribution TV', 'gate-weight velocity and acceleration']}


def artifacts(directory, inference, operations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = inference['contrasts']
    keys = ('scope', 'left', 'right', 'status', 'estimate', 'identification_bounds',
            'simultaneous_ci95', 'simultaneous_uncertainty_envelope95', 'families',
            'assigned_starts', 'unknown_paired_seed_cells', 'precision_status')
    with (directory / 'CONTRASTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key)
                             for key in keys})
    with (directory / 'ARMS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operations[0]) if operations else ['arm']); writer.writeheader()
        for row in operations:
            writer.writerow(row)
    fig, ax = plt.subplots(figsize=(11, max(3, .28 * len(rows) + 1.5)))
    for i, row in enumerate(rows):
        interval = row.get('simultaneous_ci95') or row.get('simultaneous_uncertainty_envelope95')
        if interval:
            ax.plot(interval, [i, i], color='#2463A0' if row['estimate'] is not None else '#A36322')
        if row.get('estimate') is not None:
            ax.plot(row['estimate'], i, 'o', color='#2463A0', markersize=4)
    ax.set_yticks(range(len(rows)), [f"{r['scope']}: {r['left']} − {r['right']}" for r in rows], fontsize=8)
    ax.axvline(0, color='.5', linewidth=.8); ax.invert_yaxis()
    ax.set_xlabel('Difference in selected-expert inclusion per intended opportunity')
    ax.set_title('Routing target engagement: corrected paired family intervals\nBrown: missing-outcome uncertainty envelope; no semantic effect claim')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / ('TARGET_ENGAGEMENT.' + suffix), dpi=180)
    plt.close(fig)
    return {path.name: source.file_sha(path) for path in directory.iterdir()
            if path.name in ('CONTRASTS.csv', 'ARMS.csv', 'TARGET_ENGAGEMENT.png', 'TARGET_ENGAGEMENT.pdf')}


def run(args):
    source.require_step()
    plan = validate_plan()
    manifest, price = source.sealed(args.manifest), source.sealed(args.price)
    require(manifest['analysis_seed'] == SEED and manifest['bootstrap_replicates'] == BOOTSTRAPS and
            manifest['pulse_width'] == 256 and manifest['seeds'] == [0, 1], 'frozen analysis settings differ')
    directory = output_path(manifest, args.out_parent, plan)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = directory / 'RESULT.json'
        if completed.exists():
            result = source.sealed(completed)
            require(result['manifest_sha256'] == manifest['sha256'] and result['plan_sha256'] == plan['sha256'] and
                    all(source.file_sha(directory / path) == sha for path, sha in result['artifacts'].items()),
                    'completed first-stage output binding changed')
            print('ALREADY_COMPLETE', result['sha256'], flush=True)
            return
        started = time.monotonic()
        source.seal_generation_if_needed(manifest, args.manifest, args.generation_out)
        stage, expected, found, arrays, receipts = source.source_outputs(manifest, price, args.generation_out)
        pairs, wanted = pair_inventory(manifest, expected)
        values = measurements(found, arrays, wanted)
        inference, assigned = infer(pairs, values)
        observations = source.save(directory / 'ASSIGNED_ENGAGEMENT.json', {
            'schema': 'routing-first-stage-assigned-v1', 'plan_sha256': plan['sha256'],
            'manifest_sha256': manifest['sha256'], 'assigned_generation_requests': len(expected),
            'paired_contrast_cells': assigned, 'source_receipts': receipts,
            'generation_assignment_inventory': [{**{key: row[key] for key in
                ('uid', 'prefix_uid', 'family', 'transition', 'seed', 'arm', 'role', 'finish', 'error', 'routed_present')},
                'observed_emitted_tokens': len(row['tokens']),
                'observed_reasoning_closed': source.THINK_END in row['tokens'],
                'route_source': arrays[row['uid']]} for row in found],
            'target_specs': {key: value for specs in wanted.values() for key, value in specs.items()},
            'scope': 'Every requested semantic contrast uses its left targets in both matched traces.'})
        operations = operational_summary(manifest, found)
        telemetry = source.save(directory / 'TELEMETRY.json', {'schema': 'routing-first-stage-telemetry-v1',
            'manifest_sha256': manifest['sha256'], 'plan_sha256': plan['sha256'], **telemetry_summary(found)})
        plots = artifacts(directory, inference, operations)
        plots.update({p.name: source.file_sha(p) for p in (directory / 'ASSIGNED_ENGAGEMENT.json', directory / 'TELEMETRY.json')})
        result = source.save(completed, {'schema': 'routing-first-stage-result-v1', 'status': 'COMPLETE_SECONDARY_MECHANISTIC_ANALYSIS',
            'plan_sha256': plan['sha256'], 'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
            'stage_completion_sha256': stage['sha256'], 'assigned_results_sha256': observations['sha256'],
            'telemetry_sha256': telemetry['sha256'], 'horizon': manifest['horizon'],
            'assigned_requests': len(expected), 'inference': inference, 'operational_by_arm': operations,
            'artifacts': plots, 'cpu_analysis_wall_seconds': time.monotonic() - started,
            'claim_limit': 'Target engagement and saved same-state routing dose only; no semantic-control, accuracy, or latent-reasoning claim.'})
        print(result['status'], str(directory), result['sha256'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'price', 'generation-out', 'out-parent'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
