"""Saved X2 diagnostics and registered G3 selection; no semantic-control inference."""
from __future__ import annotations
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
from moe_steer import manifests as M, reduce as RED, results as RS
from moe_steer.spec import PolicyTable
from moe_steer.trigger import Lexicon, TriggerFSM, VocabBytes, _MarkerMatcher

S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
helper = Path(__file__).with_name('g3_inventory.py')
if not helper.is_file():
    helper = S/'runs/x2prep/g3_inventory/g3_inventory.py'
spec = importlib.util.spec_from_file_location('frozen_g3_diagnostics', helper)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
SIMULTANEOUS_CONTRASTS = 564  # 42 E-N x 3 metrics + 438 possible E-M magnitude-pair marker contrasts.


class SentenceMatcher(_MarkerMatcher):
    """Record every matching sentence, preserving the frozen splitter and onset sensitivity."""
    def __init__(self, lexicon):
        super().__init__(lexicon)
        self.all_decisions = []

    def _match(self, marker, o):
        if o-self.first_o+1 <= self.limit:
            self.all_decisions.append((o, marker, self.first_o))
        super()._match(marker, o)


def new_fsm(lexicon, vocab):
    fsm = TriggerFSM({'kind': 'none'}, lexicon, vocab, 'diagnostic-only')
    matcher = SentenceMatcher(lexicon)
    fsm._matcher = matcher
    fsm._splitter.sink = matcher
    return fsm


def marker_counts(state, suffix, lexicon, vocab):
    fsm = copy.deepcopy(state, {id(lexicon): lexicon, id(vocab): vocab, id(state._hash): state._hash.copy()})
    prefix = fsm.n
    fsm._matcher.all_decisions.clear()
    for i, token in enumerate(suffix[:512]):
        fsm.feed(prefix+i, token)
    count = sum(prefix <= first and prefix <= decision < prefix+512
                for decision, marker, first in fsm._matcher.all_decisions)
    return int(count), G.marker_count(fsm.events, prefix)


def choose_match(target_dose, candidates, force=False):
    """Return the frozen nearest same-operator ladder cell; ties favor smaller magnitude."""
    if target_dose <= 0:
        return {'eligible': False, 'reason': 'ZERO_TARGET_DOSE'}
    valid = [c for c in candidates if c['D'] > 0]
    if not valid:
        return {'eligible': False, 'reason': 'NO_POSITIVE_RANDOM_DOSE'}
    chosen = min(valid, key=lambda c: (abs(math.log(c['D']/target_dose)), c['magnitude']))
    ratio = chosen['D']/target_dose
    return {**chosen, 'ratio': ratio, 'off_band': not .9 <= ratio <= 1.1,
            'eligible': .9 <= ratio <= 1.1}


def interval(values, families, count=SIMULTANEOUS_CONTRASTS):
    """Question-weighted estimates; family-clustered approximate t interval and bootstrap sensitivity."""
    from scipy.stats import t
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        return {'estimate': None, 'status': 'INCOMPLETE_MEASUREMENT'}
    estimate = float(values.mean())
    groups = sorted(set(families))
    sums = np.array([np.sum(values[np.array(families) == f]-estimate) for f in groups])
    se = float(np.sqrt(len(groups)/(len(groups)-1)*np.sum(sums*sums))/len(values))
    half = float(t.ppf(1-.05/(2*count), len(groups)-1)*se)
    rng = np.random.default_rng(20261001)
    boots = np.mean(values[rng.integers(len(values), size=(5000, len(values)))], axis=1)
    return {'estimate': estimate, 'question_bootstrap_95': np.quantile(boots, [.025, .975]).tolist(),
        'family_clustered_Bonferroni_t_approximation': [estimate-half, estimate+half],
        'family': 'all 126 E-N diagnostic contrasts and all 438 possible paired random-ladder marker contrasts',
        'simultaneous_contrast_count': count, 'n_questions': len(values), 'n_families': len(groups),
        'constant_cluster_diagnostic': bool(np.all(sums == 0)),
        'assumptions': 'independent families and approximate cluster t inference; boundary intervals do not prove retention/equivalence'}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('X2 replay, routed-array reduction and intervals require CPU Slurm')
    frozen = json.loads((S/'runs/x2-resume-v1/FROZEN.json').read_text())
    manifest = M.load_manifest(frozen['manifest'])
    table = PolicyTable.from_sealed(manifest['policy_table'])
    output = Path(frozen['output'])
    records = RS.read_shard_records(output, 0)
    by_uid = {r['uid']: r for r in records}
    if len(by_uid) != 6630 or set(by_uid) != {r['uid'] for r in manifest['requests']}:
        raise ValueError('X2 analysis requires every assigned UID exactly once')
    prefixes_path = S/'runs/x2-resume-v1/measurement-prefixes/PREFIXES.json'
    prefixes = json.loads(prefixes_path.read_text())
    if prefixes['manifest_sha256'] != manifest['sha256']:
        raise ValueError('prefix binding differs')
    out = S/'runs/x2-resume-v1/analysis'
    out.mkdir(exist_ok=True)
    if (out/'estimates.json').exists():
        raise ValueError('versioned X2 result already exists; do not overwrite')
    lexicon = Lexicon.load(manifest['files']['lexicon']['path'])
    vocab = VocabBytes.load(manifest['files']['vocab']['path'])
    states = {}
    for question, row in prefixes['questions'].items():
        state = new_fsm(lexicon, vocab)
        state.fast_forward(row['prefix_token_ids'][len(row['original_prompt_token_ids']):])
        states[question] = state
    telemetry, info = RED.load_telemetry(output/'telemetry/shard-0')
    if set(telemetry) != {0, 1} or any(set(r) != set(by_uid) for r in telemetry.values()) or info['partial_lines']:
        raise ValueError('complete TP2 telemetry required')
    amendment = json.loads((S/'runs/resume-v1/NLL_RECOVERY_AMENDMENT.json').read_text())
    nll_path = S/'runs/x2-resume-v1/native-nll'/manifest['sha256']/amendment['new_addendum']['tree_sha256']/'native-nll.json'
    nll, validation = {}, None
    if nll_path.is_file():
        native = json.loads(nll_path.read_text())
        validation = native['validation']
        if not all(v['pass'] for v in validation.values()) or native['binding']['manifest_sha256'] != manifest['sha256']:
            raise ValueError('native NLL parity or binding failed; no NLL read')
        nll = {r['uid']: r for r in native['records']}
        if set(nll) != {r['uid'] for r in manifest['requests'] if r['arm'] in ('N', 'E+', 'E-')}:
            raise ValueError('native NLL is incomplete')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(frozen['manifest']), prefixes_path, helper, output/'shard-0.index.jsonl',
         output/'telemetry/shard-0/rank0.jsonl', output/'telemetry/shard-0/rank1.jsonl']}
    if nll: hashes[str(nll_path)] = hashlib.sha256(nll_path.read_bytes()).hexdigest()
    (out/'FROZEN.json').write_text(json.dumps({'schema': 'x2-saved-analysis-v1', 'inputs': hashes,
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'job_id': os.environ['SLURM_JOB_ID'],
        'marker_primary': 'all matching sentences, literal R1-A4', 'marker_sensitivity': 'legacy onset-only helper',
        'multiplicity_contrasts': SIMULTANEOUS_CONTRASTS, 'NLL_available': bool(nll)}, indent=1)+'\n')
    observations, rows_by_key, native_arrays = [], {}, {}
    native_requests = {(r['question'], r['seed_k']): r for r in manifest['requests'] if r['arm'] == 'N'}
    routed_hashes = {}
    def routed(request):
        path = output/'shard-0.routed'/f'{request["uid"]}.npz'
        arrays, meta = RS.load_routed(path)
        routed_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if meta['uid'] != request['uid'] or meta['rows'] != len(by_uid[request['uid']]['completion_token_ids']):
            raise ValueError('routed row/UID mismatch')
        return arrays, meta
    for key, request in native_requests.items():
        native_arrays[key] = routed(request)
    for index, request in enumerate(manifest['requests']):
        uid, question, seed = request['uid'], request['question'], request['seed_k']
        record = by_uid[uid]
        RS.validate_result(record, request)
        if record['code_tree'] != manifest['code_tree'] or record['manifest_sha256'] != manifest['sha256']:
            raise ValueError('X2 record input binding mismatch')
        if not RED.compare_ranks(uid, {rank: data[uid] for rank, data in telemetry.items()})['agree']:
            raise ValueError('TP2 telemetry disagreement')
        policy = table.policies[request['policy_index']]
        layers = list(policy.targets.layers()) if policy.targets else []
        te = telemetry[0][uid]
        dose = G.tv_dose(te['counters'], layers, te['cpu_active_rows'])
        if request['arm'] == 'N' and te['cpu_active_rows'] != 0:
            raise ValueError('native sham received active edits')
        if request['arm'] != 'N' and dose['D'] <= 0:
            raise ValueError('non-native assigned cell has zero executed dose')
        if abs(dose['D']-dose['D_from_summary_fields']) > 1e-4:
            raise ValueError('layer and aggregate dose disagree')
        count, onsets = marker_counts(states[question], record['completion_token_ids'], lexicon, vocab)
        row = {'uid': uid, 'question': question, 'seed_k': seed, 'arm': request['arm'],
            'policy': policy.name, 'D': dose['D'], 'D_normalized_30': dose['D']*len(layers)/30,
            'active_rows': te['cpu_active_rows'], 'marker_count': count, 'onset_count': onsets,
            'degeneration': int(G.degeneration_flag(record['completion_token_ids'])),
            'tokens': len(record['completion_token_ids']), 'finish_reason': record['finish_reason'],
            'mean_nll': nll.get(uid, {}).get('mean_nll'), 'NLL_status': nll.get(uid, {}).get('status', 'PENDING'),
            'KL_windows': []}
        if request['arm'] != 'N':
            arrays, meta = routed(request)
            baseline, _ = native_arrays[(question, seed)]
            positions = [list(arrays['layers']).index(l) for l in layers]
            if list(arrays['layers']) != list(baseline['layers']):
                raise ValueError('E/M and N routed layer arrays differ')
            for start in (0, 256, 512, 768):
                a, b = arrays['pulse_ids'][start:start+256], baseline['pulse_ids'][start:start+256]
                vals = G.window_kl(a, b, positions) if len(a) and len(b) else {}
                row['KL_windows'].append({'start': start, 'end': start+256, 'rows_arm': len(a), 'rows_native': len(b),
                    'mean_layer_KL_nats': float(np.mean(list(vals.values()))) if vals else None,
                    'complete_256_both': len(a) == len(b) == 256})
        observations.append(row)
        rows_by_key[(policy.name, question, seed)] = row
        if index % 500 == 0: print(json.dumps({'processed': index, 'required': 6630}), flush=True)
    (out/'observations.json').write_text(json.dumps(observations, separators=(',', ':'))+'\n')
    (out/'routed-inventory.json').write_text(json.dumps(routed_hashes, indent=1)+'\n')
    questions = sorted(states)
    family_map = {q: f for f, members in frozen['families'].items() for q in members}
    families = [family_map[q] for q in questions]
    native_policy = next(r['policy_name'] for r in manifest['requests'] if r['arm'] == 'N')
    def vector(policy_name, field, seed=None):
        return np.array([np.mean([rows_by_key[(policy_name, q, k)][field] for k in (0, 1)])
            if seed is None else rows_by_key[(policy_name, q, seed)][field] for q in questions], float)
    native_markers = vector(native_policy, 'marker_count')*1000/512
    native_degen = vector(native_policy, 'degeneration')
    native_nll = vector(native_policy, 'mean_nll') if nll else None
    cells = []
    target_policies = [p for p in table.policies if p.name in {r['policy_name'] for r in manifest['requests'] if r['arm'] in ('E+', 'E-')}]
    for policy in target_policies:
        scope = next(s for s in ('L1', 'BAND', 'ALL') if '_'+s+'_' in policy.name)
        D = float(vector(policy.name, 'D').mean())
        matches = []
        for seed in (0, 1):
            candidates = []
            for other in target_policies:
                if other.operator.kind == policy.operator.kind and other.operator.sign == policy.operator.sign and '_'+scope+'_' in other.name:
                    name = other.name+'@random'+str(seed)
                    candidates.append({'policy': name, 'magnitude': other.operator.magnitude,
                        'D': float(vector(name, 'D', seed).mean())})
            matches.append(choose_match(D, candidates, policy.operator.kind == 'force'))
        markers = vector(policy.name, 'marker_count')*1000/512
        delta = markers-native_markers
        deg = vector(policy.name, 'degeneration')-native_degen
        random = None
        if all('policy' in m for m in matches):
            random = (vector(matches[0]['policy'], 'marker_count', 0)+vector(matches[1]['policy'], 'marker_count', 1))*500/512
        marker_change = float(delta.mean())
        target_random_change = float((markers-random).mean()) if random is not None else None
        dnll = vector(policy.name, 'mean_nll')-native_nll if nll else None
        checks = {'positive_dose': D > 0, 'random_dose_support': all(m['eligible'] for m in matches),
            'degeneration_increase_le_3pp': float(deg.mean()) <= .03,
            'marker_change_ge_15percent': float(native_markers.mean()) > 0 and abs(marker_change) >= .15*float(native_markers.mean())
                and (marker_change < 0 if policy.operator.sign < 0 else True),
            'target_random_same_direction_half_effect': target_random_change is not None and marker_change*target_random_change > 0
                and abs(target_random_change) >= .5*abs(marker_change),
            'native_NLL_increase_le_point10': None if dnll is None else bool(float(dnll.mean()) <= .1)}
        cell = {'policy': policy.name, 'scope': scope, 'operator': policy.operator.kind, 'sign': policy.operator.sign,
            'magnitude': policy.operator.magnitude, 'dose': interval(vector(policy.name, 'D'), families),
            'D_normalized_30': D*len(list(policy.targets.layers()))/30, 'matches': matches,
            'marker_rate_E': float(markers.mean()), 'marker_rate_N': float(native_markers.mean()),
            'marker_E_minus_N': interval(delta, families), 'degeneration_E_minus_N': interval(deg, families),
            'marker_E_minus_matched_M': interval(markers-random, families) if random is not None else None,
            'native_NLL_E_minus_N': interval(dnll, families) if dnll is not None else None,
            'onset_sensitivity_E_minus_N': interval((vector(policy.name, 'onset_count')-vector(native_policy, 'onset_count'))*1000/512, families),
            'checks': checks, 'G3_pass': all(v is True for v in checks.values()),
            'G3_status': 'EVALUATED' if nll else 'PENDING_NATIVE_NLL'}
        cells.append(cell)
    rank = lambda c: (('L1', 'BAND', 'ALL').index(c['scope']), ('reweight', 'bias', 'force').index(c['operator']), c['magnitude'])
    selected = {str(sign): next((c['policy'] for c in sorted(cells, key=rank) if c['sign'] == sign and c['G3_pass']), None)
                for sign in (-1, 1)}
    result = {'schema': 'legacy-X2-G3-results-v1', 'status': 'COMPLETE' if nll else 'PENDING_NATIVE_NLL',
        'manifest_sha256': manifest['sha256'], 'requests': 6630, 'questions': len(questions), 'eligible_of_48': len(questions),
        'ineligible_not_replaced': 48-len(questions), 'all_assigned_ITT': True, 'primary_unit': 'question with seed means',
        'sensitivity': 'frozen duplicate-family clustered simultaneous t approximation', 'telemetry': info,
        'native_NLL_validation': validation, 'native_NLL': str(nll_path) if nll else None,
        'marker_definition': 'all new matching sentences, excluding triggering sentence; onset-only sensitivity is not used for selection',
        'cells': cells, 'selected_policy_per_sign': selected,
        'interpretation': 'Registered dose-selection diagnostics on eligible dev-cal same-prefix branches; no semantic steering or engine-equivalence claim.',
        'uncertainty_limitations': 'Approximate family t and bootstrap intervals may degenerate at zero event counts; selection is a point-estimate diagnostic, not a significance test.'}
    (out/'estimates.json').write_text(json.dumps(result, indent=1, allow_nan=False)+'\n')
    with (out/'dose-cells.csv').open('w') as handle:
        fields = ['policy', 'scope', 'operator', 'sign', 'magnitude', 'D', 'D_normalized_30', 'marker_E_minus_N', 'G3_pass']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in cells:
            writer.writerow({**{k: c[k] for k in fields if k in c}, 'D': c['dose']['estimate'],
                'marker_E_minus_N': c['marker_E_minus_N']['estimate']})
    print(json.dumps({'path': str(out), 'status': result['status'], 'selected_policy_per_sign': selected}))


if __name__ == '__main__':
    run()
