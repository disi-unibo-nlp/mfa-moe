"""Prepare versioned legacy X3 eligibility, registered G3 selection and cost inventory.

Read-only inputs; writes only new, sealed report artifacts. Run the trace and
native-result scan on CPU Slurm. This program never builds or launches X3 GPU
requests and never changes the original X2/G3 files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
X1_MANIFEST = S / 'manifests/x1-v1.json'
X1_RESULTS = S / 'runs/x1'
OLD_SELECTION = S / 'runs/x2-resume-v1/analysis/G3_SELECTION.json'
AMENDMENT = REPORT / 'G3_DOSE_SUPPORT_AMENDMENT.json'
FAMILY = REPORT / 'family-freeze.json'
SELECTION_OUT = REPORT / 'X3_G3_SELECTION_v2.json'
ELIGIBILITY_OUT = REPORT / 'X3_ELIGIBILITY_v1.json'
PRICE_OUT = REPORT / 'X3_COST_INVENTORY_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'unsealed input: {path}')
    return value


def write_once(path, body):
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing version differs: {path}')
        return value
    tmp = path.with_name('.' + path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(path)
    return value


def selection():
    old, amendment = sealed(OLD_SELECTION), sealed(AMENDMENT)
    if old['schema'] != 'G3-dose-selection-v1' or not old['native_nll_parity_pass']:
        raise ValueError('old complete G3/NLL source is invalid')
    if amendment['registered_passing_cells'] != 2 or amendment['registered_selected'] != {
            '-1': 'reweight_m1_L1_landmark', '1': 'reweight_p2_BAND_landmark'}:
        raise ValueError('registered ±10% G3 support differs')
    selected = []
    for source in old['selected']:
        row = next((c for c in amendment['cells'] if c['policy'] == source['policy']), None)
        if row is None or not row['registered_G3_pass'] or not row['registered_random_dose_support']:
            raise ValueError('old selected dose fails registered amendment')
        if not all(.9 <= ratio <= 1.1 for ratio in row['random_dose_ratios']):
            raise ValueError('random dose outside registered ±10% band')
        if any(abs(row['random_dose_ratios'][k] - source['random_dose_ratios'][str(k)]) > 1e-10
               for k in (0, 1)):
            raise ValueError('old and amended random dose values differ')
        selected.append(dict(source, registered_G3_pass=True))
    if {str(c['sign']): c['policy'] for c in selected} != amendment['registered_selected']:
        raise ValueError('selection differs from registered ranking')
    body = {'schema': 'G3-dose-selection-v1', 'version': 'v2-registered-dose-support',
            'native_nll_parity_pass': True, 'manifest_sha256': old['manifest_sha256'],
            'selected': selected, 'old_selection_sha256': old['sha256'],
            'registered_amendment_sha256': amendment['sha256'],
            'registered_passing_cells': 2,
            'population': '39 eligible of 48 dev-cal questions; two seeds; no replacement',
            'interpretation': 'registered point-estimate dose selection only; no semantic steering claim',
            'X3_launch': 'HOLD eligibility, all-in price and new-study priority'}
    return write_once(SELECTION_OUT, body)


def native_results(order):
    from moe_steer import manifests as M, results as RS
    manifest = M.load_manifest(X1_MANIFEST)
    wanted = set(order)
    assigned = {(r['question'], r['seed_k']): r for r in manifest['requests']
                if r['question'] in wanted and r['seed_k'] in (0, 1)}
    if len(assigned) != 192 or set(q for q, _ in assigned) != wanted:
        raise ValueError('X1 lacks both frozen native seeds for every X3 question')
    by_uid = {r['uid']: r for r in assigned.values()}
    found = set()
    receipt_shas = {}
    for shard in range(manifest['n_shards']):
        complete = json.loads((X1_RESULTS / f'shard-{shard}.complete.json').read_text())
        if complete['schema'] != 'steer-complete-v1' or complete['manifest_sha256'] != manifest['sha256']:
            raise ValueError('X1 native shard is not complete under its manifest')
        path = X1_RESULTS / f'shard-{shard}.results.jsonl.gz'
        if sha(path) != complete['files'][path.name]:
            raise ValueError('X1 native results changed since completion')
        receipt_shas[str(shard)] = sha(X1_RESULTS / f'shard-{shard}.complete.json')
        for record in RS.read_shard_records(X1_RESULTS, shard):
            request = by_uid.get(record['uid'])
            if request is not None:
                RS.validate_result(record, request)
                if record['manifest_sha256'] != manifest['sha256'] or record['code_tree'] != manifest['code_tree']:
                    raise ValueError('retained native result binding differs')
                found.add(record['uid'])
    if found != set(by_uid):
        raise ValueError(f'X1 native result missing: {len(set(by_uid) - found)}')
    return assigned, {'manifest_sha256': manifest['sha256'],
                      'manifest_file_sha256': sha(X1_MANIFEST), 'complete_receipt_file_sha256': receipt_shas,
                      'verified_native_uids': len(found)}


def eligibility():
    from moe_steer import engine, manifests as M
    from moe_steer.trigger import Lexicon, VocabBytes
    import x2_build as X
    from x3_build import pinned_order
    frozen = json.loads((S / 'runs/s2prop/FROZEN_ADDENDA.json').read_text())
    expected_tree = frozen['snapshots']['s2']['tree_sha256']
    if engine.code_tree_sha256() != expected_tree:
        raise ValueError('CPU eligibility requires frozen capped sampler tree')
    split, world, family = M.load_split(), M.load_world(), sealed(FAMILY)
    order = pinned_order(split)
    if digest(order[:64]) != '68ffb11fb9a7e60ca7b7ec2cde2a5632829372a436efd7d3ee6ac6f37089bd82':
        raise ValueError('registered first-64 order changed')
    assigned, native_binding = native_results(order)
    lexicon, vocab = Lexicon.load(world.lexicon_path), VocabBytes.load(world.vocab_path)
    store = M.TraceStore(cache_dir=S / 'runs/resume-v1/x3-cpu/trace-offsets')
    family_of = {q: f for f, questions in family['new_parent_pools']['families'].items() for q in questions}
    confirm_connected = set(family['new_parent_pools']['confirm_connected_excluded'])
    rows = []
    for index, question in enumerate(order):
        info = world.infos[question]
        ref = info['prompt_ref']
        trace = store.trace(ref['dataset'], ref['problem_id'])
        row = X.eligibility_row(info, trace, lexicon, vocab, cross_check=True)
        if row['offline_first_onset'] != row['streaming_first_onset']:
            raise ValueError(f'offline/streaming onset mismatch: {question}')
        family_id = family_of[question]
        row['family'] = family_id
        row['confirm_connected_family'] = family_id in confirm_connected
        row['original_order_index'] = index
        row['long_endpoint'] = index < 64
        row['native_uids'] = {str(seed): assigned[question, seed]['uid'] for seed in (0, 1)}
        row['parent_replay_sha256'] = digest(trace['metadata']['token_replay'])
        if row['eligible'] and index < 64 and row['prefix_len'] >= 32768:
            row.update(eligible=False, reason='onset_after_registered_32k_endpoint')
        row['endpoint_cap'] = 32768 if index < 64 else 1024
        rows.append(row)
    if len(rows) != 96 or len({r['question'] for r in rows}) != 96:
        raise ValueError('X3 does not retain all original 96 assigned questions')
    body = {'schema': 'legacy-X3-eligibility-v1', 'status': 'COMPLETE_CPU_PREPARED',
            'job_id': os.environ['SLURM_JOB_ID'], 'driver_sha256': sha(Path(__file__)),
            'split_sha256': split['sha256'], 'world_input_seals': world.seals,
            'family_freeze_sha256': family['sha256'], 'native_X1': native_binding,
            'first64_sha256': digest(order[:64]), 'onset_floor': X.ONSET_FLOOR,
            'onset_schedule': X.ONSET_SCHEDULE,
            'retained_all_assigned': True, 'no_replacement': True,
            'family_use': 'original X3 enrollment intact; confirm-connected families flagged for sensitivity only',
            'rows': rows,
            'summary': {'assigned': 96, 'long': 64, 'short': 32,
                        'eligible': sum(r['eligible'] for r in rows),
                        'nonfires': sum(not r['eligible'] for r in rows),
                        'nonfire_reasons': dict(Counter(r['reason'] for r in rows if not r['eligible'])),
                        'confirm_connected_families': len({r['family'] for r in rows if r['confirm_connected_family']})}}
    return write_once(ELIGIBILITY_OUT, body)


def price(eligibility_value, selection_value):
    cost = json.loads((S / 'runs/x2prep/cost_model.json').read_text())
    rate = cost['inputs']['x1']['steady_tok_per_s'] * cost['constants']['pessimistic_derate']
    scenarios = []
    for signs in (1, 2):
        arms = 1 + 2 * signs
        eligible = [r for r in eligibility_value['rows'] if r['eligible']]
        prefill = sum((r['prompt_tokens'] + r['prefix_len']) * 2 * arms for r in eligible)
        decode = sum((32768 - r['prefix_len'] if r['long_endpoint'] else 1024) * 2 * arms
                     for r in eligible)
        # Historical rates are a planning scenario, not qualified 32k timing.
        generation = decode / rate * 2 / 3600 + 838 * 2 / 3600
        prefill_gpu_h = prefill / cost['inputs']['q10']['prefill_tps'] * 2 / 3600
        scenarios.append({'signs': signs, 'arms_per_firing_question': arms,
                          'assigned_questions': 96, 'firing_questions': len(eligible),
                          'retained_nonfire_questions': 96 - len(eligible),
                          'requests': len(eligible) * 2 * arms,
                          'maximum_prefill_tokens_without_cache_reuse': prefill,
                          'maximum_decode_tokens': decode,
                          'historical_generation_GPUh_including_one_838s_cold_load': generation,
                          'historical_prefill_GPUh': prefill_gpu_h,
                          'historical_generation_plus_prefill_GPUh': generation + prefill_gpu_h,
                          'native_NLL_GPUh': None, 'semantic_labeling_GPUh': None,
                          'grading_GPUh': None, 'retries_and_shutdown_GPUh': None,
                          'status': 'HOLD_CONTEXT_MATCHED_COMPLETE_PRICE'})
    body = {'schema': 'legacy-X3-cost-inventory-v1', 'status': 'HOLD_COMPLETE_PRICE_AND_PRIORITY',
            'eligibility_sha256': eligibility_value['sha256'],
            'registered_selection_sha256': selection_value['sha256'],
            'historical_cost_model_file_sha256': sha(S / 'runs/x2prep/cost_model.json'),
            'original_X3_ceiling_GPUh': 12., 'scenarios': scenarios,
            'required_next_measurements': ['context-matched 32k generation/prefill runtime',
                                           'semantic-labeling and grading runtime',
                                           'native-NLL scope/cost if required',
                                           'bounded retries and shutdown',
                                           'new-study priority gate'],
            'interpretation': 'historical throughput does not certify a complete-stage budget; versioned resource amendment required if all-in total exceeds 12 GPUh'}
    return write_once(PRICE_OUT, body)


def run():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('X3 trace and native result preparation requires CPU Slurm')
    selected = selection()
    eligible = eligibility()
    priced = price(eligible, selected)
    print(json.dumps({'status': 'CPU_PREPARED_X3_GPU_HOLD', 'selection': str(SELECTION_OUT),
                      'eligibility': str(ELIGIBILITY_OUT), 'cost': str(PRICE_OUT),
                      'eligible': eligible['summary']['eligible'],
                      'two_sign_scenario_GPUh': priced['scenarios'][1]['historical_generation_plus_prefill_GPUh']}),
          flush=True)


if __name__ == '__main__':
    run()
