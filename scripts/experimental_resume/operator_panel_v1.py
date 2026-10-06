"""Independent, outcome-blind four-operator original-prompt panel contract."""
from __future__ import annotations

import math
from pathlib import Path

import utility_scout_v1 as U
from utility_production_v1 import require, save

DOC = U.REPO / 'report/experimental-resume-v1'
SCRIPTS = U.REPO / 'scripts/experimental_resume'
ROOT = U.STAGE / 'runs/routing-control-v1/operator-panel-v1'
PLAN = DOC / 'OPERATOR_PANEL_PLAN_v1.json'
ARMS = ('native', 'bias', 'force', 'reweight')
ENDPOINTS = ('operational_correct', 'reasoning_tokens', 'tokens')
COHORTS = (12, 24, 48, 96)
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit')
CEILING = 6000.
GPU_BILLING_CORES = 32
CPU_BILLING_CORES = 8  # 32 GiB on the 128-core/512-GiB viz node; conservative.
THINK_END_ID = 248069
# Williams square: each arm occupies every position and each ordered adjacency.
ORDERS = (('native', 'bias', 'reweight', 'force'),
          ('bias', 'force', 'native', 'reweight'),
          ('force', 'reweight', 'bias', 'native'),
          ('reweight', 'native', 'force', 'bias'))


def policies(design):
    require(design['schema'] == 'overnight-routing-design-v2' and
            design['transitions'] == list(TRANSITIONS), 'fresh design differs')
    actions = {a['name']: a for a in design['actions']}
    result = {}
    for operator in ARMS[1:]:
        selected = {}
        for transition in TRANSITIONS:
            arm = next(a for a in design['arms_by_transition'][transition] if a['name'] == operator)
            names = arm['policies'][transition]
            require(arm['role'] == 'target' and arm['slots'] == [0] and len(names) == 1,
                    'registered operator is not a single target pulse')
            action = actions[names[0]]
            require(action['kind'] == operator and action['transition'] == transition and
                    action['sign'] == 1 and action['magnitude'] == (0. if operator == 'force' else 1.),
                    'registered operator dose differs')
            selected[transition] = {'arm': operator, 'slots': [0], 'pulse_width': 256,
                                   'action_names': names, 'actions': [action]}
        result[operator] = selected
    for transition in TRANSITIONS:
        require(len({U.digest(result[op][transition]['actions'][0]['experts']) for op in ARMS[1:]}) == 1,
                'operators have different expert targets')
    return result


def build_plan(utility, design):
    U.validate_plan(utility)
    source = {'utility_plan_sha256': utility['sha256'], 'fresh_design_sha256': design['sha256']}
    native = {(r['family'], r['seed']): r for r in utility['assignments'] if r['arm'] == 'native'}
    rows = []
    for index, family in enumerate(utility['family_order']):
        seeds = (0, 1) if index % 2 == 0 else (1, 0)
        for seed in seeds:
            order = ORDERS[(2 * index + seed) % 4]
            for position, arm in enumerate(order):
                original = native[family, seed]
                uid = 'operator-panel-v1|' + U.digest([source, family, seed, arm])[:32]
                rows.append({**original, 'uid': uid, 'arm': arm, 'execution_position': position,
                             'screening_uid': 'operator-prefix-v1|' + U.digest([source, family, seed])[:32]})
    return U.seal({'schema': 'operator-panel-plan-v1', 'sources': source,
        'model': utility['model'], 'tokenizer_sha256': utility['tokenizer_sha256'],
        'original_sources': utility['sources'], 'family_order': utility['family_order'],
        'arms': list(ARMS), 'seeds': [0, 1], 'assignments': rows, 'assignment_count': 768,
        'maximum_tokens': 16384, 'policies': policies(design), 'candidate_cohorts': list(COHORTS),
        'pilot_family_order': utility['family_order'][:2],
        'order_rule': 'Williams square by (2*family_index+seed)%4; alternate seed-block order by family parity.',
        'screening_seed_rule': 'Frozen side-request digest with common family/seed screening UID and exact emitted prefix; no arm enters reader seed.',
        'controller': 'Existing first accepted transition; one 256-token pulse; no re-entry or side-query cap; stop at reasoning closure.',
        'inference': {'endpoints': list(ENDPOINTS), 'operators': list(ARMS[1:]),
            'primary_comparisons': 9, 'bootstrap_replicates': 50000, 'bootstrap_seed': 20261005,
            'alpha': .05, 'method': 'Paired family bootstrap; simultaneous max absolute standardized mean deviation using original standard errors across all nine effects.',
            'estimator': 'Within-seed operator minus native; mean over two seeds then equal mean over families.',
            'unknowns': 'Identification bounds with accuracy in [0,1] and length in [0,16384]. No complete joint intervals with missing endpoints.',
            'claim_limit': 'Exploratory effects and uncertainty; no equivalence or accuracy-retention margin.'},
        'budget': {'additional_billing_core_hour_ceiling': CEILING,
            'pilot_jobs': 2, 'pilot_gpus': 4, 'pilot_cpus': 32, 'pilot_memory_GiB': 240,
            'pilot_wall_seconds': 28800, 'cpu_cpus': 2, 'cpu_memory_GiB': 32, 'cpu_wall_seconds': 3600,
            'cpu_reserve_billing_core_hours': 128., 'stress_factor': 2.,
            'recovery_waves': 1, 'recovery_fraction': .25,
            'preserved_existing_utility_reserve_billing_core_hours': 17984.},
        'selection': 'Largest of 12/24/48/96 first frozen families fitting measured all-allocation generation, reserved grading, CPU and one recovery wave. Select before strict scoring or inspecting accuracy. Import all sixteen pilot cells exactly.'})


def validate_plan(plan):
    expected = build_plan(U.sealed(DOC / 'UTILITY_SCOUT_PLAN_v1.json'),
                          U.sealed(DOC / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'))
    require(plan == expected, 'operator panel plan or factorial changed')


def assignments(plan, families):
    require(families == plan['family_order'][:len(families)] and len(set(families)) == len(families),
            'cohort must be first frozen families')
    rows = [a for a in plan['assignments'] if a['family'] in families]
    require(len(rows) == 8 * len(families) and len({r['uid'] for r in rows}) == len(rows),
            'incomplete eight-cell factorial')
    return rows


def generation_sources():
    from utility_source_contract_v2 import code_files
    paths = ('operator_panel_v1.py', 'operator_panel_backend_v1.py',
             'run_operator_panel_v1.py', 'run_operator_panel_v1.sbatch')
    return {**code_files(), **{str(SCRIPTS / p): U.file_sha(SCRIPTS / p) for p in paths}}


def validate_manifest(manifest):
    plan = U.sealed(PLAN); validate_plan(plan)
    require(manifest['schema'] == 'operator-panel-manifest-v1' and
            manifest['plan_sha256'] == plan['sha256'] and manifest['generation_sources'] == generation_sources() and
            [r['assignment'] for r in manifest['rows']] == assignments(plan, manifest['family_order']) and
            manifest['policies'] == plan['policies'], 'panel execution binding changed')
    qual = U.sealed(Path(manifest['qualification_path']))
    binding = U.sealed(Path(manifest['qualification_path']).parent / 'BINDING.json')
    require(qual['status'] == 'PASS_ENGINEERING' and qual['binding_sha256'] == binding['sha256'] and
            qual['sha256'] == manifest['qualification_sha256'] and
            all(U.file_sha(Path(p)) == h for p, h in binding['code_files'].items()),
            'original-prompt engineering qualification changed')
    for row in manifest['rows']:
        a = row['assignment']
        require(U.digest(row['original_prompt_ids']) == a['prompt_token_ids_sha256'] and
                len(row['original_prompt_ids']) == a['prompt_tokens'] and set(row) ==
                {'assignment', 'original_prompt_ids', 'problem'}, 'prompt or generator information boundary changed')
    return plan


def make_manifest(plan, count, qualification, prompts, questions):
    families = plan['family_order'][:count]
    require(U.file_sha(U.PROMPTS) == plan['original_sources']['prompt_table_file_sha256'] and
            prompts['sha256'] == plan['original_sources']['prompt_table_sha256'], 'original prompt source changed')
    rows = [{'assignment': a, 'original_prompt_ids': prompts['questions'][a['question']]['prompt_token_ids'],
             'problem': questions['questions'][a['question']]['problem']} for a in assignments(plan, families)]
    return {'schema': 'operator-panel-manifest-v1', 'plan_sha256': plan['sha256'],
            'family_order': families, 'rows': rows, 'policies': plan['policies'],
            'qualification_path': str(qualification), 'qualification_sha256': U.sealed(qualification)['sha256'],
            'generation_sources': generation_sources(), 'maximum_tokens': 16384}


def measured_profile(receipts, loads, costs, allocations):
    """No gold, strict score, verdict or correctness field enters pricing."""
    reasons, rows = [], []
    if len(receipts) != 16 or len(loads) != 2 or len(costs) != 2 or len(allocations) != 2:
        reasons.append('two complete sixteen-cell pilots and load/shutdown/allocation records required')
    for receipt in receipts:
        if receipt['status'] != 'COMMITTED_GENERATION':
            reasons.append('pilot has a committed error'); continue
        result = receipt['result']; ctl = result['controller']
        n = len(ctl['state']['completion_token_ids']); total = result['generation_wall_seconds']
        side = ctl['side_elapsed_seconds']
        require(n > 0 and all(math.isfinite(v) and v >= 0 for v in (total, side)) and side <= total + 1,
                'invalid runtime measurement')
        rows.append({'uid': receipt['assignment']['uid'], 'arm': receipt['assignment']['arm'],
                     'tokens': n, 'wall_seconds': total, 'side_seconds': side,
                     'side_queries': len(ctl['queries']), 'generator_seconds_per_token': max(0., total - side) / n})
    for arm in ARMS:
        if sum(r['arm'] == arm for r in rows) != 4:
            reasons.append('pilot arm coverage incomplete')
    if not any(r['side_queries'] for r in rows):
        reasons.append('no measured side-reader work; pricing holds without changing policy')
    if reasons:
        return {'status': 'HOLD_MEASURED_PROFILE', 'reasons': sorted(set(reasons)), 'rows': rows}
    overhead = []
    for allocation, cost in zip(allocations, costs, strict=True):
        overhead.append(max(0., allocation['elapsed_seconds'] - cost['allocated_driver_wall_seconds']))
    return {'status': 'MEASURED_RUNTIME_ONLY', 'rows': rows,
            'generator_seconds_per_token': max(r['generator_seconds_per_token'] for r in rows),
            'operator_side_seconds': {op: max(r['side_seconds'] for r in rows if r['arm'] == op) for op in ARMS[1:]},
            'load_seconds': max(r['load_wall_seconds'] for r in loads),
            'allocation_overhead_seconds': max(overhead),
            'pilot_actual_billing_core_hours': sum(r['billing_core_hours'] for r in allocations),
            'interpretation': '2x stressed observed per-token generator and full observed per-request side work. Empirical forecast; no runtime guarantee or query bound.'}


def choose_cohort(plan, profile, grading_prices, spent_cpu=0.):
    require(profile['status'] == 'MEASURED_RUNTIME_ONLY', 'complete runtime pricing is required')
    b = plan['budget']; candidates = []
    family_seconds = b['stress_factor'] * (8 * 16384 * profile['generator_seconds_per_token'] +
                                          2 * sum(profile['operator_side_seconds'].values()))
    setup = b['stress_factor'] * (profile['load_seconds'] + profile['allocation_overhead_seconds']) + 600
    for count in COHORTS:
        capacity = math.floor((28800 - setup) / family_seconds) if family_seconds > 0 else 0
        capacity = max(0, min(8, capacity))
        initial = math.ceil((count - 2) / capacity) if capacity else 0
        wall = math.ceil((setup + capacity * family_seconds) / 60) * 60 if capacity else None
        recovery = math.ceil(b['recovery_fraction'] * initial)
        generation = (initial + recovery) * GPU_BILLING_CORES * wall / 3600 if wall else math.inf
        grade = grading_prices[count]['requested_allocation_GPU_hour_ceiling'] * 8
        total = profile['pilot_actual_billing_core_hours'] + generation + grade + b['cpu_reserve_billing_core_hours'] + spent_cpu
        candidates.append({'families': count, 'families_per_shard': capacity, 'initial_shards': initial,
            'recovery_shards': recovery, 'requested_wall_seconds': wall,
            'generation_with_recovery_billing_core_hours': generation if math.isfinite(generation) else None,
            'grading_reserve_billing_core_hours': grade, 'additional_total_billing_core_hours': total if math.isfinite(total) else None,
            'fits': capacity > 0 and total <= CEILING})
    chosen = next((r for r in reversed(candidates) if r['fits']), None)
    return {'status': 'PRICED_COHORT' if chosen else 'HOLD_NO_COMPLETE_COHORT_FITS', 'selected': chosen,
            'candidates': candidates, 'family_forecast_seconds': family_seconds, 'setup_forecast_seconds': setup,
            'selection_uses_accuracy': False, 'profile': profile}
