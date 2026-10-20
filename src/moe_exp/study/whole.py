"""Whole-reasoning experiment on the reduced driver's continuous engine.

The frozen manifest owns family order, original prompts, doses and seeds. Plain
receipts bind fresh stores and report canary/production gates; no historical
continuation is reused. Run through scripts/study/driver.sh with STUDY_MODULE
set to moe_exp.study.whole and RELEASE_ROOT set to a materialized commit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import CLASSES, REPO, check_files, digest, load, locked, publish, require, verified
from . import driver
from moe_exp.final_routing.selection import select_pairs

MODELS = ('gpt', 'qwen36')
SUPPRESS = {'gpt': ('Read', 'Analyze', 'Plan', 'Explore'), 'qwen36': ('Analyze', 'Plan', 'Implement')}
DOSES = {
    'gpt': {'main': {'promote': (10, 50), 'suppress': (0, 20)},
            'half': {'promote': (5, 25), 'suppress': (0, 10)}},
    'qwen36': {'main': {'promote': (15, 0), 'suppress': (0, 500)},
               'half': {'promote': (8, 0), 'suppress': (0, 250)}},
}
ROOT = Path(os.environ.get('WHOLE_ROOT', REPO/'report/whole-reasoning-20261020'))
# Main canaries and all nine production allocations reserve 71.5 GPU-hours.
GPU_CEILING = 72.
SHARDS = {'gpt': 2, 'qwen36': 7}
RUN_MINUTES = {'gpt': 165, 'qwen36': 240}
CANARY_MINUTES = {'gpt': 45, 'qwen36': 90}
CUTOFF = 1792726200.  # 2026-10-23 03:30 UTC (05:30 Europe/Rome).
# CPU allocation time is charged across monitor, extraction and analysis. Leave
# half a core-hour for concurrent shutdown and native-accounting latency.
CPU_WORK_LIMIT = 15.5
CPU_TAIL_RESERVE = 8.


def verify_release() -> dict:
    """Verify archived code and pinned study inputs, without touching live source.

    Returns:
        The unchanged source release receipt after physical file checks.
    """
    release = verified(ROOT/'RELEASE.json')
    require(Path(release['release_root']).resolve() == REPO, 'whole stage must run from its materialized release')
    require((REPO/'RELEASE_COMMIT').read_text().strip() == release['commit'], 'release commit marker differs')
    check_files(release['code_files'])
    check_files(release['inputs'])
    return release


def work_remaining_bytes() -> int:
    """Read the verified Lustre project quota, preserving the existing WORK reserve."""
    result = subprocess.run(['lfs', 'quota', '-p', '20149688', '/leonardo_work'],
                            capture_output=True, text=True, check=True)
    fields = next(line.split() for line in result.stdout.splitlines() if line.strip().startswith('/leonardo_work '))
    used, soft, hard = (int(value.rstrip('*')) for value in fields[1:4])
    limits = [value for value in (soft, hard) if value]
    require(limits, 'native WORK project quota limit is unavailable')
    return (min(limits)-used)*1024


def _layer_actions(roles: dict, profile: dict) -> list[dict]:
    """Collect disjoint roles once per layer while preserving native top-k capacity."""
    result = []
    for layer in profile['layers']:
        active = sorted(p['expert'] for p in roles['activate'] if p['layer'] == layer)
        inactive = sorted(p['expert'] for p in roles['deactivate'] if p['layer'] == layer)
        require(not set(active) & set(inactive), 'whole target/control roles overlap')
        require(len(active) < profile['native_k'], 'whole action fills native top-k')
        require(profile['experts']-len(inactive) >= profile['native_k'], 'whole exclusion exhausts experts')
        if active or inactive:
            result.append({'layer': layer, 'activate': active, 'deactivate': inactive,
                           'operator': 'score_override_dual', 'epsilon': .01, 'pulse_tokens': 0})
    return result


def cell_actions(nomination: dict, profile: dict, model: str, dose: str, operation: str) -> dict:
    """Build one cell and its role/layer/frequency matched random control."""
    from moe_exp.final_routing.selection import layer_matched_controls
    layers = profile['layers']
    act_count, deact_count = DOSES[model][dose][operation]
    positive, negative = nomination['candidates']['positive'], nomination['candidates']['negative']
    original_activate, _ = select_pairs(positive, act_count, layers, 'activate')
    deact_candidates = negative if operation == 'promote' else positive
    original_deactivate, _ = select_pairs(deact_candidates, deact_count, layers, 'deactivate')
    opposite_act, _ = select_pairs(negative, act_count, layers, 'activate')
    opposite_deact, _ = select_pairs(positive if operation == 'promote' else negative,
                                     deact_count, layers, 'deactivate')
    opposite = {(p['layer'], p['expert']) for p in opposite_act+opposite_deact}
    activate, act_availability = select_pairs(positive, act_count, layers, 'activate',
                                             frequency=nomination['frequency'], excluded=opposite)
    fixed = opposite | {(p['layer'], p['expert']) for p in activate}
    deactivate, deact_availability = select_pairs(deact_candidates, deact_count, layers, 'deactivate',
        frequency=nomination['frequency'], excluded=fixed, fixed_targets=activate)
    excluded = fixed | {(p['layer'], p['expert']) for p in deactivate}
    qualified = act_availability['qualified'] and deact_availability['qualified']
    controls = (layer_matched_controls(activate, deactivate, nomination['frequency'], layers, excluded,
                [model, nomination['fold'], nomination['class'], dose, operation]) if qualified else
                {'activate': [], 'deactivate': []})

    original_rd = sum(abs(p['delta']) for p in original_deactivate)
    retained_rd = sum(abs(p['delta']) for p in deactivate)
    native_frequency = {}
    for arm, roles in (('target', {'activate': activate, 'deactivate': deactivate}), ('random', controls)):
        pairs = roles['activate']+roles['deactivate']
        native_frequency[arm] = (float(np.mean([nomination['frequency'][layers.index(p['layer'])][p['expert']]
                                               for p in pairs])) if pairs else None)
    return {
        'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation,
        'dose': dose, 'qualified': qualified,
        'activate': act_availability, 'deactivate': deact_availability,
        'target': _layer_actions({'activate': activate, 'deactivate': deactivate}, profile),
        'random': _layer_actions(controls, profile),
        'excluded_pairs': sorted(map(list, excluded)),
        'selection_rule': 'control-feasible absolute-RD greedy; exact layer/quintile/role matching',
        'retained_rd_fraction': retained_rd/original_rd if original_rd else 1.,
        'original_deactivation_rd': original_rd, 'retained_deactivation_rd': retained_rd,
        'activation_changed': activate != original_activate,
        'mean_native_frequency': native_frequency,
        'replaced_deactivation_pairs': len({(p['layer'], p['expert']) for p in original_deactivate} -
                                           {(p['layer'], p['expert']) for p in deactivate}),
    }


def build_manifest(model: str, dose: str, nominations: dict, b: dict, natives: list[dict],
                   inputs: dict, prompt_scope: bool) -> dict:
    """Freeze all planned rows, including unavailable cells, in complete family blocks.

    Args:
        model, dose: Model identity and declared main/half fluency operating point.
        nominations: Discovery-only global risk-difference receipt.
        b: Original study profile binding.
        natives: Original main-design native assignments.
        inputs: Family to original prompt token list.
        prompt_scope: Whether prompt qualification passed before the fallback cutoff.

    Returns:
        Manifest with planned row counts, actual cell counts and unique UIDs.
    """
    from moe_exp.final_routing.runtime import remaining
    profile = b['models'][model]
    cells = [cell_actions(n, profile, model, dose, operation) for n in nominations['records']
             for operation in ('promote', 'suppress')
             if operation == 'promote' or n['class'] in SUPPRESS[model]]
    by_cell = {(c['fold'], c['class'], c['operation']): c for c in cells}
    require(all(c['qualified'] for c in cells if c['operation'] == 'suppress' or
                c['class'] in ('Verify', 'Monitor')), 'every efficiency-directed cell must be nominated')
    rows = []
    for native in sorted(natives, key=lambda row: row['family']):
        prompt = inputs[native['family']]
        allowance = remaining(profile, prompt, [])
        base = {key: native[key] for key in ('model', 'family', 'benchmark', 'fold', 'seed', 'profile_sha256')}
        base.update(source={**native['source'], 'prefix_tokens': 0}, original_prompt=True,
                    prompt_sha256=digest(prompt), whole=True, prompt_scope=prompt_scope,
                    pulse_tokens=0, horizon=allowance, total_allowance=allowance, prefix_status='enrolled',
                    capture_layers=profile['layers'], action=None, dose=dose, depth_third=None)
        for arm in ('native', 'native_repeat'):
            row = {**base, 'arm': arm, 'class': None, 'operation': 'native', 'actions': [],
                   'nomination_status': 'nominated'}
            if arm == 'native_repeat':
                row['seed'] = int(digest([native['seed'], 'whole-native-repeat'])[:8], 16)
                require(row['seed'] != native['seed'], 'native repeat seed collision')
            rows.append(row)
        for operation in ('promote', 'suppress'):
            classes = CLASSES if operation == 'promote' else SUPPRESS[model]
            for class_name in classes:
                cell = by_cell[native['fold'], class_name, operation]
                for role in ('target', 'random'):
                    rows.append({**base, 'arm': role+'_'+operation, 'class': class_name,
                                 'operation': operation, 'actions': cell[role],
                                 'steered_native_frequency': cell.get('mean_native_frequency', {}).get(role),
                                 'nomination_status': 'nominated' if cell['qualified'] else 'dropped_availability'})
    # Audit flags are selected without reading a whole-arm outcome, before execution.
    audit_keys = sorted(range(len(rows)), key=lambda i: digest(['whole-audit', model, dose, rows[i]]))[:32]
    for index, row in enumerate(rows):
        row['audit_capture'] = index in audit_keys
        row['uid'] = digest(['whole-v1', nominations.get('release_commit', 'development'), nominations['sha256'], row])
    expected = 3096 if model == 'gpt' else 2838
    require(len(rows) == expected and len({r['uid'] for r in rows}) == expected, 'whole row count/UID differs')
    return {'model': model, 'dose': dose, 'scope': 'prompt-and-generated-reasoning' if prompt_scope else
            'generated-reasoning-only; not a SteerMoE whole-token replication',
            'nominations_sha256': nominations['sha256'], 'cells': cells, 'records': rows,
            'planned': expected, 'executable': sum(r['nomination_status'] == 'nominated' for r in rows),
            'audit_units': 32, 'outcomes_used_for_selection': False}


def prepare(models=MODELS):
    """Derive and publish main/half manifests and native-only planning evidence on CPUs."""
    require(os.environ.get('SLURM_JOB_ID'), 'whole discovery and planning require an allocated CPU job')
    verify_release()
    from moe_exp.final_routing.importer import derive_whole
    from .runtime import prompt_prefix
    b = driver.binding()
    results = {}
    for model in models:
        folder = ROOT/model
        nomination_path = folder/'WHOLE_NOMINATIONS.json'
        nominations = verified(nomination_path) if nomination_path.exists() else derive_whole(b, model, folder)
        natives = [r for r in verified(driver.DRIVER/model/'ASSIGNMENTS.json')['records'] if r['arm'] == 'native']
        require(len(natives) == 129, 'whole original native family roster differs')
        inputs = {}
        for row in natives:
            original = {**row, 'original_prompt': True, 'source': {**row['source'], 'prefix_tokens': 0}}
            prompt, prefix, _ = prompt_prefix(b, model, original, None)
            require(not prefix, 'original prompt row retained a prefix')
            inputs[row['family']] = prompt
        scope = os.environ.get('WHOLE_PROMPT_SCOPE', '1') == '1'
        for dose in ('main', 'half'):
            value = publish(folder/('WHOLE.json' if dose == 'main' else 'WHOLE_HALF.json'),
                            build_manifest(model, dose, nominations, b, natives, inputs, scope))
            results[model+':'+dose] = {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}
    from .whole_effects import planning
    try:
        planning_result = planning(models)
    except Exception as error:
        planning_result = {'transport_status': 'SUSPECTED', 'available': False,
                           'error': f'{type(error).__name__}: {error}', 'gates_science': False}
    publish(ROOT/'SD_MDE.json', planning_result)
    return results


def manifest(model: str, dose: str | None = None) -> dict:
    """Read the selected manifest; selection is fixed by the passing canary."""
    if dose is None:
        selected = ROOT/model/'SELECTED.json'
        dose = verified(selected)['dose'] if selected.exists() else 'main'
    return verified(ROOT/model/('WHOLE.json' if dose == 'main' else 'WHOLE_HALF.json'))


def bound_store(model: str, dose: str, canary: bool = False) -> driver.Store:
    """Reject any nonempty unbound store before the engine is loaded."""
    value = manifest(model, dose)
    release = verified(ROOT/'RELEASE.json')
    require(Path(release['release_root']).resolve() == REPO, 'whole code is outside its immutable materialized release')
    folder = ROOT/'stores'/model/dose/('canary' if canary else 'units')
    identity = {'manifest_sha256': value['sha256'], 'release_sha256': release['sha256'], 'canary': canary}
    if (folder/'EXECUTION.json').exists():
        require({k: v for k, v in verified(folder/'EXECUTION.json').items() if k != 'sha256'} == identity,
                'whole store belongs to a different release')
    else:
        require(not folder.exists() or not any(folder.iterdir()), 'nonempty unbound whole store')
        publish(folder/'EXECUTION.json', identity)
    return driver.Store(folder)


def canary_rows(value: dict) -> list[dict]:
    """Use eight outcome-independent families and 32 separate target/random units."""
    native = [row for row in value['records'] if row['arm'] == 'native']
    ordered = sorted(native, key=lambda row: digest(['whole-canary-families', row['family']]))
    families = []
    for benchmark in sorted({row['benchmark'] for row in native}):
        families.append(next(row['family'] for row in ordered if row['benchmark'] == benchmark))
    families += [row['family'] for row in ordered if row['family'] not in families][:8-len(families)]
    records = []
    model = value['model']
    chosen_class = SUPPRESS[model][0]
    for family in families:
        members = [r for r in value['records'] if r['family'] == family]
        for row in members:
            keep = row['arm'] in ('native', 'native_repeat') or (
                row['operation'] == 'promote' and row['class'] == 'Verify') or (
                row['operation'] == 'suppress' and row['class'] == chosen_class)
            if keep and row['nomination_status'] == 'nominated':
                canary = {**row, 'canary': True, 'audit_capture': True}
                canary['uid'] = digest(['whole-canary', value['sha256'], canary])
                records.append(canary)
    require(sum(r['operation'] != 'native' for r in records) == 32, 'canary cell availability differs')
    return records


def repetition_rate(tokens: list[int], width: int = 16) -> float:
    """Fraction of token n-gram positions whose n-gram has already appeared."""
    grams = [tuple(tokens[i:i+width]) for i in range(max(0, len(tokens)-width+1))]
    return 1-len(set(grams))/len(grams) if grams else 0.


def relative_fluency(observed: list[dict]) -> dict:
    """Compare completed steered units with same-family native and repeat units.

    Args:
        observed: Completed rows with token length, cap and repetition diagnostics.

    Returns:
        Relative gates and their denominators. Early gates use only families whose
        two native references completed; the canary requires all 48 rows.
    """
    native = {}
    for row in observed:
        if row['arm'] in ('native', 'native_repeat'):
            native.setdefault(row['family'], []).append(row)
    steered = [r for r in observed if r['operation'] != 'native' and len(native.get(r['family'], [])) == 2]
    references = [native[r['family']] for r in steered]
    cap_rate = float(np.mean([r['cap'] for r in steered])) if steered else None
    native_cap = float(np.mean([n['cap'] for pair in references for n in pair])) if steered else None
    repetition = float(np.mean([r['repetition'] for r in steered])) if steered else None
    native_repetition = float(np.mean([n['repetition'] for pair in references for n in pair])) if steered else None
    ratios = [math.log(max(1, r['tokens']))-np.mean([math.log(max(1, n['tokens'])) for n in pair])
              for r, pair in zip(steered, references)]
    ratio = math.exp(float(np.mean(ratios))) if ratios else None
    failures = []
    if cap_rate is not None and cap_rate-native_cap > .15:
        failures.append('cap')
    if ratio is not None and ratio > 1.40:
        failures.append('length')
    if repetition is not None and repetition-native_repetition > .10:
        failures.append('repetition')
    return {
        'steered_units': len(steered),
        'cap_count': sum(r['cap'] for r in steered),
        'cap_rate': cap_rate,
        'native_cap_rate': native_cap,
        'cap_rate_difference': cap_rate-native_cap if steered else None,
        'geometric_length_ratio_to_native': ratio,
        'repetition_16gram': repetition,
        'native_repetition_16gram': native_repetition,
        'repetition_16gram_difference': repetition-native_repetition if steered else None,
        'fluency_failures': failures,
        'reference': 'same-family native and native_repeat; both complete; weighted by compared steered units',
    }


def fluency(store, rows: list[dict], *, require_complete: bool = False) -> dict:
    """Evaluate the frozen fluency and same-state dose gates on completed units."""
    observed, doses, software = [], [], []
    for row in rows:
        if (store.state(row['uid']) or {}).get('status') != 'complete':
            continue
        raw = verified(store.folder(row['uid'])/'OUTPUT.json')
        capture = verified(store.folder(row['uid'])/'CAPTURE.json')
        observed.append({**row, 'tokens': len(raw['token_ids']), 'cap': raw['finish_reason'] == 'length',
                         'repetition': repetition_rate(raw['token_ids'])})
        if row['operation'] == 'native':
            continue
        by_layer = capture['stats']['dose']
        for action in row['actions']:
            dose = by_layer.get(str(action['layer']))
            if not dose or dose['active_positions'] == 0:
                software.append(row['uid']+':zero_active_positions')
                continue
            if dose['finite'] != dose['observations'] or dose.get('membership_failures', 0):
                software.append(row['uid']+':nonfinite_or_membership')
            if row['prompt_scope'] and dose['active_prompt_positions'] == 0:
                software.append(row['uid']+':zero_prompt_positions')
            if action['activate'] and dose['executed_activate_share'] <= dose['native_activate_share']:
                software.append(row['uid']+':activation_weight_not_increased')
            if action['deactivate'] and dose['executed_deactivate_weight'] != 0:
                software.append(row['uid']+':deactivated_weight_present')
            doses.append((row, dose))
    if require_complete and len(observed) != len(rows):
        software.append('incomplete_canary')
    relative = relative_fluency(observed)
    deactivation = [dose for row, dose in doses if row['operation'] == 'suppress']
    changed = (sum(d['changed_membership'] for d in deactivation)/sum(d['active_positions'] for d in deactivation)
               if deactivation else None)
    if changed is not None and changed < .05:
        software.append('deactivation_membership_below_5_percent')
    return {'passed': not software and not relative['fluency_failures'] and bool(relative['steered_units']),
            'completed': len(observed), **relative,
            'deactivation_changed_membership': changed, 'software_failures': software,
            'observed_steered_units': sum(r['operation'] != 'native' for r in observed)}


def admission_order(rows: list[dict]) -> list[dict]:
    """Order all primary and native rows before other promote rows, without barriers."""
    return sorted(rows, key=lambda r: (
        int(r['operation'] == 'promote' and r['class'] not in ('Verify', 'Monitor')),
        r['family'], r['arm'], r['class'] or '',
    ))


def run(model, shard, deadline, *, max_inflight=64, canary=False, dose=None):
    """Run one source-bound shard or canary on the existing hooked engine."""
    from .runtime import closure
    verify_release()
    value = manifest(model, dose)
    require(verified(ROOT/'CPU_VALIDATION.json')['passed'], 'allocated CPU validation did not pass')
    if not canary:
        selected = verified(ROOT/model/'SELECTED.json')
        require(selected['manifest_sha256'] == value['sha256'], 'production manifest lacks a passing canary')
        proof = verified(ROOT/model/('CANARY_HALF.json' if value['dose'] == 'half' else 'CANARY.json'))
        require(proof['gate']['passed'] and proof['job'] == selected['canary_job'] and
                proof['manifest_sha256'] == value['sha256'], 'production canary qualification differs')
    rows = canary_rows(value) if canary else driver.shard_rows(model, shard, 'whole')
    store = bound_store(model, value['dose'], canary)
    require(not any(store.state(r['uid']) for r in rows), 'whole execution is fresh only; preserve partials')
    executable = admission_order([r for r in rows if r['nomination_status'] == 'nominated'])
    b = driver.binding()
    llm, kwargs, load_seconds = driver.hooked_llm(b, model, max_inflight)
    try:
        contract = driver.contract_for(b, model, executable)
        reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract})
        require(len(reports) == 2, 'whole requires both tensor-parallel ranks')
        tokenizer = llm.get_tokenizer()
        engine = driver.Engine(llm, store, min(deadline, CUTOFF), hooked=True, max_inflight=max_inflight,
                               checkpoint_seconds=120, drain_seconds=120)
        build = driver.intervention_builder(b, model, tokenizer, closure(tokenizer, model))
        metrics = engine.run(executable, build, driver.intervention_finisher(store, b, model),
                             stop_execution=lambda: (ROOT/'STOP.json').exists() or (ROOT/model/'STOP.json').exists())
        validation = llm.collective_rpc('fs_validation')
        llm.collective_rpc('fs_end')
    finally:
        llm.llm_engine.engine_core.shutdown()
    result = {'model': model, 'dose': value['dose'], 'manifest_sha256': value['sha256'],
              'job': os.environ['SLURM_JOB_ID'], 'shard': list(shard), 'load_seconds': load_seconds,
              'metrics': metrics, 'native_validation': validation[0], **driver.rescue_completion(store, rows)}
    if canary:
        result['gate'] = fluency(store, rows, require_complete=True)
        publish(ROOT/model/('CANARY.json' if value['dose'] == 'main' else 'CANARY_HALF.json'), result)
        if result['gate']['passed']:
            publish(ROOT/model/'SELECTED.json', {'dose': value['dose'], 'manifest_sha256': value['sha256'],
                                                'canary_job': result['job']})
        else:
            raise SystemExit(3 if result['gate']['fluency_failures'] and not result['gate']['software_failures'] else 2)
    else:
        publish(ROOT/model/'jobs'/(result['job']+'.json'), result)
    return result


def native_accounting(jobs: list[dict]) -> dict:
    """Read exact native job states and allocation time, charging failures and startup."""
    ids = [job['id'] for job in jobs]
    if not ids:
        return {}
    query = subprocess.run(['sacct', '-nP', '-X', '-j', ','.join(ids),
                            '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],
                           capture_output=True, text=True, check=True)
    result = {}
    for line in query.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 8 or fields[0] not in ids:
            continue
        job_id, state, exit_code, elapsed, cpus, tres, start, end = fields[:8]
        elapsed_seconds = int(elapsed or 0)
        if state in ('RUNNING', 'COMPLETING') and start not in ('', 'None', 'Unknown'):
            from datetime import datetime
            elapsed_seconds = max(elapsed_seconds, int(time.time()-datetime.fromisoformat(start).timestamp()))
        result[job_id] = {'state': state.split()[0], 'exit_code': exit_code, 'elapsed': int(elapsed or 0),
                          'cpus': int(cpus or 0), 'tres': tres, 'start': start, 'end': end}
        result[job_id]['elapsed'] = elapsed_seconds
    return result


def submit_stage(command: str, *, model=None, dose=None, shard=None, dependency=None, held=False) -> dict:
    """Submit one bounded approved stage and read back its exact native resource request."""
    release = verified(ROOT/'RELEASE.json')
    gpu = command in ('canary', 'run')
    cpus = 16 if gpu else 1 if command == 'monitor' else 4 if command == 'validate' else 8
    memory = 120 if gpu else 2 if command == 'monitor' else 24 if command != 'analyze' else 32
    minutes = {
        'canary': CANARY_MINUTES.get(model),
        'run': RUN_MINUTES.get(model),
        'validate': 30,
        'monitor': 960,
        'extract': 90,
        'analyze': 120,
    }[command]
    args = [command]
    if model:
        args += ['--model', model]
    if dose:
        args += ['--dose', dose]
    if shard:
        args += ['--shard', f'{shard[0]}/{shard[1]}']
    if gpu:
        args += ['--max-inflight', '64']
    exports = {'RELEASE_ROOT': release['release_root'], 'STUDY_DATA_ROOT': str(driver.ROOT),
               'WHOLE_ROOT': str(ROOT), 'STUDY_MODULE': 'moe_exp.study.whole',
               'MOE_EXP_SCORING_PROCESSES': '8' if command == 'extract' else '2'}
    cmd = ['sbatch', '--parsable', '--account=iscrc_miosr', '--partition=boost_usr_prod', '--qos=normal',
           '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus}', f'--mem={memory}G', f'--time={minutes}',
           '--job-name=whole-'+command+('-'+model if model else '')+('-'+dose if dose else ''),
           '--chdir='+release['release_root'], '--output='+str(ROOT/'logs/%x-%j.out'),
           '--error='+str(ROOT/'logs/%x-%j.err'),
           '--export=ALL,'+','.join(key+'='+value for key, value in exports.items())]
    if gpu:
        cmd += ['--gres=gpu:2']
    if dependency:
        cmd += ['--dependency='+dependency]
    if held:
        cmd += ['--hold']
    cmd += [str(Path(release['release_root'])/'scripts/study/driver.sh'), *args]
    folder = ROOT/'submissions'/digest([command, model, dose, shard])
    if (folder/'ACCEPTED.json').exists():
        return verified(folder/'ACCEPTED.json')
    if (folder/'SUBMITTED.json').exists():
        submitted = verified(folder/'SUBMITTED.json')
        require(submitted['args'] == cmd, 'recorded whole submission arguments differ')
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split(';')[0]
        require(job_id.isdecimal(), 'sbatch did not return a job ID')
        submitted = publish(folder/'SUBMITTED.json', {
            'id': job_id, 'command': command, 'model': model, 'dose': dose, 'shard': shard,
            'gpus': 2 if gpu else 0, 'cpus': cpus, 'minutes': minutes, 'held': held,
            'sbatch': result.stdout.strip(), 'args': cmd, 'submitted': time.time(),
        })
    # Persist the ID before scheduler readback: a transient scontrol failure
    # retries this acceptance instead of submitting another allocation.
    job_id = submitted['id']
    readback = subprocess.run(['scontrol', 'show', 'job', '-o', job_id], capture_output=True, text=True, check=True).stdout
    require('UserId=lmolfett(' in readback and 'Account=iscrc_miosr ' in readback and
            'Partition=boost_usr_prod ' in readback and f'NumCPUs={cpus} ' in readback,
            'accepted whole job resource/account readback differs')
    require(not gpu or 'gres/gpu=2' in readback, 'accepted whole GPU count differs')
    require(f'TimeLimit={minutes//60:02d}:{minutes%60:02d}:00 ' in readback,
            'accepted whole walltime differs')
    return publish(folder/'ACCEPTED.json', {**submitted, 'readback': readback})


def all_jobs(schedule: dict) -> list[dict]:
    """Flatten the single plain schedule for native accounting."""
    jobs = [schedule['validation']]
    if schedule.get('monitor'):
        jobs.append(schedule['monitor'])
    for model in MODELS:
        jobs.extend(schedule['canaries'][model])
        jobs.extend(schedule['production'][model])
        if model in schedule['extraction']:
            jobs.append(schedule['extraction'][model])
    if schedule.get('analysis'):
        jobs.append(schedule['analysis'])
    return jobs


def terminal(state: dict) -> bool:
    """Recognize native terminal states, without interpreting pending dependencies as success."""
    return state.get('state') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED')


def cancel_jobs(jobs: list[dict], accounting: dict) -> None:
    """Cancel only this experiment's explicitly recorded nonterminal IDs."""
    ids = [job['id'] for job in jobs if not terminal(accounting.get(job['id'], {}))]
    if ids:
        subprocess.run(['scancel', *ids], check=True)


def production_progress(model: str) -> dict:
    """Count completed target/random pairs and complete families without loading tokenizers."""
    value = manifest(model)
    store = bound_store(model, value['dose'])
    rows = value['records']
    complete = [row for row in rows if (store.state(row['uid']) or {}).get('status') == 'complete']
    by_pair = Counter((r['family'], r['operation'], r['class']) for r in complete if r['operation'] != 'native')
    planned_pairs = sum(r['arm'].startswith('target_') for r in rows)
    executable_pairs = sum(r['arm'].startswith('target_') and r['nomination_status'] == 'nominated' for r in rows)
    by_family = Counter(r['family'] for r in complete)
    planned_family = Counter(r['family'] for r in rows)
    return {'completed': len(complete), 'planned': len(rows), 'pairs': sum(n == 2 for n in by_pair.values()),
            'planned_pairs': planned_pairs, 'executable_pairs': executable_pairs,
            'complete_families': sum(by_family[f] == count for f, count in planned_family.items())}


def reserved_gpu_hours(schedule: dict, accounting: dict, stopped: set) -> float:
    """Charge finished allocations and reserve every outstanding planned GPU maximum."""
    jobs = [job for kind in ('canaries', 'production') for rows in schedule[kind].values() for job in rows]
    reserved = verified(ROOT/'RELEASE.json').get('prior_usage', {}).get('new_gpu_hours', 0.)
    for job in jobs:
        state = accounting.get(job['id'], {})
        reserved += state['elapsed']*job['gpus']/3600 if terminal(state) else job['gpus']*job['minutes']/60
    for model in MODELS:
        if model not in stopped:
            missing = SHARDS[model]-len(schedule['production'].get(model, []))
            reserved += missing*2*RUN_MINUTES[model]/60
    return reserved


def advance_canary(schedule: dict, model: str, accounting: dict, stopped: set) -> None:
    """Advance only passing canaries, or the one declared fluency fallback."""
    attempts = schedule['canaries'][model]
    attempt = attempts[-1]
    state = accounting.get(attempt['id'], {})
    if (ROOT/'STOP.json').exists() or (ROOT/model/'STOP.json').exists() or (ROOT/model/'PRE_PRODUCTION_STOP.json').exists():
        stopped.add(model)
    if model in stopped:
        return
    path = ROOT/model/('CANARY.json' if attempt['dose'] == 'main' else 'CANARY_HALF.json')
    gate = verified(path)['gate'] if path.exists() else None
    current_canary = attempt['id'] == os.environ.get('SLURM_JOB_ID')
    if gate and gate['passed'] and (state.get('state') == 'COMPLETED' or current_canary):
        bound_store(model, attempt['dose'])
        count = SHARDS[model]
        for index in range(len(schedule['production'][model]), count):
            job = submit_stage('run', model=model, dose=attempt['dose'], shard=[index, count],
                               dependency='afterok:'+attempt['id'])
            schedule['production'][model].append(job)
            driver.checkpoint(ROOT/'JOBS.json', schedule)
        # Queue dependent CPU work in the same step as these shards. Their
        # execution no longer needs a live monitor once this step returns.
        advance_extraction(schedule, accounting, stopped)
    elif not terminal(state):
        return
    elif gate and gate['fluency_failures'] and not gate['software_failures'] and attempt['dose'] == 'main':
        fallback_maximum = 2*CANARY_MINUTES[model]/60
        reserved = reserved_gpu_hours(schedule, accounting, stopped)+fallback_maximum
        if reserved > GPU_CEILING:
            stopped.add(model)
            publish(ROOT/model/'PRE_PRODUCTION_STOP.json', {
                'reason': 'half-dose canary cannot fit its full walltime plus the remaining shard maxima',
                'reserved_gpu_hours': reserved,
            })
            return
        attempts.append(submit_stage('canary', model=model, dose='half', dependency='afterany:'+attempt['id']))
        driver.checkpoint(ROOT/'JOBS.json', schedule)
        publish(ROOT/model/'SELECTED.json', {'dose': 'half', 'manifest_sha256': manifest(model, 'half')['sha256'],
                                            'canary_job': attempts[-1]['id']})
    else:
        stopped.add(model)
        publish(ROOT/model/'PRE_PRODUCTION_STOP.json', {'job': attempt['id'], 'state': state,
                'gate': gate, 'reason': 'canary_failure; no efficacy-driven tuning or software retry'})


def check_projection(schedule: dict, model: str, accounting: dict, progress: dict, stopped: set,
                     gpu_hours: float) -> None:
    """Project at the first completed quarter, with the other model's quarter or frozen margin."""
    production = schedule['production'][model]
    model_usage = sum(accounting.get(job['id'], {}).get('elapsed', 0)*job['gpus']/3600 for job in production)
    fraction = progress['pairs']/progress['executable_pairs']
    projected_model = model_usage/fraction
    other = next(m for m in MODELS if m != model)
    other_jobs = schedule['production'][other]
    other_progress = production_progress(other) if other_jobs else None
    other_usage = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600 for j in other_jobs)
    if other_progress and other_progress['pairs'] >= .25*other_progress['executable_pairs']:
        other_projection = other_usage/(other_progress['pairs']/other_progress['executable_pairs'])
    else:
        planned_maximum = sum(j['gpus']*j['minutes']/60 for j in other_jobs)
        planned_maximum += (SHARDS[other]-len(other_jobs))*2*RUN_MINUTES[other]/60
        other_projection = other_usage if other in stopped else planned_maximum
    canary_usage = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600
                       for m in MODELS for j in schedule['canaries'][m])
    prior_gpu_hours = verified(ROOT/'RELEASE.json').get('prior_usage', {}).get('new_gpu_hours', 0.)
    projection = prior_gpu_hours+projected_model+max(other_usage, other_projection)+canary_usage
    publish(ROOT/model/'PROJECTION.json', {'progress': progress, 'gpu_hours': gpu_hours,
            'model_projected_gpu_hours': projected_model, 'total_projected_gpu_hours': projection})
    if projection > GPU_CEILING:
        publish(ROOT/'STOP.json', {'reason': 'quarter_pair_projection', 'projection': projection})
        cancel_jobs([j for m in MODELS for j in schedule['production'][m]], accounting)
        stopped.update(MODELS)


def check_production(schedule: dict, model: str, accounting: dict, checked: set, projected: set,
                     stopped: set, gpu_hours: float) -> None:
    """Evaluate the fixed early fluency gate and the quarter-pair projection."""
    production = schedule['production'][model]
    if not production or model in stopped:
        return
    progress = production_progress(model)
    elapsed = max((accounting.get(job['id'], {}).get('elapsed', 0) for job in production), default=0)
    if model not in checked and elapsed >= (2700 if model == 'gpt' else 4500):
        value = manifest(model)
        gate = fluency(bound_store(model, value['dose']), value['records'])
        if gate['steered_units']:
            publish(ROOT/model/'EARLY_GATE.json', {'gate': gate, 'elapsed': elapsed, 'progress': progress})
            checked.add(model)
            if not gate['passed']:
                publish(ROOT/model/'STOP.json', {'reason': 'early_fluency_gate', 'gate': gate, 'elapsed': elapsed})
                cancel_jobs(production, accounting)
                stopped.add(model)
        else:
            driver.checkpoint(ROOT/model/'EARLY_GATE_PENDING.json', {'elapsed': elapsed, 'progress': progress})
    if model in stopped:
        return
    if model not in projected and progress['pairs'] >= .25*progress['executable_pairs']:
        check_projection(schedule, model, accounting, progress, stopped, gpu_hours)
        projected.add(model)


def advance_extraction(schedule: dict, accounting: dict, stopped: set) -> None:
    """Queue afterany extraction with each complete shard graph, then analysis."""
    schedule.setdefault('extraction', {})
    for model in MODELS:
        production = schedule['production'][model]
        ready = len(production) == SHARDS[model] or model in stopped
        if ready and model not in schedule['extraction']:
            dependencies = production or schedule['canaries'][model]
            schedule['extraction'][model] = submit_stage('extract', model=model,
                dependency='afterany:'+':'.join(j['id'] for j in dependencies))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
    if len(schedule['extraction']) == 2 and not schedule.get('analysis'):
        schedule['analysis'] = submit_stage('analyze', dependency='afterany:'+':'.join(
            j['id'] for j in schedule['extraction'].values()))
        driver.checkpoint(ROOT/'JOBS.json', schedule)


def accounting_snapshot(jobs: list[dict]) -> tuple[dict, float]:
    """Charge native allocation time and reject stale GPU accounting before progression."""
    accounting = native_accounting(jobs)
    require(all(j['id'] in accounting for j in jobs if j['gpus'] and time.time()-j['submitted'] > 120),
            'native GPU accounting unavailable beyond the bounded registration delay')
    prior = verified(ROOT/'RELEASE.json').get('prior_usage', {})
    gpu_hours = prior.get('new_gpu_hours', 0.)+sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600 for j in jobs)
    cpu_hours = prior.get('new_cpu_core_hours', 0.)+sum(
        accounting.get(j['id'], {}).get('elapsed', 0)*j['cpus']/3600 for j in jobs if not j['gpus'])
    driver.checkpoint(ROOT/'ACCOUNTING.json', {'native': accounting, 'new_gpu_hours': gpu_hours,
                      'retained_r3_gpu_hours': .3461111111111111, 'new_cpu_core_hours': cpu_hours,
                      'gpu_ceiling': GPU_CEILING, 'cpu_ceiling': 16, 'time': time.time(),
                      'work_remaining_bytes': work_remaining_bytes()})
    return accounting, gpu_hours


def queue_after_canary(model: str) -> None:
    """Queue dependent work from the passing canary even if the monitor has exited."""
    with locked(ROOT, 'schedule-progression'):
        schedule = load(ROOT/'JOBS.json')
        accounting = native_accounting(all_jobs(schedule))
        advance_canary(schedule, model, accounting, set())


def cpu_stage(command: str, model: str | None) -> None:
    """Run a CPU stage under the cumulative actual-allocation ceiling.

    Args:
        command, model: Extraction or analysis stage to dispatch in a child process.

    Returns:
        On child success. A budget-exhausted CPU stage fails visibly while every
        GPU shard continues independently under its own allocation deadline.
    """
    import signal
    import sys
    arguments = [sys.executable, '-B', '-m', 'moe_exp.study.whole', command, '--cpu-worker']
    if model:
        arguments += ['--model', model]
    prior = verified(ROOT/'RELEASE.json').get('prior_usage', {}).get('new_cpu_core_hours', 0.)
    jobs = [j for j in all_jobs(load(ROOT/'JOBS.json')) if not j['gpus']]
    accounting = native_accounting(jobs)
    require(all(j['id'] in accounting for j in jobs), 'native CPU accounting required before CPU stage')
    last_usage = prior+sum(accounting[j['id']]['elapsed']*j['cpus']/3600 for j in jobs)
    last_time = time.time()
    require(last_usage < CPU_WORK_LIMIT, 'CPU allowance exhausted before stage; GPU production continues')
    process = subprocess.Popen(arguments, start_new_session=True)
    while process.poll() is None:
        try:
            jobs = [j for j in all_jobs(load(ROOT/'JOBS.json')) if not j['gpus']]
            accounting = native_accounting(jobs)
            require(all(j['id'] in accounting for j in jobs), 'CPU accounting registration pending')
            last_usage = prior+sum(accounting[j['id']]['elapsed']*j['cpus']/3600 for j in jobs)
            last_time = time.time()
        except Exception as error:
            driver.log({'event': 'whole-cpu-accounting-retry', 'error': str(error)})
        # At most two eight-CPU extractions and the one-CPU monitor overlap.
        upper_usage = last_usage+17*(time.time()-last_time)/3600
        if upper_usage >= CPU_WORK_LIMIT:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            publish(ROOT/'cpu-budget'/(command+'-'+str(model)+'.json'), {
                'command': command, 'model': model, 'charged_core_hours_upper': upper_usage,
                'reason': 'actual CPU core-hour ceiling; GPU production continues',
            })
            raise SystemExit(2)
        time.sleep(10)
    require(process.returncode == 0, f'whole {command} worker failed with exit {process.returncode}')


def monitor() -> None:
    """Supervise this bounded graph; every native failure and load second is charged.

    Returns:
        After analysis or its own deadline. Only declared scientific gates stop GPUs.
    """
    deadline = driver.deadline_from_slurm(margin=120)
    checked, projected, stopped = set(), set(), set()
    while time.time() < deadline:
        try:
            schedule = load(ROOT/'JOBS.json')
            jobs = all_jobs(schedule)
            accounting, gpu_hours = accounting_snapshot(jobs)
            storage_stop = load(ROOT/'ACCOUNTING.json')['work_remaining_bytes'] < 14*2**30
            if (ROOT/'STOP.json').exists() or time.time() >= CUTOFF or storage_stop:
                if not (ROOT/'STOP.json').exists():
                    publish(ROOT/'STOP.json', {'reason': 'storage_reserve' if storage_stop else 'science_cutoff',
                                               'gpu_hours': gpu_hours})
                cancel_jobs([j for j in jobs if j['gpus']], accounting)
                stopped.update(MODELS)
            validation = accounting.get(schedule['validation']['id'], {})
            if terminal(validation) and validation.get('state') != 'COMPLETED':
                cancel_jobs([j for j in jobs if j['gpus']], accounting)
                driver.checkpoint(ROOT/'VALIDATION_STOP.json', {'reason': 'cpu_validation_failed', 'native': validation})
                return
            for model in MODELS:
                check_production(schedule, model, accounting, checked, projected, stopped, gpu_hours)
            with locked(ROOT, 'schedule-progression'):
                schedule = load(ROOT/'JOBS.json')
                for model in MODELS:
                    advance_canary(schedule, model, accounting, stopped)
                advance_extraction(schedule, accounting, stopped)
            if schedule.get('analysis') and terminal(accounting.get(schedule['analysis']['id'], {})):
                return
            if load(ROOT/'ACCOUNTING.json').get('new_cpu_core_hours', 0.) >= CPU_WORK_LIMIT-CPU_TAIL_RESERVE:
                driver.log({'event': 'whole-monitor-cpu-reserve', 'production_continues': True})
                return
            time.sleep(60)
        except Exception:
            import traceback
            driver.log({'event': 'whole-monitor-retry', 'error': traceback.format_exc()})
            time.sleep(60)
    driver.log({'event': 'whole-monitor-deadline', 'production_continues': True})


def inference_stage(args: argparse.Namespace) -> None:
    """Run a GPU stage and preserve an explicit software-failure canary receipt."""
    require(os.environ.get('SLURM_JOB_GPUS'), 'whole inference requires allocated GPUs')
    try:
        run(args.model, args.shard, driver.deadline_from_slurm(margin=120), max_inflight=args.max_inflight,
            canary=args.command == 'canary', dose=args.dose)
    except BaseException as error:
        if args.command == 'canary':
            path = ROOT/args.model/('CANARY_HALF.json' if args.dose == 'half' else 'CANARY.json')
            if not path.exists():
                publish(path, {'model': args.model, 'dose': args.dose or 'main',
                    'job': os.environ['SLURM_JOB_ID'], 'gate': {'passed': False, 'fluency_failures': [],
                    'software_failures': [f'{type(error).__name__}: {error}']}})
        raise


def validate_stage() -> None:
    """Run the allocated study suite, nominations and planning before publishing qualification."""
    import unittest
    native_quota = work_remaining_bytes()
    require(native_quota > 14*2**30, 'WORK reserve is insufficient for the bounded whole experiment')
    preparation = prepare()
    nomination_proof = audit_nominations()
    suite = unittest.defaultTestLoader.discover(str(REPO/'tests/study'), top_level_dir=str(REPO/'tests/study'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        publish(ROOT/'CPU_VALIDATION.json', {'passed': False, 'tests': result.testsRun,
                                            'job': os.environ['SLURM_JOB_ID']})
        raise ValueError('allocated whole study suite failed')
    publish(ROOT/'CPU_VALIDATION.json', {'passed': True, 'tests': result.testsRun,
                                        'job': os.environ['SLURM_JOB_ID'], 'preparation': preparation,
                                        'work_remaining_bytes': native_quota,
                                        'nomination_audit_sha256': nomination_proof['sha256'],
                                        'sd_mde_sha256': verified(ROOT/'SD_MDE.json')['sha256']})


def audit_nominations() -> dict:
    """Check rebuilt manifests against the saved DP selections and record every cell.

    Returns:
        Source-bound per-cell audit, including actual retained RD and manipulation
        frequencies. A mismatch in any GPT promote optimum fails qualification.
    """
    release = verified(ROOT/'RELEASE.json')
    proposal = verified(Path(release['nomination_reference']))
    references = {(r['dose'], r['fold'], r['class']): r for r in proposal['records']}
    records = []
    for model in MODELS:
        for dose in ('main', 'half'):
            for cell in manifest(model, dose)['cells']:
                record = {key: cell[key] for key in ('fold', 'class', 'operation', 'qualified',
                    'activate', 'deactivate', 'retained_rd_fraction', 'replaced_deactivation_pairs',
                    'activation_changed', 'mean_native_frequency')}
                if model == 'gpt' and cell['operation'] == 'promote':
                    reference = references[dose, cell['fold'], cell['class']]
                    for arm, roles in (('target', reference), ('random', reference['controls'])):
                        for role in ('activate', 'deactivate'):
                            actual = {(a['layer'], e) for a in cell[arm] for e in a[role]}
                            expected = {(p['layer'], p['expert']) for p in roles[role]}
                            require(actual == expected, f'rebuilt GPT {dose}/{cell["fold"]}/{cell["class"]} differs from DP')
                    require(math.isclose(cell['retained_rd_fraction'], reference['rd_fraction']), 'retained RD differs from DP')
                    record['saved_dp_match'] = True
                records.append({'model': model, 'dose': dose, **record})
    return publish(ROOT/'NOMINATION_AUDIT.json', {
        'records': records, 'dp_reference_sha256': proposal['sha256'],
        'gpt_promote_dp_matches': sum(r.get('saved_dp_match', False) for r in records),
        'outcomes_used': False,
    })


def main(argv=None):
    """Dispatch preparation, canary and generation through one maintained entry point."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'canary', 'run', 'extract', 'analyze', 'validate', 'monitor', 'launch'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dose', choices=('main', 'half'))
    parser.add_argument('--shard', type=driver.parse_shard, default=(0, 1))
    parser.add_argument('--max-inflight', type=int, default=64)
    parser.add_argument('--cpu-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    verify_release()
    if args.command == 'launch':
        require(not (ROOT/'JOBS.json').exists(), 'whole graph already accepted; supervise the existing jobs')
        (ROOT/'logs').mkdir(parents=True, exist_ok=True, mode=0o700)
        validation = submit_stage('validate')
        schedule = {'validation': validation, 'canaries': {m: [] for m in MODELS},
                    'production': {m: [] for m in MODELS}, 'extraction': {}}
        driver.checkpoint(ROOT/'JOBS.json', schedule)
        for model in MODELS:
            schedule['canaries'][model].append(submit_stage('canary', model=model, dose='main',
                dependency='afterok:'+validation['id']))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
        schedule['monitor'] = submit_stage('monitor', dependency='afterany:'+validation['id'])
        driver.checkpoint(ROOT/'JOBS.json', schedule)
        driver.log({'event': 'whole-accepted', 'jobs': {j['command']+':'+str(j['model']): j['id'] for j in all_jobs(schedule)}})
        return
    require(os.environ.get('SLURM_JOB_ID'), 'whole stages require allocated Slurm resources')
    if args.command == 'prepare':
        driver.log({'event': 'prepare-whole', 'results': prepare((args.model,) if args.model else MODELS)})
    elif args.command in ('canary', 'run'):
        inference_stage(args)
        if args.command == 'canary':
            try:
                queue_after_canary(args.model)
            except Exception as error:
                driver.log({'event': 'whole-canary-scheduling-retry-needed', 'error': str(error)})
    elif args.command == 'extract':
        if not args.cpu_worker:
            return cpu_stage(args.command, args.model)
        from .whole_effects import extract
        extract(args.model)
    elif args.command == 'analyze':
        if not args.cpu_worker:
            return cpu_stage(args.command, args.model)
        from .whole_effects import analyze
        analyze()
        from .effects import analyze as historical_analysis
        models = [m for m in driver.MODELS if (driver.DRIVER/m/'ASSIGNMENTS.json').exists()]
        main_result = historical_analysis(models, ROOT/'corrected-main', 'main')
        strong_models = [m for m in models if (driver.DRIVER/m/'STRONG.json').exists()]
        historical_analysis(strong_models, ROOT/'corrected-strong', 'strong')
        link = main_result.get('efficiency_link', {})
        publish(ROOT/'analysis/CORRECTED_MAIN_LINK.json', link)
        with (ROOT/'analysis/REPORT.md').open('a') as report:
            report.write('\nCorrected main activate:log_total link status: '+str(link.get('status', 'qualified'))+
                         '; bounded rows: '+str(link.get('bounded_row_count', 0))+'. '
                         'Every bounded identity, interval and reason is in CORRECTED_MAIN_LINK.json.\n')
    elif args.command == 'validate':
        validate_stage()
    elif args.command == 'monitor':
        monitor()


if __name__ == '__main__':
    main()
