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

from .common import CLASSES, REPO, check_files, digest, load, publish, require, verified
from . import driver

MODELS = ('gpt', 'qwen36')
SUPPRESS = {'gpt': ('Read', 'Analyze', 'Plan', 'Explore'), 'qwen36': ('Analyze', 'Plan', 'Implement')}
DOSES = {
    'gpt': {'main': {'promote': (10, 50), 'suppress': (0, 20)},
            'half': {'promote': (5, 25), 'suppress': (0, 10)}},
    'qwen36': {'main': {'promote': (15, 0), 'suppress': (0, 500)},
               'half': {'promote': (8, 0), 'suppress': (0, 250)}},
}
ROOT = Path(os.environ.get('WHOLE_ROOT', REPO/'report/whole-reasoning-20261020'))
# A minute of the full 12-GPU concurrency plus cancellation latency stays inside 72.
GPU_CEILING = 72.
GPU_STOP = 71.6
CUTOFF = 1792726200.  # 2026-10-23 03:30 UTC (05:30 Europe/Rome).


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


def select_pairs(candidates: list[dict], count: int, layers: list[int], role: str) -> tuple[list[dict], dict]:
    """Apply the declared layer cap and half-availability rule to ranked pairs."""
    cap = 1 if role == 'activate' else math.ceil(count/len(layers))
    used, selected = Counter(), []
    for pair in candidates:
        if len(selected) == count:
            break
        if used[pair['layer']] < cap:
            selected.append(pair)
            used[pair['layer']] += 1
    qualified = len(selected) >= math.ceil(count/2)
    return (selected if qualified else []), {
        'requested': count, 'available': len(selected), 'used': len(selected) if qualified else 0,
        'per_layer_cap': cap, 'qualified': qualified,
    }


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
    activate, act_availability = select_pairs(positive, act_count, layers, 'activate')
    deactivate, deact_availability = select_pairs(negative if operation == 'promote' else positive,
                                                  deact_count, layers, 'deactivate')
    opposite_act, _ = select_pairs(negative, act_count, layers, 'activate')
    opposite_deact, _ = select_pairs(positive if operation == 'promote' else negative,
                                     deact_count, layers, 'deactivate')
    excluded = {(pair['layer'], pair['expert']) for pair in activate+deactivate+opposite_act+opposite_deact}
    qualified = act_availability['qualified'] and deact_availability['qualified']
    controls = (layer_matched_controls(activate, deactivate, nomination['frequency'], layers, excluded,
                [model, nomination['fold'], nomination['class'], dose, operation]) if qualified else
                {'activate': [], 'deactivate': []})

    return {
        'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation,
        'dose': dose, 'qualified': qualified,
        'activate': act_availability, 'deactivate': deact_availability,
        'target': _layer_actions({'activate': activate, 'deactivate': deactivate}, profile),
        'random': _layer_actions(controls, profile),
        'excluded_pairs': sorted(map(list, excluded)),
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
    publish(ROOT/'SD_MDE.json', planning(models))
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
                canary = {**row, 'canary': True, 'audit_capture': True, 'horizon': 4096,
                          'total_allowance': min(4096, row['total_allowance'])}
                canary['uid'] = digest(['whole-canary', value['sha256'], canary])
                records.append(canary)
    require(sum(r['operation'] != 'native' for r in records) == 32, 'canary cell availability differs')
    return records


def repetition_rate(tokens: list[int], width: int = 16) -> float:
    """Fraction of token n-gram positions whose n-gram has already appeared."""
    grams = [tuple(tokens[i:i+width]) for i in range(max(0, len(tokens)-width+1))]
    return 1-len(set(grams))/len(grams) if grams else 0.


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
    steered = [row for row in observed if row['operation'] != 'native']
    native = {r['family']: r['tokens'] for r in observed if r['arm'] == 'native'}
    ratios = [math.log(max(1, row['tokens'])/max(1, native[row['family']])) for row in steered
              if row['family'] in native]
    cap_count = sum(row['cap'] for row in steered)
    cap_rate = cap_count/len(steered) if steered else None
    ratio = math.exp(float(np.mean(ratios))) if ratios else None
    repetition = float(np.mean([row['repetition'] for row in steered])) if steered else None
    deactivation = [dose for row, dose in doses if row['operation'] == 'suppress']
    changed = (sum(d['changed_membership'] for d in deactivation)/sum(d['active_positions'] for d in deactivation)
               if deactivation else None)
    if changed is not None and changed < .05:
        software.append('deactivation_membership_below_5_percent')
    failures = []
    if cap_rate is not None and cap_rate > 2/32:
        failures.append('cap')
    if ratio is not None and ratio > 1.40:
        failures.append('length')
    if repetition is not None and repetition > .20:
        failures.append('repetition')
    return {'passed': not software and not failures and bool(steered), 'completed': len(observed),
            'steered_units': len(steered), 'cap_count': cap_count, 'cap_rate': cap_rate,
            'geometric_length_ratio_to_native': ratio, 'repetition_16gram': repetition,
            'deactivation_changed_membership': changed, 'software_failures': software,
            'fluency_failures': failures}


def run(model, shard, deadline, *, max_inflight=24, canary=False, dose=None):
    """Run one source-bound shard or canary on the existing hooked engine."""
    from .runtime import closure
    verify_release()
    value = manifest(model, dose)
    require(verified(ROOT/'CPU_VALIDATION.json')['passed'], 'allocated CPU validation did not pass')
    if not canary:
        selected = verified(ROOT/model/'SELECTED.json')
        require(selected['manifest_sha256'] == value['sha256'], 'production manifest lacks a passing canary')
    rows = canary_rows(value) if canary else driver.shard_rows(model, shard, 'whole')
    store = bound_store(model, value['dose'], canary)
    require(not any(store.state(r['uid']) for r in rows), 'whole execution is fresh only; preserve partials')
    executable = [r for r in rows if r['nomination_status'] == 'nominated']
    b = driver.binding()
    llm, kwargs, load_seconds = driver.hooked_llm(b, model, max_inflight)
    try:
        contract = driver.contract_for(b, model, executable)
        reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract})
        require(len(reports) == 2, 'whole requires both tensor-parallel ranks')
        tokenizer = llm.get_tokenizer()
        engine = driver.Engine(llm, store, min(deadline, CUTOFF), hooked=True, max_inflight=max_inflight,
                               checkpoint_seconds=120, drain_seconds=120)
        build = driver.intervention_builder(b, model, tokenizer, closure(tokenizer, model),
                                             limit=4096 if canary else None)
        metrics = engine.run(executable, build, driver.intervention_finisher(store, b, model),
                             family_blocks=True, stop_execution=lambda: (ROOT/'STOP.json').exists())
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
    cpus = 16 if gpu else 1 if command == 'monitor' else 4 if command == 'validate' else 2
    memory = 120 if gpu else 2 if command == 'monitor' else 24 if command != 'analyze' else 32
    minutes = {
        'canary': 60,
        'run': 290,
        'validate': 30,
        'monitor': 540,
        'extract': 30,
        'analyze': 60,
    }[command]
    args = [command]
    if model:
        args += ['--model', model]
    if dose:
        args += ['--dose', dose]
    if shard:
        args += ['--shard', f'{shard[0]}/{shard[1]}']
    exports = {'RELEASE_ROOT': release['release_root'], 'STUDY_DATA_ROOT': str(driver.ROOT),
               'WHOLE_ROOT': str(ROOT), 'STUDY_MODULE': 'moe_exp.study.whole',
               'MOE_EXP_SCORING_PROCESSES': '2'}
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(';')[0]
    require(job_id.isdecimal(), 'sbatch did not return a job ID')
    readback = subprocess.run(['scontrol', 'show', 'job', '-o', job_id], capture_output=True, text=True, check=True).stdout
    require('UserId=lmolfett(' in readback and 'Account=iscrc_miosr ' in readback and
            'Partition=boost_usr_prod ' in readback and f'NumCPUs={cpus} ' in readback,
            'accepted whole job resource/account readback differs')
    require(not gpu or 'gres/gpu=2' in readback, 'accepted whole GPU count differs')
    return {'id': job_id, 'command': command, 'model': model, 'dose': dose, 'shard': shard,
            'gpus': 2 if gpu else 0, 'cpus': cpus, 'minutes': minutes, 'held': held,
            'sbatch': result.stdout.strip(), 'args': cmd, 'readback': readback, 'submitted': time.time()}


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


def advance_canary(schedule: dict, model: str, accounting: dict, stopped: set) -> None:
    """Advance only passing canaries, or the one declared fluency fallback."""
    attempts = schedule['canaries'][model]
    attempt = attempts[-1]
    state = accounting.get(attempt['id'], {})
    if model in stopped or schedule['production'][model] or not terminal(state):
        return
    path = ROOT/model/('CANARY.json' if attempt['dose'] == 'main' else 'CANARY_HALF.json')
    gate = verified(path)['gate'] if path.exists() else None
    if state.get('state') == 'COMPLETED' and gate and gate['passed']:
        bound_store(model, attempt['dose'])
        count = 2 if model == 'gpt' else 5
        for index in range(count):
            job = submit_stage('run', model=model, dose=attempt['dose'], shard=[index, count],
                               dependency='afterok:'+attempt['id'], held=model == 'qwen36' and index == 4)
            schedule['production'][model].append(job)
            driver.checkpoint(ROOT/'JOBS.json', schedule)
    elif gate and gate['fluency_failures'] and not gate['software_failures'] and attempt['dose'] == 'main':
        attempts.append(submit_stage('canary', model=model, dose='half', dependency='afterany:'+attempt['id']))
        driver.checkpoint(ROOT/'JOBS.json', schedule)
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
        other_projection = 0. if other in stopped else 11.2 if other == 'gpt' else 48.3
    canary_usage = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600
                       for m in MODELS for j in schedule['canaries'][m])
    projection = projected_model+max(other_usage, other_projection)+canary_usage
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
    """Release at most the fifth shard, then arrange afterany extraction and analysis."""
    production_jobs = [j for m in MODELS for j in schedule['production'][m]]
    if any(terminal(accounting.get(j['id'], {})) for j in production_jobs):
        for job in production_jobs:
            if job['held'] and not terminal(accounting.get(job['id'], {})) and not (ROOT/'STOP.json').exists():
                subprocess.run(['scontrol', 'release', job['id']], check=True)
                job['held'] = False
                driver.checkpoint(ROOT/'JOBS.json', schedule)
    for model in MODELS:
        production = schedule['production'][model]
        if (production or model in stopped) and model not in schedule['extraction']:
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
    gpu_hours = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600 for j in jobs)
    cpu_hours = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['cpus']/3600 for j in jobs if not j['gpus'])
    driver.checkpoint(ROOT/'ACCOUNTING.json', {'native': accounting, 'new_gpu_hours': gpu_hours,
                      'retained_r3_gpu_hours': .3461111111111111, 'new_cpu_core_hours': cpu_hours,
                      'gpu_ceiling': GPU_CEILING, 'cpu_ceiling': 16, 'time': time.time(),
                      'work_remaining_bytes': work_remaining_bytes()})
    return accounting, gpu_hours


def monitor() -> None:
    """Supervise this bounded graph; every native failure and load second is charged.

    Returns:
        After final CPU analysis, or after cancelling GPUs at the bounded deadline.
    """
    deadline = min(driver.deadline_from_slurm(margin=240), CUTOFF-3600)
    checked, projected, stopped = set(), set(), set()
    try:
        while time.time() < deadline:
            schedule = load(ROOT/'JOBS.json')
            jobs = all_jobs(schedule)
            accounting, gpu_hours = accounting_snapshot(jobs)
            storage_stop = load(ROOT/'ACCOUNTING.json')['work_remaining_bytes'] < 14*2**30
            if gpu_hours >= GPU_STOP or time.time() >= CUTOFF-600 or storage_stop:
                if not (ROOT/'STOP.json').exists():
                    publish(ROOT/'STOP.json', {'reason': 'storage_reserve' if storage_stop else 'hard_budget_or_cutoff',
                                               'gpu_hours': gpu_hours})
                cancel_jobs([j for j in jobs if j['gpus']], accounting)
                stopped.update(MODELS)
            validation = accounting.get(schedule['validation']['id'], {})
            if terminal(validation) and validation.get('state') != 'COMPLETED':
                cancel_jobs([j for j in jobs if j['gpus']], accounting)
                stopped.update(MODELS)
            for model in MODELS:
                advance_canary(schedule, model, accounting, stopped)
                check_production(schedule, model, accounting, checked, projected, stopped, gpu_hours)
            advance_extraction(schedule, accounting, stopped)
            if schedule.get('analysis') and terminal(accounting.get(schedule['analysis']['id'], {})):
                return
            time.sleep(45)
        if not (ROOT/'STOP.json').exists():
            publish(ROOT/'STOP.json', {'reason': 'bounded_supervision_deadline'})
    except BaseException as error:
        if not (ROOT/'STOP.json').exists():
            publish(ROOT/'STOP.json', {'reason': 'supervision_failure', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        schedule = load(ROOT/'JOBS.json')
        jobs = all_jobs(schedule)
        try:
            accounting = native_accounting(jobs)
        except Exception:
            accounting = {}
        cancel_jobs([j for j in jobs if j['gpus']], accounting)


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
    suite = unittest.defaultTestLoader.discover(str(REPO/'tests/study'), top_level_dir=str(REPO/'tests/study'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        publish(ROOT/'CPU_VALIDATION.json', {'passed': False, 'tests': result.testsRun,
                                            'job': os.environ['SLURM_JOB_ID']})
        raise ValueError('allocated whole study suite failed')
    preparation = prepare()
    publish(ROOT/'CPU_VALIDATION.json', {'passed': True, 'tests': result.testsRun,
                                        'job': os.environ['SLURM_JOB_ID'], 'preparation': preparation,
                                        'work_remaining_bytes': native_quota,
                                        'sd_mde_sha256': verified(ROOT/'SD_MDE.json')['sha256']})


def main(argv=None):
    """Dispatch preparation, canary and generation through one maintained entry point."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'canary', 'run', 'extract', 'analyze', 'validate', 'monitor', 'launch'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dose', choices=('main', 'half'))
    parser.add_argument('--shard', type=driver.parse_shard, default=(0, 1))
    parser.add_argument('--max-inflight', type=int, default=24)
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
    elif args.command == 'extract':
        from .whole_effects import extract
        extract(args.model)
    elif args.command == 'analyze':
        from .whole_effects import analyze
        analyze()
        from .effects import analyze as historical_analysis
        models = [m for m in driver.MODELS if (driver.DRIVER/m/'ASSIGNMENTS.json').exists()]
        historical_analysis(models, ROOT/'corrected-main', 'main')
        strong_models = [m for m in models if (driver.DRIVER/m/'STRONG.json').exists()]
        historical_analysis(strong_models, ROOT/'corrected-strong', 'strong')
    elif args.command == 'validate':
        validate_stage()
    elif args.command == 'monitor':
        monitor()


if __name__ == '__main__':
    main()
