"""Whole-reasoning experiment on the reduced driver's continuous engine.

The frozen manifest owns family order, original prompts, doses and seeds. Plain
receipts bind fresh stores and report canary/production gates; no historical
continuation is reused. Run through scripts/study/driver.sh with STUDY_MODULE
set to moe_exp.study.whole and RELEASE_ROOT set to a materialized commit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from .common import CLASSES, REPO, check_files, digest, file_sha, load, locked, publish, require, verified
from . import driver
from moe_exp.final_routing.selection import select_pairs

# Per-model run plan: production shards x walltime, canary walltime and the shared canary token cap.
# Sized from main-study throughput at ~70% speed with a margin, so shards finish their rows instead of
# being cut at the walltime (unfinished rows are lost work). GLM decodes about four times slower than
# the others and gets proportionally more shards and a lower canary cap. Strength-sweep releases run
# about 60% of the rows (efficiency cells only, no natives) with `sweep_shards`; GPT and GLM get extra
# sweep shards because their full dose reasons much longer and some rows run to the length limit.
PLAN = {
    # GPT's full dose broke down in canary 61601039 (6/32 rows stuck at the cap), so it starts at the
    # declared half dose; its promote arms reason ~3x longer, hence the larger production plan.
    'gpt': dict(shards=4, sweep_shards=6, run_minutes=240, canary_minutes=90, canary_tokens=32768,
                start_dose='half'),
    'qwen36': dict(shards=9, sweep_shards=6, run_minutes=240, canary_minutes=150, canary_tokens=32768),
    'qwen35': dict(shards=10, sweep_shards=6, run_minutes=240, canary_minutes=150, canary_tokens=32768),
    'qwen330b': dict(shards=6, sweep_shards=4, run_minutes=240, canary_minutes=120, canary_tokens=32768),
    'gemma': dict(shards=9, sweep_shards=6, run_minutes=240, canary_minutes=150, canary_tokens=32768),
    'glm': dict(shards=24, sweep_shards=16, run_minutes=240, canary_minutes=180, canary_tokens=16384),
    'nemotron': dict(shards=6, sweep_shards=4, run_minutes=240, canary_minutes=120, canary_tokens=32768),
    'lfm25': dict(shards=7, sweep_shards=5, run_minutes=240, canary_minutes=120, canary_tokens=32768),
}
# Steering strength levels as fractions of SteerMoE's published counts ('main' = full).
DOSES = {'main': 1., 'half': .5, 'quarter': .25}

# One release steers the models named in WHOLE_MODELS under its own ceilings and science cutoff, all
# exported to every stage. A strength-sweep release also names one dose per model in WHOLE_SWEEP
# ("model:dose,..."): it runs only the efficiency-directed cells at that dose, without native rows and
# without a canary, and is analyzed together with the base releases (whole_effects.combine).
MODELS = tuple(os.environ.get('WHOLE_MODELS', 'gpt,qwen36').split(','))
SWEEP = dict(item.split(':') for item in os.environ.get('WHOLE_SWEEP', '').split(',') if item)
GPU_CEILING = float(os.environ.get('WHOLE_GPU_CEILING', '72'))
CPU_CEILING = float(os.environ.get('WHOLE_CPU_CEILING', '16'))
CUTOFF = float(os.environ.get('WHOLE_CUTOFF', '1792726200'))  # default 2026-10-23 05:30 Europe/Rome
require(set(MODELS) <= PLAN.keys(), 'unknown whole model')
require(set(SWEEP) <= set(MODELS) and set(SWEEP.values()) <= DOSES.keys(), 'unknown whole sweep model or dose')
SHARDS = {model: plan['sweep_shards' if model in SWEEP else 'shards'] for model, plan in PLAN.items()}
# A release may size its own shards ("model:count,..."), e.g. a top-up of a few hundred rows.
SHARDS.update({model: int(count) for model, count in
               (item.split(':') for item in os.environ.get('WHOLE_SHARDS', '').split(',') if item)})
RUN_MINUTES = {model: plan['run_minutes'] for model, plan in PLAN.items()}
CANARY_MINUTES = {model: plan['canary_minutes'] for model, plan in PLAN.items()}
ROOT = Path(os.environ.get('WHOLE_ROOT', REPO/'report/whole-reasoning-20261020'))
# Bootstrap draws and seed shared by every whole-reasoning interval.
DRAWS, SEED = 10000, 20261015


def manifest_path(model: str, dose: str, root: Path | None = None) -> Path:
    """The frozen manifest of one model and dose; main and half keep their original names."""
    return (root or ROOT)/model/('WHOLE.json' if dose == 'main' else f'WHOLE_{dose.upper()}.json')


def canary_path(model: str, dose: str) -> Path:
    """The canary receipt of one model and dose; main and half keep their original names."""
    return ROOT/model/('CANARY.json' if dose == 'main' else f'CANARY_{dose.upper()}.json')

# SteerMoE's published (activated, deactivated) counts: GPT-OSS-20B "faithful"/"unsafe", and
# Qwen3-30B-A3B "safe" for promotion and "faithful" for suppression (48 layers x 128 experts).
# The Qwen3 family uses those counts directly; other models get the same share of their own
# layer-expert pairs. Lower doses take the ceiling of the same fraction of every count.
PUBLISHED = {'gpt': {'promote': (10, 50), 'suppress': (0, 20)},
             'qwen3': {'promote': (15, 0), 'suppress': (0, 500)}}
QWEN3_LAYERS, QWEN3_EXPERTS = 48, 128


def dose_counts(model: str, profile: dict, dose: str, operation: str) -> tuple[int, int]:
    """Return (activated, deactivated) pair counts for one model, dose and operation."""
    if model == 'gpt':
        counts = PUBLISHED['gpt'][operation]
    elif model in ('qwen36', 'qwen35', 'qwen330b'):
        counts = PUBLISHED['qwen3'][operation]
    else:
        activate, deactivate = PUBLISHED['qwen3'][operation]
        layers, experts = len(profile['layers']), profile['experts']
        counts = (round(activate*layers/QWEN3_LAYERS),
                  round(deactivate*layers*experts/(QWEN3_LAYERS*QWEN3_EXPERTS)))
    return tuple(math.ceil(count*DOSES[dose]) for count in counts)


@lru_cache(maxsize=None)
def directions(model: str) -> dict:
    """Classes to promote (tied to shorter reasoning) and suppress (tied to longer reasoning).

    Signs come from the pre-registered efficiency link on native reasoning only. LFM2.5 was not in
    that receipt (no native lexical table then), so it uses the receipt's pooled cross-model signs.
    """
    prereg = verified(driver.DRIVER/'EFFICIENCY_LINK_PREREG.json')
    rows = ([cell for cell in prereg['cells'] if cell['model'] == model] if model in prereg['models']
            else prereg['pooled'])
    signs = {row['class']: row['tok_reason']['sign'] for row in rows}
    return {'promote': tuple(c for c in CLASSES if signs[c] == '-'),
            'suppress': tuple(c for c in CLASSES if signs[c] == '+'),
            'source': 'model' if model in prereg['models'] else 'pooled'}


def efficiency_directed(model: str, operation: str, class_name: str | None) -> bool:
    """Whether a cell enters the primary efficiency contrast."""
    return class_name in directions(model)['promote' if operation == 'promote' else 'suppress']


# CPU allocation time is charged across monitor, extraction and analysis, under the release's own
# core-hour ceiling. Leave half a core-hour for concurrent shutdown and native-accounting latency; the
# monitor stops supervising once only the tail reserve remains.
CPU_WORK_LIMIT = CPU_CEILING-.5
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
    act_count, deact_count = dose_counts(model, profile, dose, operation)
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
                   inputs: dict, prompt_scope: bool, *, sweep: bool = False) -> dict:
    """Freeze all planned rows, including unavailable cells, in complete family blocks.

    Args:
        model, dose: Model identity and steering strength (see DOSES).
        nominations: Discovery-only global risk-difference receipt.
        b: Original study profile binding.
        natives: Original main-design native assignments.
        inputs: Family to original prompt token list.
        prompt_scope: Whether prompt qualification passed before the fallback cutoff.
        sweep: Strength-sweep manifest: efficiency-directed cells only, no native rows (they are the same
            computation as the base release's) and no audit captures.

    Returns:
        Manifest with planned row counts, actual cell counts and unique UIDs.
    """
    from moe_exp.final_routing.runtime import remaining
    profile = b['models'][model]
    suppressed = directions(model)['suppress']
    promoted = directions(model)['promote'] if sweep else CLASSES
    cells = [cell_actions(n, profile, model, dose, operation) for n in nominations['records']
             for operation in ('promote', 'suppress')
             if n['class'] in (promoted if operation == 'promote' else suppressed)]
    by_cell = {(c['fold'], c['class'], c['operation']): c for c in cells}
    require(all(c['qualified'] for c in cells if efficiency_directed(model, c['operation'], c['class'])),
            'every efficiency-directed cell must be nominated')
    rows = []
    for native in sorted(natives, key=lambda row: row['family']):
        prompt = inputs[native['family']]
        allowance = remaining(profile, prompt, [])
        base = {key: native[key] for key in ('model', 'family', 'benchmark', 'fold', 'seed', 'profile_sha256')}
        base.update(source={**native['source'], 'prefix_tokens': 0}, original_prompt=True,
                    prompt_sha256=digest(prompt), whole=True, prompt_scope=prompt_scope,
                    pulse_tokens=0, horizon=allowance, total_allowance=allowance, prefix_status='enrolled',
                    capture_layers=profile['layers'], action=None, dose=dose, depth_third=None)
        for arm in (() if sweep else ('native', 'native_repeat')):
            row = {**base, 'arm': arm, 'class': None, 'operation': 'native', 'actions': [],
                   'nomination_status': 'nominated'}
            if arm == 'native_repeat':
                row['seed'] = int(digest([native['seed'], 'whole-native-repeat'])[:8], 16)
                require(row['seed'] != native['seed'], 'native repeat seed collision')
            rows.append(row)
        for operation in ('promote', 'suppress'):
            for class_name in (promoted if operation == 'promote' else suppressed):
                cell = by_cell[native['fold'], class_name, operation]
                for role in ('target', 'random'):
                    rows.append({**base, 'arm': role+'_'+operation, 'class': class_name,
                                 'operation': operation, 'actions': cell[role],
                                 'steered_native_frequency': cell.get('mean_native_frequency', {}).get(role),
                                 'nomination_status': 'nominated' if cell['qualified'] else 'dropped_availability'})
    # Audit flags are selected without reading a whole-arm outcome, before execution.
    audit_units = 0 if sweep else 32
    audit_keys = sorted(range(len(rows)), key=lambda i: digest(['whole-audit', model, dose, rows[i]]))[:audit_units]
    for index, row in enumerate(rows):
        row['audit_capture'] = index in audit_keys
        row['uid'] = digest(['whole-v1', nominations.get('release_commit', 'development'), nominations['sha256'], row])
    # Two native arms (none in a sweep) plus a target and a random arm for every planned cell.
    expected = len(natives)*((0 if sweep else 2)+2*(len(promoted)+len(suppressed)))
    require(len(rows) == expected and len({r['uid'] for r in rows}) == expected, 'whole row count/UID differs')
    value = {'model': model, 'dose': dose, 'scope': 'prompt-and-generated-reasoning' if prompt_scope else
             'generated-reasoning-only; not a SteerMoE whole-token replication',
             'nominations_sha256': nominations['sha256'], 'cells': cells, 'records': rows,
             'planned': expected, 'executable': sum(r['nomination_status'] == 'nominated' for r in rows),
             'audit_units': audit_units, 'outcomes_used_for_selection': False}
    # Base manifests keep their original keys, so rebuilt r7/r8 manifests stay byte-identical.
    return {**value, 'design': 'efficiency-sweep'} if sweep else value


def prepare(models=MODELS):
    """Derive and publish main/half manifests and native-only planning evidence on CPUs."""
    require(os.environ.get('SLURM_JOB_ID'), 'whole discovery and planning require an allocated CPU job')
    verify_release()
    b = driver.binding()
    results = {}
    scope = os.environ.get('WHOLE_PROMPT_SCOPE', '1') == '1'
    for model in models:
        try:
            results.update(prepare_model(b, model, scope))
        except Exception as error:
            # One infeasible model must not block the others; its canary then stops on the missing manifest.
            results[model] = {'prepared': False, 'error': f'{type(error).__name__}: {error}'}
    require(any(':' in key for key in results), 'no whole model could be prepared')
    from .whole_effects import planning
    try:
        planning_result = planning(models)
    except Exception as error:
        planning_result = {'transport_status': 'SUSPECTED', 'available': False,
                           'error': f'{type(error).__name__}: {error}', 'gates_science': False}
    publish(ROOT/'SD_MDE.json', planning_result)
    return results


def prepare_model(b: dict, model: str, scope: bool) -> dict:
    """Derive one model's nominations and publish its manifests.

    A base release publishes the main and half-dose manifests; a sweep release publishes the one sweep
    dose and selects it (no canary).
    """
    if (ROOT/'TOPUP.json').exists():
        return {model+':'+SWEEP[model]: topup_manifest(model, verified(ROOT/'TOPUP.json')['models'][model])}
    from moe_exp.final_routing.importer import derive_whole
    from .runtime import prompt_prefix
    folder = ROOT/model
    nomination_path = folder/'WHOLE_NOMINATIONS.json'
    nominations = verified(nomination_path) if nomination_path.exists() else derive_whole(b, model, folder)
    natives = [r for r in verified(driver.DRIVER/model/'ASSIGNMENTS.json')['records'] if r['arm'] == 'native']
    # Native rosters: 129 families for most models, 128 for Nemotron and Qwen3-30B, 112 for LFM2.5.
    require(natives and len({r['family'] for r in natives}) == len(natives), 'whole native family roster differs')
    inputs = {}
    for row in natives:
        original = {**row, 'original_prompt': True, 'source': {**row['source'], 'prefix_tokens': 0}}
        prompt, prefix, _ = prompt_prefix(b, model, original, None)
        require(not prefix, 'original prompt row retained a prefix')
        inputs[row['family']] = prompt
    results = {}
    for dose in ([SWEEP[model]] if model in SWEEP else ['main', 'half']):
        value = publish(manifest_path(model, dose),
                        build_manifest(model, dose, nominations, b, natives, inputs, scope, sweep=model in SWEEP))
        results[model+':'+dose] = {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}
        if model in SWEEP:
            publish(folder/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                             'sweep': True})
    return results


def topup_manifest(model: str, spec: dict) -> dict:
    """Publish the never-started rows of a base release's failed shards as this release's manifest.

    A top-up release reruns nothing: rows with any saved state in the parent store (complete, output or
    partial) stay with the parent, where an unfinished row remains an unknown outcome. The rows keep their
    parent UIDs, seeds and actions, so whole_effects joins them to the parent release by UID.

    Args:
        model: Model whose base release lost shards.
        spec: {'parent': parent release root, 'shards': [[index, count], ...]} of the failed jobs.
    """
    dose, parent = SWEEP[model], Path(spec['parent'])
    base = verified(manifest_path(model, dose, parent))
    store = driver.Store(parent/'stores'/model/dose/'units')
    families = sorted({row['family'] for row in base['records']})
    failed = {family for i, family in enumerate(families) for index, count in spec['shards'] if i % count == index}
    rows = [row for row in base['records'] if row['family'] in failed and not store.state(row['uid'])]
    require(rows, 'top-up has no never-started rows')
    kept = {k: v for k, v in base.items() if k not in ('sha256', 'records', 'planned', 'executable')}
    value = publish(manifest_path(model, dose), {
        **kept, 'records': rows, 'planned': len(rows),
        'executable': sum(row['nomination_status'] == 'nominated' for row in rows), 'design': 'top-up',
        'parent_root': str(parent), 'parent_manifest_sha256': base['sha256'], 'failed_shards': spec['shards']})
    publish(ROOT/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                         'sweep': True, 'topup': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}


def manifest(model: str, dose: str | None = None) -> dict:
    """Read the selected manifest; selection is fixed by the passing canary (or the sweep)."""
    if dose is None:
        selected = ROOT/model/'SELECTED.json'
        dose = verified(selected)['dose'] if selected.exists() else 'main'
    return verified(manifest_path(model, dose))


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
    # Verify is a promote class for every model; the suppress cell is the model's first suppress class.
    require('Verify' in directions(model)['promote'], 'canary promote class differs')
    chosen_class = directions(model)['suppress'][0]
    for family in families:
        members = [r for r in value['records'] if r['family'] == family]
        for row in members:
            keep = row['arm'] in ('native', 'native_repeat') or (
                row['operation'] == 'promote' and row['class'] == 'Verify') or (
                row['operation'] == 'suppress' and row['class'] == chosen_class)
            if keep and row['nomination_status'] == 'nominated':
                # Gates read dose counters only; full audits stay on the pre-selected production units.
                canary = {**row, 'canary': True}
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
    # Only breakdown blocks a run: rows stuck at the length limit or looping. A longer but fluent
    # reasoning is the outcome under study, so the length ratio is reported, never gated
    # (user decision 2026-10-21, before any production outcome).
    failures = []
    if cap_rate is not None and cap_rate-native_cap > .15:
        failures.append('cap')
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


def fluency(store, rows: list[dict], *, require_complete: bool = False, censor_unfinished: bool = False) -> dict:
    """Evaluate the frozen fluency and same-state dose gates.

    Args:
        store, rows: The execution store and the rows to gate.
        require_complete: Every row must be observed (the canary).
        censor_unfinished: A row still decoding when the job reached its deadline counts as reaching the
            length limit at its observed length, for steered and native rows alike. Dose checks use
            completed rows only.

    Returns:
        Relative fluency gates, software checks and their denominators.
    """
    observed, doses, software = [], [], []
    for row in rows:
        state = store.state(row['uid']) or {}
        if state.get('status') != 'complete':
            if censor_unfinished and state.get('status') == 'partial' and state.get('tokens'):
                observed.append({**row, 'tokens': len(state['tokens']), 'cap': True, 'censored': True,
                                 'repetition': repetition_rate(state['tokens'])})
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
            'completed': sum(not r.get('censored') for r in observed),
            'censored_at_deadline': sum(bool(r.get('censored')) for r in observed), **relative,
            'deactivation_changed_membership': changed, 'software_failures': software,
            'observed_steered_units': sum(r['operation'] != 'native' for r in observed)}


def canary_decision(model: str, value: dict) -> dict:
    """Verify a recorded user decision that replaces a failed canary gate for this exact manifest.

    The decision references the canary receipt it overrides by path and file hash, so the evidence
    behind it stays auditable.
    """
    decision = verified(ROOT/model/'CANARY_DECISION.json')
    require(decision['decision'] == 'proceed' and decision['manifest_sha256'] == value['sha256'] and
            file_sha(Path(decision['canary_receipt'])) == decision['canary_receipt_sha256_file'],
            'production canary decision differs')
    return decision


def produce(model: str) -> None:
    """Submit production for a model whose canary gate is replaced by a recorded user decision.

    Runs on the login node after the allocated CPU validation passed, like `launch`.
    """
    with locked(ROOT, 'schedule-progression'):
        schedule = load(ROOT/'JOBS.json')
        require(verified(ROOT/'CPU_VALIDATION.json')['passed'], 'allocated CPU validation did not pass')
        selected = verified(ROOT/model/'SELECTED.json')
        canary_decision(model, manifest(model, selected['dose']))
        bound_store(model, selected['dose'])
        accounting = native_accounting(all_jobs(schedule))
        admit_shard_plan(schedule, model, accounting, set())
        for index in range(len(schedule['production'][model]), SHARDS[model]):
            schedule['production'][model].append(submit_stage('run', model=model, dose=selected['dose'],
                                                              shard=[index, SHARDS[model]]))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
        advance_extraction(schedule, accounting, set())


def admission_order(rows: list[dict]) -> list[dict]:
    """Order all primary and native rows before other promote rows, without barriers."""
    return sorted(rows, key=lambda r: (
        int(r['operation'] == 'promote' and not efficiency_directed(r['model'], 'promote', r['class'])),
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
        # A sweep dose runs without a canary (breakdown is measured as an outcome). Otherwise a recorded
        # user decision replaces a failed canary, or the passing canary receipt qualifies the manifest.
        if not selected.get('sweep'):
            if (ROOT/model/'CANARY_DECISION.json').exists():
                canary_decision(model, value)
            else:
                proof = verified(canary_path(model, value['dose']))
                require(proof['gate']['passed'] and proof['job'] == selected['canary_job'] and
                        proof['manifest_sha256'] == value['sha256'], 'production canary qualification differs')
    rows = canary_rows(value) if canary else driver.shard_rows(model, shard, 'whole')
    store = bound_store(model, value['dose'], canary)
    require(not any(store.state(r['uid']) for r in rows), 'whole execution is fresh only; preserve partials')
    if time.time() >= CUTOFF:
        # A shard that starts after the science cutoff leaves its rows unknown without loading the model.
        driver.log({'event': 'whole-shard-after-cutoff', 'model': model, 'shard': list(shard)})
        return None
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
        # Canary rows share one per-model token cap so the relative gates compare like with like
        # inside the canary walltime; production keeps the full model-card allowance.
        build = driver.intervention_builder(b, model, tokenizer, closure(tokenizer, model),
                                            limit=PLAN[model]['canary_tokens'] if canary else None)
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
        result['gate'] = fluency(store, rows, require_complete=True, censor_unfinished=True)
        publish(canary_path(model, value['dose']), result)
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


def scheduler_dependency(dependency: str | None) -> str | None:
    """Reconcile only purged, natively proven satisfied scheduler conditions."""
    if not dependency:
        return None
    kind, *identifiers = dependency.split(':')
    require(kind in ('afterok', 'afterany'), 'unsupported whole dependency condition')
    live, retired = [], []
    for job_id in identifiers:
        result = subprocess.run(['scontrol', 'show', 'job', '-o', job_id], capture_output=True, text=True)
        if result.returncode == 0:
            live.append(job_id)
            continue
        if 'invalid job id' not in result.stderr.lower():
            result.check_returncode()
        state = native_accounting([{'id': job_id}]).get(job_id, {})
        require(terminal(state) and (kind == 'afterany' or
                state['state'] == 'COMPLETED' and state['exit_code'] == '0:0'),
                'purged dependency lacks its required native successful/terminal evidence')
        retired.append({'id': job_id, 'native': state})
    effective = kind+':'+':'.join(live) if live else None
    if retired:
        publish(ROOT/'dependencies'/(digest([dependency, effective])+'.json'), {
            'requested': dependency, 'effective': effective, 'satisfied_purged_conditions': retired,
        })
    return effective


def submission_environment(exports: dict) -> dict:
    """Return the caller's environment without its Slurm job state, plus the stage exports.

    Slurm always propagates SLURM_* and, under --export=ALL, the caller's values win over
    explicit ones. A stage submitted from inside a GPU canary would otherwise inherit
    SLURM_JOB_GPUS and the canary's own exports.
    """
    prefixes = ('SLURM_', 'SBATCH_', 'SRUN_', 'SALLOC_')
    return {**{k: v for k, v in os.environ.items() if not k.startswith(prefixes)}, **exports}


def submit_stage(command: str, *, model=None, dose=None, shard=None, dependency=None, held=False) -> dict:
    """Submit one bounded approved stage and read back its exact native resource request."""
    release = verified(ROOT/'RELEASE.json')
    gpu = command in ('canary', 'run')
    cpus = 16 if gpu else 1 if command == 'monitor' else 4 if command == 'validate' else 8
    memory = {'monitor': 2, 'validate': 24, 'extract': 24, 'analyze': 32, 'combine': 96}.get(command, 120)
    minutes = {
        'canary': CANARY_MINUTES.get(model),
        'run': planned_run_minutes(model) if model else None,
        'validate': 60,
        'monitor': 960,
        'extract': 90,
        'analyze': 120,
        'combine': 360,
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
               'WHOLE_MODELS': ','.join(MODELS), 'WHOLE_GPU_CEILING': str(GPU_CEILING),
               'WHOLE_CPU_CEILING': str(CPU_CEILING), 'WHOLE_CUTOFF': str(CUTOFF),
               'WHOLE_SWEEP': ','.join(m+':'+d for m, d in SWEEP.items()),
               'WHOLE_SHARDS': ','.join(f'{m}:{SHARDS[m]}' for m in MODELS),
               'MOE_EXP_SCORING_PROCESSES': '8' if command == 'extract' else '2'}
    cmd = ['sbatch', '--parsable', '--account=iscrc_miosr', '--partition=boost_usr_prod', '--qos=normal',
           '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus}', f'--mem={memory}G', f'--time={minutes}',
           '--job-name=whole-'+command+('-'+model if model else '')+('-'+dose if dose else ''),
           '--chdir='+release['release_root'], '--output='+str(ROOT/'logs/%x-%j.out'),
           '--error='+str(ROOT/'logs/%x-%j.err'),
           # Stage exports travel in the scrubbed environment (submission_environment); a comma-separated
           # --export list would split WHOLE_MODELS.
           '--export=ALL']
    if gpu:
        cmd += ['--gres=gpu:2']
    if held:
        cmd += ['--hold']
    cmd += [str(Path(release['release_root'])/'scripts/study/driver.sh'), *args]
    folder = ROOT/'submissions'/digest([command, model, dose, shard])
    if (folder/'ACCEPTED.json').exists():
        return verified(folder/'ACCEPTED.json')
    if (folder/'SUBMITTED.json').exists():
        submitted = verified(folder/'SUBMITTED.json')
        require([arg for arg in submitted['args'] if not arg.startswith('--dependency=')] == cmd and
                submitted['requested_dependency'] == dependency, 'recorded whole submission arguments differ')
    else:
        effective = scheduler_dependency(dependency)
        if effective:
            cmd.insert(1, '--dependency='+effective)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=submission_environment(exports))
        job_id = result.stdout.strip().split(';')[0]
        require(job_id.isdecimal(), 'sbatch did not return a job ID')
        submitted = publish(folder/'SUBMITTED.json', {
            'id': job_id, 'command': command, 'model': model, 'dose': dose, 'shard': shard,
            'gpus': 2 if gpu else 0, 'cpus': cpus, 'minutes': minutes, 'held': held,
            'sbatch': result.stdout.strip(), 'args': cmd, 'submitted': time.time(),
            'requested_dependency': dependency,
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


def planned_run_minutes(model: str) -> int:
    """Use the budget-bound shard limit once published, with the approved maximum before admission."""
    path = ROOT/model/'SHARD_PLAN.json'
    return verified(path)['minutes'] if path.exists() else RUN_MINUTES[model]


def reserved_gpu_hours(schedule: dict, accounting: dict, stopped: set, *, include_unsubmitted=True) -> float:
    """Charge finished allocations and reserve every outstanding planned GPU maximum."""
    jobs = [job for kind in ('canaries', 'production') for rows in schedule[kind].values() for job in rows]
    reserved = verified(ROOT/'RELEASE.json').get('prior_usage', {}).get('new_gpu_hours', 0.)
    for job in jobs:
        state = accounting.get(job['id'], {})
        reserved += state['elapsed']*job['gpus']/3600 if terminal(state) else job['gpus']*job['minutes']/60
    for model in MODELS:
        if include_unsubmitted and model not in stopped:
            missing = SHARDS[model]-len(schedule['production'].get(model, []))
            reserved += missing*2*planned_run_minutes(model)/60
    return reserved


def admit_shard_plan(schedule: dict, model: str, accounting: dict, stopped: set) -> None:
    """Bind all shard walltimes to remaining budget, charging every earlier failure."""
    path = ROOT/model/'SHARD_PLAN.json'
    if path.exists():
        return
    committed = reserved_gpu_hours(schedule, accounting, stopped, include_unsubmitted=False)
    # Reserve every other unstopped model's not-yet-submitted shard maxima before sizing this one.
    future_others = sum((SHARDS[m]-len(schedule['production'][m]))*2*planned_run_minutes(m)/60
                        for m in MODELS if m != model and m not in stopped)
    available = GPU_CEILING-committed-future_others
    minutes = min(RUN_MINUTES[model], math.floor(available*60/(2*SHARDS[model])))
    require(minutes > 0, 'remaining GPU allowance cannot admit this model')
    publish(path, {'model': model, 'shards': SHARDS[model], 'minutes': minutes,
                   'approved_maximum_minutes': RUN_MINUTES[model], 'committed_gpu_hours': committed,
                   'other_future_maximum_gpu_hours': future_others, 'gpu_ceiling': GPU_CEILING,
                   'reason': 'full allocation maxima plus all charged predecessor costs stay within the GPU ceiling'})


def advance_canary(schedule: dict, model: str, accounting: dict, stopped: set) -> None:
    """Advance only passing canaries, or the one declared fluency fallback."""
    attempts = schedule['canaries'][model]
    if not attempts:
        return  # no canary: production was submitted by `produce` under a recorded decision
    attempt = attempts[-1]
    state = accounting.get(attempt['id'], {})
    if (ROOT/'STOP.json').exists() or (ROOT/model/'STOP.json').exists() or (ROOT/model/'PRE_PRODUCTION_STOP.json').exists():
        stopped.add(model)
    if model in stopped:
        return
    path = canary_path(model, attempt['dose'])
    gate = verified(path)['gate'] if path.exists() else None
    current_canary = attempt['id'] == os.environ.get('SLURM_JOB_ID')
    if gate and gate['passed'] and (state.get('state') == 'COMPLETED' or current_canary):
        bound_store(model, attempt['dose'])
        admit_shard_plan(schedule, model, accounting, stopped)
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
        reserved = reserved_gpu_hours(schedule, accounting, stopped, include_unsubmitted=False)+fallback_maximum
        if reserved > GPU_CEILING:
            stopped.add(model)
            publish(ROOT/model/'PRE_PRODUCTION_STOP.json', {
                'reason': 'half-dose canary cannot fit alongside already committed allocation maxima',
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


def projected_gpu_hours(schedule: dict, model: str, accounting: dict, stopped: set) -> float:
    """A model's production GPU-hours: usage scaled from a completed quarter of its pairs, else its maxima."""
    jobs = schedule['production'][model]
    usage = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600 for j in jobs)
    progress = production_progress(model) if jobs else None
    if progress and progress['pairs'] >= .25*progress['executable_pairs']:
        return max(usage, usage/(progress['pairs']/progress['executable_pairs']))
    if model in stopped:
        return usage
    planned = sum(j['gpus']*j['minutes']/60 for j in jobs)+(SHARDS[model]-len(jobs))*2*planned_run_minutes(model)/60
    return max(usage, planned)


def check_projection(schedule: dict, model: str, accounting: dict, progress: dict, stopped: set,
                     gpu_hours: float) -> None:
    """Project at this model's first completed quarter, adding every other model's projection or maxima."""
    production = schedule['production'][model]
    model_usage = sum(accounting.get(job['id'], {}).get('elapsed', 0)*job['gpus']/3600 for job in production)
    fraction = progress['pairs']/progress['executable_pairs']
    projected_model = model_usage/fraction
    others = sum(projected_gpu_hours(schedule, m, accounting, stopped) for m in MODELS if m != model)
    canary_usage = sum(accounting.get(j['id'], {}).get('elapsed', 0)*j['gpus']/3600
                       for m in MODELS for j in schedule['canaries'][m])
    prior_gpu_hours = verified(ROOT/'RELEASE.json').get('prior_usage', {}).get('new_gpu_hours', 0.)
    projection = prior_gpu_hours+projected_model+others+canary_usage
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
    # The early gate compares steered rows with their family's native rows, which a sweep does not run;
    # sweep breakdown is reported as an outcome instead.
    if model not in checked and model not in SWEEP and elapsed >= (2700 if model == 'gpt' else 4500):
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
    # A sweep release's ceiling covers every shard's full walltime, so walltimes already bound its charge;
    # a projection from slow early rows (cold start, rows at the length limit) could only stop it falsely.
    if model not in projected and model not in SWEEP and progress['pairs'] >= .25*progress['executable_pairs']:
        check_projection(schedule, model, accounting, progress, stopped, gpu_hours)
        projected.add(model)


def advance_extraction(schedule: dict, accounting: dict, stopped: set) -> None:
    """Queue afterany extraction with each complete shard graph, then the per-release analysis.

    A sweep release has no analysis of its own: whole_effects.combine reads it with the base releases.
    """
    schedule.setdefault('extraction', {})
    for model in MODELS:
        production = schedule['production'][model]
        ready = len(production) == SHARDS[model] or model in stopped
        if ready and model not in schedule['extraction']:
            dependencies = production or schedule['canaries'][model]
            schedule['extraction'][model] = submit_stage('extract', model=model,
                dependency='afterany:'+':'.join(j['id'] for j in dependencies))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
    if len(schedule['extraction']) == len(MODELS) and not schedule.get('analysis') and not SWEEP:
        schedule['analysis'] = submit_stage('analyze', dependency='afterany:'+':'.join(
            j['id'] for j in schedule['extraction'].values()))
        driver.checkpoint(ROOT/'JOBS.json', schedule)


def source_extractions(schedule: dict) -> list[str]:
    """Extraction job IDs of every release named in SOURCES.json, for the combined analysis."""
    ids = set()
    for source in verified(ROOT/'SOURCES.json')['sources']:
        for root in map(Path, [source['root'], *source.get('topups', [])]):
            jobs = schedule if root == ROOT else load(root/'JOBS.json')
            require(source['model'] in jobs['extraction'], f'{root.name} has no {source["model"]} extraction yet')
            ids.add(jobs['extraction'][source['model']]['id'])
    return sorted(ids)


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
                      'gpu_ceiling': GPU_CEILING, 'cpu_ceiling': CPU_CEILING, 'time': time.time(),
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
            final = [schedule['analysis']] if schedule.get('analysis') else list(schedule['extraction'].values())
            if len(schedule['extraction']) == len(MODELS) and all(terminal(accounting.get(j['id'], {})) for j in final):
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
            path = canary_path(args.model, args.dose or 'main')
            if not path.exists():
                publish(path, {'model': args.model, 'dose': args.dose or 'main',
                    'job': os.environ['SLURM_JOB_ID'], 'gate': {'passed': False, 'fluency_failures': [],
                    'software_failures': [f'{type(error).__name__}: {error}']}})
        raise


def validate_stage() -> None:
    """Run the allocated study suite, nominations and planning before publishing qualification."""
    native_quota = work_remaining_bytes()
    require(native_quota > 14*2**30, 'WORK reserve is insufficient for the bounded whole experiment')
    preparation = prepare()
    nomination_proof = audit_nominations()
    # The suite tests the default GPT/Qwen3.6 configuration plus explicit multi-model cases, so it runs
    # in a fresh interpreter without this release's model selection.
    settings = ('WHOLE_MODELS', 'WHOLE_GPU_CEILING', 'WHOLE_CPU_CEILING', 'WHOLE_CUTOFF', 'WHOLE_SWEEP',
                'WHOLE_SHARDS')
    env = {k: v for k, v in os.environ.items() if k not in settings}
    suite = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-v', '-s', str(REPO/'tests/study'),
                            '-t', str(REPO/'tests/study')], env=env, capture_output=True, text=True)
    sys.stderr.write(suite.stderr)
    ran = re.search(r'^Ran (\d+) tests?', suite.stderr, re.MULTILINE)
    tests = int(ran.group(1)) if ran else 0
    if suite.returncode != 0 or not tests:
        publish(ROOT/'CPU_VALIDATION.json', {'passed': False, 'tests': tests, 'job': os.environ['SLURM_JOB_ID']})
        raise ValueError('allocated whole study suite failed')
    publish(ROOT/'CPU_VALIDATION.json', {'passed': True, 'tests': tests,
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
        doses = [SWEEP[model]] if model in SWEEP else ['main', 'half']
        if not manifest_path(model, doses[0]).exists():
            continue  # preparation failed for this model; recorded in CPU_VALIDATION.json
        for dose in doses:
            for cell in manifest(model, dose)['cells']:
                record = {key: cell[key] for key in ('fold', 'class', 'operation', 'qualified',
                    'activate', 'deactivate', 'retained_rd_fraction', 'replaced_deactivation_pairs',
                    'activation_changed', 'mean_native_frequency')}
                if model == 'gpt' and cell['operation'] == 'promote' and (dose, cell['fold'], cell['class']) in references:
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
    parser.add_argument('command', choices=('prepare', 'canary', 'run', 'extract', 'analyze', 'combine', 'historical',
                                            'validate', 'monitor', 'launch', 'produce'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dose', choices=tuple(DOSES))
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
            if model in SWEEP:
                # A sweep dose has no canary: its shards wait only for the allocated validation.
                admit_shard_plan(schedule, model, {}, set())
                for index in range(SHARDS[model]):
                    schedule['production'][model].append(submit_stage('run', model=model, dose=SWEEP[model],
                        shard=[index, SHARDS[model]], dependency='afterok:'+validation['id']))
                    driver.checkpoint(ROOT/'JOBS.json', schedule)
                continue
            if (ROOT/model/'CANARY_DECISION.json').exists():
                continue  # a recorded decision replaces the canary; production follows `produce`
            schedule['canaries'][model].append(submit_stage('canary', model=model,
                dose=PLAN[model].get('start_dose', 'main'), dependency='afterok:'+validation['id']))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
        advance_extraction(schedule, {}, set())
        if (ROOT/'SOURCES.json').exists():
            schedule['analysis'] = submit_stage('combine', dependency='afterany:'+':'.join(source_extractions(schedule)))
            driver.checkpoint(ROOT/'JOBS.json', schedule)
        schedule['monitor'] = submit_stage('monitor', dependency='afterany:'+validation['id'])
        driver.checkpoint(ROOT/'JOBS.json', schedule)
        driver.log({'event': 'whole-accepted', 'jobs': {j['command']+':'+str(j['model']): j['id'] for j in all_jobs(schedule)}})
        return
    if args.command == 'produce':
        produce(args.model)
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
    elif args.command in ('analyze', 'combine'):
        if not args.cpu_worker:
            return cpu_stage(args.command, args.model)
        from .whole_effects import analyze, combine
        analyze() if args.command == 'analyze' else combine(ROOT/'SOURCES.json')
    elif args.command == 'historical':
        # Corrected re-analysis of the pulsed main and strong designs (the paper's earlier study).
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
