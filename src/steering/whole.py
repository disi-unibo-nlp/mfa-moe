"""Whole-reasoning steering: the manifests, the shards and the extraction of one release.

A release is a folder report/NAME holding RELEASE.json (sealed by steering.release) and everything its
stages write. The frozen manifest of each model owns the family order, the original prompts, the doses
and the seeds; no row is ever redrawn. Run each stage as a Slurm job from the release's code snapshot
(scripts/steering/launch.sh submits them):

  python -m steering.whole plan --release R                   what to run: the snapshot and the shards per model
  python -m steering.whole prepare --release R                CPU: publish the manifests of every model
  python -m steering.whole canary --release R --model M       GPU: the fluency canary of a base release
  python -m steering.whole run --release R --model M --shard I/N    GPU: the rows of one shard
  python -m steering.whole extract --release R --model M      CPU: replay and score the finished rows
  python -m steering.whole combine --release R                CPU: the bounded analysis of SOURCES.json
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import math
import os
from pathlib import Path
import time

import numpy as np

from store import CLASSES, REPO, check_files, digest, file_sha, load, publish, require, verified
from steering import pulse as driver
from steering.selection import select_pairs

# The models in the order the tables list them.
MODELS = ('gpt', 'qwen36', 'qwen35', 'qwen330b', 'gemma', 'glm', 'nemotron', 'lfm25')
# Steering strength levels as fractions of SteerMoE's published counts ('main' = full).
DOSES = {'main': 1., 'half': .5, 'quarter': .25}
# Largest generation of a canary row; GLM decodes about four times slower than the others.
CANARY_TOKENS = {'glm': 16384}
# Bootstrap draws and seed shared by every whole-reasoning interval.
DRAWS, SEED = 10000, 20261015
# No GPU row starts after this time; releases sealed without a cutoff stop at 2026-10-23 05:30 Europe/Rome.
DEFAULT_CUTOFF = 1792726200


def manifest_path(model: str, dose: str, root: Path) -> Path:
    """The frozen manifest of one model and dose; main and half keep their original names."""
    return root/model/('WHOLE.json' if dose == 'main' else f'WHOLE_{dose.upper()}.json')


def canary_path(root: Path, model: str, dose: str) -> Path:
    """The canary receipt of one model and dose; main and half keep their original names."""
    return root/model/('CANARY.json' if dose == 'main' else f'CANARY_{dose.upper()}.json')


def sweep_doses(root: Path) -> dict:
    """Model -> the one dose a sweep, complement or outcome release runs (empty for a base release)."""
    return verified(root/'RELEASE.json').get('sweep', {})


def link_receipt(root: Path) -> Path:
    """The efficiency-link receipt whose native signs pick the efficiency-directed classes.

    Releases up to r18 use the seven-model pre-registration; a relink release names the eight-model
    receipt (routing.efficiency_link) in RELINK.json, in which LFM2.5 has its own signs.
    """
    relink = root/'RELINK.json'
    return Path(verified(relink)['link']) if relink.exists() else driver.DRIVER/'EFFICIENCY_LINK_PREREG.json'


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
def directions(model: str, link: Path) -> dict:
    """Classes to promote (tied to shorter reasoning) and suppress (tied to longer reasoning).

    Signs come from the efficiency-link receipt on native reasoning only. A model outside that receipt
    (LFM2.5 in the seven-model pre-registration) uses the receipt's pooled cross-model signs.
    """
    prereg = verified(link)
    rows = ([cell for cell in prereg['cells'] if cell['model'] == model] if model in prereg['models']
            else prereg['pooled'])
    signs = {row['class']: row['tok_reason']['sign'] for row in rows}
    return {'promote': tuple(c for c in CLASSES if signs[c] == '-'),
            'suppress': tuple(c for c in CLASSES if signs[c] == '+'),
            'source': 'model' if model in prereg['models'] else 'pooled'}


def efficiency_directed(model: str, operation: str, class_name: str | None, link: Path) -> bool:
    """Whether a cell enters the primary efficiency contrast."""
    return class_name in directions(model, link)['promote' if operation == 'promote' else 'suppress']


def verify_release(root: Path) -> dict:
    """Check the archived code and the pinned inputs of a release; stages run only from its snapshot.

    Returns:
        The sealed release receipt.
    """
    release = verified(root/'RELEASE.json')
    require(Path(release['release_root']).resolve() == REPO, 'whole stage must run from its materialized release')
    require((REPO/'RELEASE_COMMIT').read_text().strip() == release['commit'], 'release commit marker differs')
    check_files(release['code_files'])
    check_files(release['inputs'])
    return release


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
    from steering.selection import layer_matched_controls
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
                   inputs: dict, link: Path, *, sweep: bool = False) -> dict:
    """Freeze all planned rows, including unavailable cells, in complete family blocks.

    Args:
        model, dose: Model identity and steering strength (see DOSES).
        nominations: Discovery-only global risk-difference receipt.
        b: Original study profile binding.
        natives: Original main-design native assignments.
        inputs: Family to original prompt token list.
        link: The efficiency-link receipt that picks the efficiency-directed classes.
        sweep: Strength-sweep manifest: efficiency-directed cells only, no native rows (they are the same
            computation as the base release's) and no audit captures.

    Returns:
        Manifest with planned row counts, actual cell counts and unique UIDs.
    """
    from steering.runtime import remaining
    profile = b['models'][model]
    suppressed = directions(model, link)['suppress']
    promoted = directions(model, link)['promote'] if sweep else CLASSES
    cells = [cell_actions(n, profile, model, dose, operation) for n in nominations['records']
             for operation in ('promote', 'suppress')
             if n['class'] in (promoted if operation == 'promote' else suppressed)]
    by_cell = {(c['fold'], c['class'], c['operation']): c for c in cells}
    require(all(c['qualified'] for c in cells if efficiency_directed(model, c['operation'], c['class'], link)),
            'every efficiency-directed cell must be nominated')
    rows = []
    for native in sorted(natives, key=lambda row: row['family']):
        prompt = inputs[native['family']]
        allowance = remaining(profile, prompt, [])
        base = {key: native[key] for key in ('model', 'family', 'benchmark', 'fold', 'seed', 'profile_sha256')}
        base.update(source={**native['source'], 'prefix_tokens': 0}, original_prompt=True,
                    prompt_sha256=digest(prompt), whole=True, prompt_scope=True,
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
    value = {'model': model, 'dose': dose, 'scope': 'prompt-and-generated-reasoning',
             'nominations_sha256': nominations['sha256'], 'cells': cells, 'records': rows,
             'planned': expected, 'executable': sum(r['nomination_status'] == 'nominated' for r in rows),
             'audit_units': audit_units, 'outcomes_used_for_selection': False}
    # Base manifests keep their original keys, so rebuilt r7/r8 manifests stay byte-identical.
    return {**value, 'design': 'efficiency-sweep'} if sweep else value


def prepare(root: Path) -> dict:
    """Publish the manifests of every model of the release (CPU job), then check the rebuilt nominations."""
    require(os.environ.get('SLURM_JOB_ID'), 'whole discovery requires an allocated CPU job')
    release = verify_release(root)
    b = driver.binding()
    results = {}
    for model in release['models']:
        try:
            results.update(prepare_model(root, b, model))
        except Exception as error:
            # One infeasible model must not block the others; its shards then stop on the missing manifest.
            results[model] = {'prepared': False, 'error': f'{type(error).__name__}: {error}'}
    require(any(':' in key for key in results), 'no whole model could be prepared')
    publish(root/'PREPARED.json', results)
    audit_nominations(root)
    return results


def prepare_model(root: Path, b: dict, model: str) -> dict:
    """Derive one model's nominations and publish its manifests.

    A base release publishes the main and half-dose manifests; a sweep release publishes the one sweep
    dose and selects it (no canary). A top-up or resume release (TOPUP.json / RESUME.json) publishes
    the rows it takes over from its parent, with the parent's dose and no canary.
    """
    from steering.outcome.manifest import outcome_manifest
    builders = {'TOPUP.json': topup_manifest, 'RESUME.json': resume_manifest, 'OUTCOME.json': outcome_manifest,
                'COMPLEMENT.json': complement_manifest}
    sweep = sweep_doses(root)
    present = [name for name in builders if (root/name).exists()]
    require(len(present) <= 1,
            'a release is either a top-up, a resume or an outcome-steering release (or a complement)')
    if present:
        spec = verified(root/present[0])['models'][model]
        return {model+':'+sweep[model]: builders[present[0]](root, model, spec)}
    from steering.nominations import derive_whole
    from steering.runtime import prompt_prefix
    folder = root/model
    link = link_receipt(root)
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
    for dose in ([sweep[model]] if model in sweep else ['main', 'half']):
        value = publish(manifest_path(model, dose, root),
                        build_manifest(model, dose, nominations, b, natives, inputs, link, sweep=model in sweep))
        results[model+':'+dose] = {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}
        if model in sweep:
            publish(folder/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                             'sweep': True})
    return results


def topup_manifest(root: Path, model: str, spec: dict) -> dict:
    """Publish the never-started rows of a base release's failed shards as this release's manifest.

    A top-up release reruns nothing: rows with any saved state in the parent store (complete, output or
    partial) stay with the parent, where an unfinished row remains an unknown outcome. The rows keep their
    parent UIDs, seeds and actions, so whole_effects joins them to the parent release by UID.

    Args:
        root: The folder of this release.
        model: Model whose base release lost shards.
        spec: {'parent': parent release root, 'shards': [[index, count], ...] of the failed jobs, or
            'all' once every parent shard has ended, and optionally 'siblings': earlier top-up roots of
            the same parent, all of whose jobs have ended; a row started there stays there}.
    """
    dose, parent = sweep_doses(root)[model], Path(spec['parent'])
    base = verified(manifest_path(model, dose, parent))
    stores = [driver.Store(parent/'stores'/model/dose/'units')]
    for sibling in map(Path, spec.get('siblings', [])):
        require(verified(manifest_path(model, dose, sibling))['parent_manifest_sha256'] == base['sha256'],
                'sibling top-up has another parent')
        stores.append(driver.Store(sibling/'stores'/model/dose/'units'))
    families = sorted({row['family'] for row in base['records']})
    failed = set(families) if spec['shards'] == 'all' else {
        family for i, family in enumerate(families) for index, count in spec['shards'] if i % count == index}
    rows = [row for row in base['records']
            if row['family'] in failed and not any(store.state(row['uid']) for store in stores)]
    require(rows, 'top-up has no never-started rows')
    kept = {k: v for k, v in base.items() if k not in ('sha256', 'records', 'planned', 'executable')}
    value = publish(manifest_path(model, dose, root), {
        **kept, 'records': rows, 'planned': len(rows),
        'executable': sum(row['nomination_status'] == 'nominated' for row in rows), 'design': 'top-up',
        'parent_root': str(parent), 'parent_manifest_sha256': base['sha256'], 'failed_shards': spec['shards'],
        'siblings': spec.get('siblings', [])})
    publish(root/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                         'sweep': True, 'topup': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}


def complement_manifest(root: Path, model: str, spec: dict) -> dict:
    """Publish the suppress cells a base release left out, so every state has both operations.

    A base release suppresses only the classes tied to longer reasoning (directions). This release
    suppresses the other classes at the parent's dose, for the per-cell effects only: the cells are not
    efficiency-directed, so they never enter the primary or the dose response. Each row copies its
    family's native row of the parent (seed, prompt, allowance) and is compared with it by family, like
    a sweep cell; the experts come from the parent's nominations through the same cell_actions.

    Args:
        root: The folder of this release.
        model: Model whose base release lacks the cells.
        spec: {'parent': root holding the base manifest of this model and dose and its nominations}.
    """
    dose, parent = sweep_doses(root)[model], Path(spec['parent'])
    base = verified(manifest_path(model, dose, parent))
    nominations = verified(parent/model/'WHOLE_NOMINATIONS.json')
    require(nominations['sha256'] == base['nominations_sha256'], 'parent nominations differ from its manifest')
    profile = driver.binding()['models'][model]
    classes = tuple(c for c in CLASSES if c not in directions(model, link_receipt(root))['suppress'])
    cells = [cell_actions(n, profile, model, dose, 'suppress') for n in nominations['records']
             if n['class'] in classes]
    by_cell = {(c['fold'], c['class']): c for c in cells}
    natives = [row for row in base['records'] if row['arm'] == 'native']
    rows = []
    for native in sorted(natives, key=lambda row: row['family']):
        common = {k: v for k, v in native.items() if k not in ('uid', 'audit_capture')}
        for class_name in classes:
            cell = by_cell[native['fold'], class_name]
            for role in ('target', 'random'):
                rows.append({**common, 'arm': role+'_suppress', 'class': class_name, 'operation': 'suppress',
                             'actions': cell[role],
                             'steered_native_frequency': cell.get('mean_native_frequency', {}).get(role),
                             'nomination_status': 'nominated' if cell['qualified'] else 'dropped_availability',
                             'audit_capture': False, 'complement': True})
    for row in rows:
        row['uid'] = digest(['whole-complement-v1', nominations['sha256'], row])
    expected = 2*len(natives)*len(classes)
    require(len(rows) == expected and len({r['uid'] for r in rows}) == expected, 'complement row count/UID differs')
    kept = {k: v for k, v in base.items() if k not in ('sha256', 'records', 'planned', 'executable', 'cells')}
    value = publish(manifest_path(model, dose, root), {
        **kept, 'cells': cells, 'records': rows, 'planned': expected,
        'executable': sum(row['nomination_status'] == 'nominated' for row in rows), 'audit_units': 0,
        'design': 'complement', 'complement_classes': list(classes), 'parent_root': str(parent),
        'parent_manifest_sha256': base['sha256']})
    publish(root/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'], 'canary_job': None,
                                         'sweep': True, 'complement': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}


def resume_manifest(root: Path, model: str, spec: dict) -> dict:
    """Publish the interrupted rows of a base release as this release's manifest.

    A row interrupted at a job walltime or crash holds only a PARTIAL.json, so its outcome is
    unknown. A resume release continues it from a copy of that file, re-prefilling prompt + saved
    tokens through the same hooks as the pulsed study did: UIDs, seeds and actions stay the parent's
    and nothing is redrawn. The source stores are never written (see seed_resume); whole_effects
    joins the outcomes by UID.

    Args:
        root: The folder of this release.
        model: Model whose base release has interrupted rows.
        spec: {'parent': root holding the base manifest of this model and dose, 'sources': roots
            whose stores may hold partial rows of that manifest (the parent and its top-ups)}.
    """
    dose, parent = sweep_doses(root)[model], Path(spec['parent'])
    base = verified(manifest_path(model, dose, parent))
    roots = list(map(Path, spec['sources']))
    require(parent in roots, 'resume sources must include the parent')
    for source in roots:
        if source != parent:
            topup = verified(manifest_path(model, dose, source))
            require(topup['parent_manifest_sha256'] == base['sha256'],
                    'resume source belongs to another manifest')
    stores = {str(source): driver.Store(source/'stores'/model/dose/'units') for source in roots}
    rows, sources = [], {}
    for row in base['records']:
        states = {root: store.state(row['uid']) for root, store in stores.items()}
        saved = {root: state for root, state in states.items() if state}
        if not any(state['status'] == 'partial' for state in saved.values()):
            continue  # never started, or finished: nothing to continue
        # A row is continued from exactly one partial; any other saved state is a conflict.
        require(len(saved) == 1, f'interrupted row has state in several stores: {row["uid"]}')
        (root, state), = saved.items()
        require(state.get('assignment_sha256') == digest(row),
                f'partial belongs to another assignment: {row["uid"]}')
        rows.append(row)
        sources[row['uid']] = root
    require(rows, 'resume has no interrupted rows')
    kept = {k: v for k, v in base.items()
            if k not in ('sha256', 'records', 'planned', 'executable')}
    value = publish(manifest_path(model, dose, root), {
        **kept, 'records': rows, 'planned': len(rows),
        'executable': sum(row['nomination_status'] == 'nominated' for row in rows),
        'design': 'resume', 'parent_root': str(parent), 'parent_manifest_sha256': base['sha256'],
        'resume_sources': sources})
    publish(root/model/'SELECTED.json', {'dose': dose, 'manifest_sha256': value['sha256'],
                                         'canary_job': None, 'sweep': True, 'resume': True})
    return {key: value[key] for key in ('sha256', 'planned', 'executable', 'scope')}


def manifest(root: Path, model: str, dose: str | None = None) -> dict:
    """Read the selected manifest; selection is fixed by the passing canary (or the sweep)."""
    if dose is None:
        selected = root/model/'SELECTED.json'
        dose = verified(selected)['dose'] if selected.exists() else 'main'
    return verified(manifest_path(model, dose, root))


def bound_store(root: Path, model: str, dose: str, canary: bool = False) -> driver.Store:
    """Reject any nonempty unbound store before the engine is loaded."""
    value = manifest(root, model, dose)
    release = verified(root/'RELEASE.json')
    require(Path(release['release_root']).resolve() == REPO, 'whole code is outside its immutable materialized release')
    folder = root/'stores'/model/dose/('canary' if canary else 'units')
    identity = {'manifest_sha256': value['sha256'], 'release_sha256': release['sha256'], 'canary': canary}
    if (folder/'EXECUTION.json').exists():
        require({k: v for k, v in verified(folder/'EXECUTION.json').items() if k != 'sha256'} == identity,
                'whole store belongs to a different release')
    else:
        require(not folder.exists() or not any(folder.iterdir()), 'nonempty unbound whole store')
        publish(folder/'EXECUTION.json', identity)
    return driver.Store(folder)


def check_saved_state(value: dict, rows: list[dict], store: driver.Store,
                      canary: bool = False) -> None:
    """Refuse a shard that would redraw or overwrite saved work.

    Execution is fresh: no row of the shard may have saved state. The one exception is a resume
    release, which continues its own seeded partial rows, provided each is a PARTIAL of this
    manifest's assignment (never a complete, output or foreign state).

    Args:
        value: The selected manifest.
        rows: This shard's rows.
        store: The release's bound store.
        canary: Canary shards never resume.
    """
    resume = value.get('design') == 'resume' and not canary
    planned = {row['uid'] for row in value['records']}
    for row in rows:
        state = store.state(row['uid'])
        if state is None:
            continue
        folder = store.folder(row['uid'])
        if resume and row['uid'] in planned and state['status'] == 'complete':
            continue  # finished by an earlier job of this resume release; the engine skips it
        require(resume and row['uid'] in planned and state['status'] == 'partial' and
                state.get('assignment_sha256') == digest(row) and
                not (folder/'OUTPUT.json').exists() and not (folder/'RESULT.json').exists(),
                f'whole execution is fresh only; preserve partials ({row["uid"]}, {state["status"]})')


def seed_resume(root: Path, model: str, value: dict, rows: list[dict], store: driver.Store,
                shard: tuple) -> dict:
    """Copy each row's source PARTIAL.json into the bound store and publish what was copied.

    The source stores are only read. A target that already holds a PARTIAL (a restarted shard) must
    be the source plus later progress, so tokens a previous job generated are never replaced.

    Args:
        root: The folder of this release.
        model: Model of the shard.
        value: The selected resume manifest; 'resume_sources' maps each UID to its partial's root.
        rows: This shard's rows.
        store: The release's bound store.
        shard: (index, count) of the shard.

    Returns:
        The published seeding receipt.
    """
    require(value.get('design') == 'resume', 'only a resume release seeds saved partials')
    seeded = []
    for row in rows:
        uid = row['uid']
        origin = Path(value['resume_sources'][uid])/'stores'/model/value['dose']/'units'
        path = driver.Store(origin).folder(uid)/'PARTIAL.json'
        partial = load(path)
        require(partial['uid'] == uid and partial['assignment_sha256'] == digest(row),
                f'resume source is not this assignment\'s partial: {path}')
        saved = store.state(uid)
        if saved is None:
            driver.checkpoint(store.folder(uid)/'PARTIAL.json', partial)
        else:
            keys = ('uid', 'assignment_sha256', 'total_allowance')
            require(all(saved[key] == partial[key] for key in keys) and
                    saved['tokens'][:len(partial['tokens'])] == partial['tokens'] and
                    saved['segments'][:len(partial['segments'])] == partial['segments'],
                    f'seeded partial diverged from its source: {uid}')
        last = partial['segments'][-1] if partial['segments'] else {}
        seeded.append({'uid': uid, 'source': str(path), 'source_sha256': file_sha(path),
                       'history': len(partial['tokens']), 'drained': 'drained' in last})
    index, count = shard
    return publish(root/model/'resume'/f'shard-{index}-{count}.json', {
        'model': model, 'dose': value['dose'], 'manifest_sha256': value['sha256'],
        'shard': [index, count], 'rows': seeded})


def canary_rows(value: dict, link: Path) -> list[dict]:
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
    require('Verify' in directions(model, link)['promote'], 'canary promote class differs')
    chosen_class = directions(model, link)['suppress'][0]
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






def admission_order(rows: list[dict], link: Path) -> list[dict]:
    """Order all primary and native rows before other promote rows, without barriers."""
    return sorted(rows, key=lambda r: (
        int(r['operation'] == 'promote' and not efficiency_directed(r['model'], 'promote', r['class'], link)),
        r['family'], r['arm'], r['class'] or '',
    ))


def run(root, model, shard, deadline, *, max_inflight=64, canary=False, dose=None):
    """Run one source-bound shard or canary on the existing hooked engine."""
    from steering.runtime import closure
    release = verify_release(root)
    link = link_receipt(root)
    value = manifest(root, model, dose)
    if not canary:
        selected = verified(root/model/'SELECTED.json')
        require(selected['manifest_sha256'] == value['sha256'], 'production manifest lacks a passing canary')
        # A sweep dose runs without a canary (breakdown is measured as an outcome); otherwise the passing
        # canary receipt qualifies the manifest.
        if not selected.get('sweep'):
            proof = verified(canary_path(root, model, value['dose']))
            require(proof['gate']['passed'] and proof['job'] == selected['canary_job'] and
                    proof['manifest_sha256'] == value['sha256'], 'production canary qualification differs')
    rows = canary_rows(value, link) if canary else driver.shard_rows(value['records'], shard)
    store = bound_store(root, model, value['dose'], canary)
    check_saved_state(value, rows, store, canary)
    cutoff = release.get('cutoff', DEFAULT_CUTOFF)
    if time.time() >= cutoff:
        # A shard that starts after the science cutoff leaves its rows unknown without loading the model.
        driver.log({'event': 'whole-shard-after-cutoff', 'model': model, 'shard': list(shard)})
        return None
    if value.get('design') == 'resume' and not canary:
        seed_resume(root, model, value, rows, store, shard)
    executable = admission_order([r for r in rows if r['nomination_status'] == 'nominated'], link)
    b = driver.binding()
    llm, kwargs, load_seconds = driver.hooked_llm(b, model, max_inflight)
    try:
        contract = driver.contract_for(b, model, executable)
        reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract})
        require(len(reports) == 2, 'whole requires both tensor-parallel ranks')
        tokenizer = llm.get_tokenizer()
        engine = driver.Engine(llm, store, min(deadline, cutoff), hooked=True, max_inflight=max_inflight,
                               checkpoint_seconds=120, drain_seconds=120)
        # Canary rows share one per-model token cap so the relative gates compare like with like
        # inside the canary walltime; production keeps the full model-card allowance.
        build = driver.intervention_builder(b, model, tokenizer, closure(tokenizer, model),
                                            limit=CANARY_TOKENS.get(model, 32768) if canary else None)
        metrics = engine.run(executable, build, driver.intervention_finisher(store, b, model),
                             stop_execution=lambda: (root/'STOP.json').exists() or (root/model/'STOP.json').exists())
        validation = llm.collective_rpc('fs_validation')
        llm.collective_rpc('fs_end')
    finally:
        llm.llm_engine.engine_core.shutdown()
    result = {'model': model, 'dose': value['dose'], 'manifest_sha256': value['sha256'],
              'job': os.environ['SLURM_JOB_ID'], 'shard': list(shard), 'load_seconds': load_seconds,
              'metrics': metrics, 'native_validation': validation[0], **driver.shard_completion(store, rows)}
    if canary:
        result['gate'] = fluency(store, rows, require_complete=True, censor_unfinished=True)
        publish(canary_path(root, model, value['dose']), result)
        if result['gate']['passed']:
            publish(root/model/'SELECTED.json', {'dose': value['dose'], 'manifest_sha256': value['sha256'],
                                                'canary_job': result['job']})
        else:
            raise SystemExit(3 if result['gate']['fluency_failures'] and not result['gate']['software_failures'] else 2)
    else:
        publish(root/model/'jobs'/(result['job']+'.json'), result)
    return result


def audit_nominations(root: Path) -> dict:
    """Check rebuilt manifests against the saved DP selections and record every cell.

    Returns:
        Source-bound per-cell audit, including actual retained RD and manipulation
        frequencies. A mismatch in any GPT promote optimum fails qualification.
    """
    release = verified(root/'RELEASE.json')
    sweep = release.get('sweep', {})
    proposal = verified(Path(release['nomination_reference']))
    references = {(r['dose'], r['fold'], r['class']): r for r in proposal['records']}
    records = []
    for model in release['models']:
        doses = [sweep[model]] if model in sweep else ['main', 'half']
        if not manifest_path(model, doses[0], root).exists():
            continue  # preparation failed for this model; recorded in PREPARED.json
        for dose in doses:
            for cell in manifest(root, model, dose)['cells']:
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
    return publish(root/'NOMINATION_AUDIT.json', {
        'records': records, 'dp_reference_sha256': proposal['sha256'],
        'gpt_promote_dp_matches': sum(r.get('saved_dp_match', False) for r in records),
        'outcomes_used': False,
    })


def plan(root: Path) -> None:
    """Tell the launcher what to submit: the release's code snapshot, then 'MODEL SHARDS' per model."""
    release = verified(root/'RELEASE.json')
    print(release['release_root'])
    for model in release['models']:
        if release.get('shards', {}).get(model):
            print(model, release['shards'][model])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='steering.whole', description=__doc__.split('\n\n')[0])
    parser.add_argument('command', choices=('plan', 'prepare', 'canary', 'run', 'extract', 'combine'))
    parser.add_argument('--release', type=Path, required=True, help='the release folder, report/NAME')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--dose', choices=tuple(DOSES), help='canary only; default: the model\'s declared start dose')
    parser.add_argument('--shard', type=driver.parse_shard, default=(0, 1))
    args = parser.parse_args(argv)
    root = args.release.resolve()
    if args.command == 'plan':
        return plan(root)
    require(os.environ.get('SLURM_JOB_ID'), 'whole stages run inside Slurm jobs')
    if args.command == 'prepare':
        driver.log({'event': 'prepare-whole', 'results': prepare(root)})
    elif args.command in ('canary', 'run'):
        require(os.environ.get('SLURM_JOB_GPUS'), 'whole inference requires allocated GPUs')
        deadline = driver.deadline_from_slurm(margin=120)
        if args.command == 'run':
            run(root, args.model, args.shard, deadline)
            return
        dose = args.dose or ('half' if args.model == 'gpt' else 'main')  # GPT's full dose broke down in its canary
        try:
            run(root, args.model, (0, 1), deadline, canary=True, dose=dose)
        except BaseException as error:
            # A software failure still leaves a canary receipt, so it is never mistaken for a pass.
            path = canary_path(root, args.model, dose)
            if not path.exists():
                publish(path, {'model': args.model, 'dose': dose, 'job': os.environ['SLURM_JOB_ID'],
                               'gate': {'passed': False, 'fluency_failures': [],
                                        'software_failures': [f'{type(error).__name__}: {error}']}})
            raise
    else:
        verify_release(root)
        from steering import whole_effects
        if args.command == 'extract':
            whole_effects.extract(root, args.model)
        else:
            whole_effects.combine(root)


if __name__ == '__main__':
    main()
