"""Bounded, pre-registered score-override rescue for the final routing study.

The main study used one layer and a short hard-selection pulse.  This module
constructs a separate manifest that keeps the same frozen prefixes and
nominations but applies the reference-style router-score override across a
full continuation window, either at the pre-registered single layer or at all
three depth-region winners.  No row or threshold is selected from outcomes.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from moe_exp.io import BENCHMARKS, CLASSES, MODELS, digest, load, publish, require, verified

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/final-routing-20261015-r1')
DRIVER = ROOT/'driver'
RESCUE_MODELS = ('gpt', 'qwen36')
RESCUE_SEED = 'final-routing-score-override-rescue-v1'
RESCUE_COUNT = 64
RESCUE_WINDOW = 512
RESCUE_EPSILON = .01


def _rank(seed, *parts):
    return digest([seed, 'rescue-family', *parts])


def select_families(allocation, *, count=RESCUE_COUNT, seed=RESCUE_SEED):
    """Choose a benchmark-balanced family panel without reading outcomes."""
    records = records_by_key(allocation)
    by_benchmark = defaultdict(list)
    for record in records:
        by_benchmark[record['benchmark']].append(record)
    require(set(by_benchmark) == set(BENCHMARKS), 'allocation benchmark roster differs')
    chosen = []
    for benchmark in BENCHMARKS:
        ordered = sorted(by_benchmark[benchmark], key=lambda r: _rank(seed, benchmark, r['family']))
        require(len(ordered) >= 4, 'rescue needs at least four families per benchmark')
        chosen.extend(ordered[:4])
    remaining = count-len(chosen)
    require(remaining >= 0, 'rescue family count is below the benchmark floor')
    # Largest-remainder allocation over the six benchmark populations, with a
    # digest tie-break so the panel is reproducible but not benchmark-selected.
    ideal = {b: count*len(by_benchmark[b])/len(records) for b in BENCHMARKS}
    quota = {b: 4 for b in BENCHMARKS}
    for _ in range(count-sum(quota.values())):
        b = max(BENCHMARKS, key=lambda x: (ideal[x]-quota[x], _rank(seed, 'quota', x)))
        quota[b] += 1
    chosen = []
    for b in BENCHMARKS:
        ordered = sorted(by_benchmark[b], key=lambda r: _rank(seed, b, r['family']))
        chosen.extend(ordered[:quota[b]])
    require(len(chosen) == count and len({(r['benchmark'], r['family']) for r in chosen}) == count,
            'rescue family panel is not unique')
    return {'records': sorted(chosen, key=lambda r: (r['benchmark'], r['family'])),
            'seed': seed, 'count': count, 'quotas': quota}


def records_by_key(allocation):
    return list(allocation.get('records', allocation))


def _nominations(model):
    return verified(ROOT/model/'COUNT_NOMINATIONS.json')['records']


def _selected_layers(model):
    value = verified(DRIVER/model/'ASSIGNMENTS.json')
    return {(r['fold'], r['class']): r for r in value['selection']}


def _high(nomination, key):
    sets = nomination.get(key) or []
    require(len(sets) >= 3 and sets[2], 'rescue nomination lacks its high-count set')
    return sorted({int(e) for e in sets[2]})


def _actions_for(nominations, selected, fold, cls, regime):
    """Reuse original count nominations at the frozen single or triple layers."""
    by_depth = {n['depth_third']: n for n in nominations
                if n['fold'] == fold and n['class'] == cls and n['status'] == 'nominated'}
    if regime == 'single':
        chosen = selected[(fold, cls)]
        n = by_depth.get(chosen['depth_third'])
        require(n is not None and n['layer'] == chosen['layer'], 'selected rescue layer differs')
        candidates = [n]
    elif regime == 'triple':
        candidates = [by_depth.get(d) for d in (0, 1, 2)]
        require(all(n is not None for n in candidates), 'rescue class lacks a depth-region nomination')
    else:
        raise ValueError('unknown rescue layer regime')
    actions = []
    for n in candidates:
        actions.append({'operator': 'score_override', 'direction': None,
                        'experts': _high(n, 'expert_sets'), 'layer': int(n['layer']),
                        'pulse_tokens': RESCUE_WINDOW, 'start': 0, 'epsilon': RESCUE_EPSILON})
    return actions


def _native_layers(nominations, selected, fold):
    out = set()
    for c in CLASSES:
        chosen = selected.get((fold, c))
        if chosen:
            out.add(int(chosen['layer']))
        for n in nominations:
            if n['fold'] == fold and n['class'] == c and n['status'] == 'nominated':
                out.add(int(n['layer']))
    return sorted(out)


def build(model, allocation, *, seed=RESCUE_SEED, count=RESCUE_COUNT):
    """Build one model's immutable rescue manifest from frozen nominations."""
    panel = select_families(allocation, count=count, seed=seed)
    native_rows = {r['family']: r for r in verified(DRIVER/model/'ASSIGNMENTS.json')['records']
                   if r['arm'] == 'native' and r['prefix_status'] == 'enrolled'}
    nominations = _nominations(model)
    selected = _selected_layers(model)
    rows = []
    for slot in panel['records']:
        family, benchmark, fold = slot['family'], slot['benchmark'], slot['fold']
        native = native_rows.get(family)
        require(native is not None and native['benchmark'] == benchmark and native['fold'] == fold,
                'rescue family lacks its native assignment')
        base = {k: native[k] for k in ('benchmark', 'family', 'fold', 'model', 'seed', 'source',
                                       'prefix_status', 'profile_sha256')}
        common = {**base, 'horizon': RESCUE_WINDOW, 'pulse_tokens': RESCUE_WINDOW,
                  'native_uid': native['uid'], 'nomination_status': 'nominated'}
        native_row = {**common, 'arm': 'native', 'class': None, 'depth_third': None,
                      'action': None, 'actions': None, 'dose': 'rescue-native',
                      'layer_regime': 'native', 'capture_layers': _native_layers(nominations, selected, fold)}
        native_row['uid'] = digest([seed, model, benchmark, family, 'native', native['seed']])
        rows.append(native_row)
        repeat = {**native_row, 'arm': 'native_repeat', 'dose': 'rescue-native-repeat',
                  'seed': int(digest([seed, model, benchmark, family, 'native-repeat'])[:8], 16)}
        repeat['uid'] = digest([seed, model, benchmark, family, 'native-repeat', repeat['seed']])
        rows.append(repeat)
        for c in CLASSES:
            selected_c = selected.get((fold, c))
            require(selected_c is not None, 'rescue class lacks a selected layer')
            for regime in ('single', 'triple'):
                template = _actions_for(nominations, selected, fold, c, regime)
                for direction in ('activate', 'deactivate'):
                    for kind, key in (('target', 'expert_sets'), ('random', 'control_sets')):
                        actions = []
                        for action in template:
                            source = next(n for n in nominations
                                          if n['fold'] == fold and n['class'] == c
                                          and n['layer'] == action['layer'])
                            actions.append({**action, 'experts': _high(source, key),
                                            'direction': direction})
                        row = {**common, 'arm': kind+'_'+direction, 'class': c,
                               'depth_third': selected_c['depth_third'] if regime == 'single' else None,
                               'action': None, 'actions': actions,
                               'dose': 'rescue-score-override:'+regime, 'layer_regime': regime,
                               'capture_layers': sorted(a['layer'] for a in actions)}
                        row['uid'] = digest([seed, model, benchmark, family, c, regime, direction, kind,
                                             [(a['layer'], a['experts']) for a in actions]])
                        rows.append(row)
    require(len({r['uid'] for r in rows}) == len(rows), 'duplicate rescue assignment identity')
    rows.sort(key=lambda r: (r['family'], r['arm'], r['class'] or '', r['layer_regime'], r['uid']))
    counts = defaultdict(int)
    for row in rows:
        counts[row['arm']] += 1
    value = publish(DRIVER/model/'RESCUE.json', {
        'schema': 'final-routing-score-override-rescue-v1', 'model': model, 'seed': seed,
        'window': RESCUE_WINDOW, 'epsilon': RESCUE_EPSILON, 'families': panel,
        'arms': dict(sorted(counts.items())), 'rows': len(rows), 'outcomes_used': False,
        'selector': 'existing frozen COUNT nominations; no outcome-based retuning',
        'records': rows})
    return value


def plan(models=RESCUE_MODELS, *, count=RESCUE_COUNT):
    allocation = load(ROOT/'ALLOCATION.json')
    return {model: build(model, allocation, count=count)['sha256'] for model in models}


def generate(cohort_path: Path, model: str, deadline: float, *, max_inflight: int = 8, source_receipt=None) -> dict:
    """Generate only the frozen untouched Qwen roster, never resume positive tokens.

    Args:
        cohort_path: Required immutable cohort descriptor.
        model: The authorized Qwen model.
        deadline: Allocation deadline, additionally bounded by the science cutoff.
        max_inflight: At most eight concurrent requests.
        source_receipt: Optional explicit source, which must match the cohort.

    Returns:
        Complete/incomplete job receipt, retaining every assigned state.
    """
    from .cohort import assignments, identity, read
    from .driver import Store, bulk, execute_rows
    from .ownership import check_time, worker
    from .rescue_source import load_study_binding
    value = read(cohort_path)
    require(source_receipt is None or Path(source_receipt).resolve() == Path(value['source']['path']),
            'explicit source receipt differs from cohort source')
    require(model == 'qwen36' and 1 <= max_inflight <= 8, 'only Qwen completion with at most eight requests is authorized')
    deadline = min(deadline, check_time(value))
    control = Path(value['paths']['control'])
    for engineering_model in RESCUE_MODELS:
        gate = verified(control/'engineering'/engineering_model/'QUALIFIED.json')
        require(gate['status'] == 'complete' and gate['cohort_sha256'] == value['sha256'],
                'scientific generation requires both allocated engineering qualifications')
    qualification = verified(Path(value['paths']['qualification']))
    require(qualification['cohort_sha256'] == value['sha256'], 'saved evidence qualification differs')
    study = load_study_binding(Path(value['study_binding_path']))
    rows = [row for row in assignments(value, model) if row['uid'] in set(value['generation_uids'])]
    execution = identity(value, model, 'generation', value['generation_uids'])
    root = bulk(study)/model/'rescue'
    with worker(root, execution):
        publish(root/'EXECUTION_IDENTITY.json', execution)
        store = Store(root/'units')
        pending = {row['uid'] for row in rows if store.state(row['uid']) is None}
        result = execute_rows(study, model, rows, store, deadline, max_inflight=max_inflight,
                              design='rescue', execution=execution, pending_uids=pending)
        read(cohort_path)
        return result


def main(argv=None):
    """Expose cohort preparation and allocated stages through the maintained study."""
    import argparse
    import json
    from . import cohort
    parser = argparse.ArgumentParser(prog='moe_exp.study.rescue')
    parser.add_argument('command', choices=('cohort', 'qualify-saved', 'index', 'engineering-plan', 'engineering',
                                           'submit', 'status', 'analyze', 'worker', 'reader-admit'))
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--predecessor', type=Path)
    parser.add_argument('--classifier', type=Path)
    parser.add_argument('--model', choices=(*RESCUE_MODELS, 'all'))
    parser.add_argument('--stage')
    args = parser.parse_args(argv)
    if args.command == 'cohort':
        require(args.source_receipt is not None and args.classifier is not None, 'cohort needs source and deployed classifier')
        value = cohort.create(args.source_receipt, args.cohort, args.classifier, predecessor_path=args.predecessor)
    elif args.command == 'qualify-saved':
        value = stage_worker(args.cohort, 'validate', None)
    elif args.command == 'index':
        value = cohort.terminal_index(args.cohort)
    elif args.command in ('engineering-plan', 'engineering'):
        from . import engineering
        value = (engineering.plan(args.cohort) if args.command == 'engineering-plan'
                 else engineering.run(args.cohort, args.model))
    elif args.command in ('submit', 'status'):
        from .reservations import CohortSchedule
        schedule = CohortSchedule(args.cohort)
        value = schedule.submit(args.stage, args.model) if args.command == 'submit' else schedule.status()
    elif args.command == 'reader-admit':
        from .rescue_reader import admit_reader
        value = admit_reader(cohort.read(args.cohort))
    elif args.command == 'worker':
        value = stage_worker(args.cohort, args.stage, None if args.model == 'all' else args.model)
    else:
        value = stage_worker(args.cohort, 'analysis', None)
    print(json.dumps({'command': args.command, **{key: value[key] for key in
                      ('sha256', 'status', 'job', 'counts', 'totals', 'availability') if key in value}}, default=str))


def stage_worker(path: Path, stage: str, model: str | None) -> dict:
    """Run one exact accepted resource card, stopping on any software failure.

    Args:
        path: Frozen cohort descriptor.
        stage: Stage named by the accepted submission journal.
        model: Optional engineering/generation model.

    Returns:
        Actual terminal artifact; incomplete generation retains CLI exit two.
    """
    import os
    import subprocess
    import sys
    import time
    from .cohort import extract_terminal, identity, qualify_saved, read, terminal_index
    from .driver import deadline_from_slurm
    from .ownership import check_time, worker
    from .reservations import CohortSchedule
    value = read(path)
    schedule = CohortSchedule(path)
    card = schedule.card(stage, model)
    folder = schedule.root/'schedule'/(stage+'-'+(model or 'all'))
    accepted_path = folder/'ACCEPTED.json'
    # sbatch may start before the login-side readback has returned. Wait only
    # for that exact acceptance, never fabricate ownership inside the worker.
    for _ in range(60):
        if accepted_path.exists():
            break
        time.sleep(1)
    accepted = verified(accepted_path)
    require(accepted['job'] == os.environ['SLURM_JOB_ID'] and accepted['card'] == card, 'worker acceptance differs')
    if stage == 'reserve':
        stage, model = card['operation'], card['operation_model']
    check_time(value, reporting=stage == 'analysis')
    if stage == 'engineering':
        from .engineering import run
        return run(path, model)
    if stage == 'generate':
        result = generate(path, model, deadline_from_slurm())
        if result['status'] != 'complete':
            raise SystemExit(2)
        return result
    if stage in ('pilot', 'wave-0', 'wave-1'):
        from .rescue_reader import run_cohort
        result = run_cohort(path, stage, deadline_from_slurm())
        if result['status'] != 'complete':
            raise SystemExit(2)
        return result
    store, execution = schedule._store(card)
    with worker(store, execution):
        return _cpu_stage(path, stage, value, store)


def _cpu_stage(path: Path, stage: str, value: dict, store: Path) -> dict:
    """Run allocated validation, task preparation or protected final analysis."""
    import subprocess
    import sys
    from .cohort import extract_terminal, qualify_saved, terminal_index
    if stage == 'validate':
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                                 '-s', 'tests/study', '-q'], capture_output=True, text=True)
        publish(store/'TESTS.json', {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        require(result.returncode == 0, 'allocated study tests failed; downstream release halted')
        return qualify_saved(path)
    if stage == 'prepare':
        from .rescue_reader import cohort_tasks, freeze_reuse, freeze_waves
        for name in RESCUE_MODELS:
            extract_terminal(path, name)
        terminal_index(path)
        tasks = cohort_tasks(path)
        freeze_reuse(value, tasks)
        return freeze_waves(value, tasks)
    require(stage == 'analysis', 'unknown CPU cohort phase')
    return analyze_all(path)



def analyze_all(path: Path) -> dict:
    """Write separate corrected historical and rescue analyses and final exhibits.

    Args:
        path: Immutable cohort descriptor.

    Returns:
        Delivery receipt with artifact hashes and explicit manuscript checklist.
    """
    from .cohort import read
    from .effects import analyze
    from .export import causal_exhibits
    from .reservations import CohortSchedule
    value = read(path)
    root = Path(value['paths']['analysis'])
    results = {}
    for design, models in (('main', ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')),
                           ('strong', ('gpt', 'nemotron', 'qwen330b', 'gemma')), ('rescue', RESCUE_MODELS)):
        result = analyze(list(models), root/design, design, cohort=path if design == 'rescue' else None,
                         classifier=value['classifier'], closeout=design == 'rescue')
        causal_exhibits(root/design, result)
        results[design] = result['sha256']
    read(path)
    return publish(Path(value['paths']['control'])/'DELIVERY.json', {
        'cohort_sha256': value['sha256'], 'analyses': results, 'accounting': CohortSchedule(path).status(),
        'manuscript_checklist': ['LFM/readout integration', 'accepted text-judge corrections',
                                 'outstanding 49 page-review items', 'final PDF verification'],
        'canonical_manuscript_edited': False, 'remote_push': False,
    })


if __name__ == '__main__':
    main()
