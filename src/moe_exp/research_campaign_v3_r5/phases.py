"""Immutable delivery phases over the unchanged scientific execution identities."""
from collections import defaultdict
from pathlib import Path
from .contracts import Units, load, publish

PHASES = ('memory', 'central', 'sensitivity')


def assign(table):
    usage = defaultdict(set)
    for cell in table['cells']:
        if cell.get('uid'):
            phase = ('memory' if cell['study'] == 'memory' else
                     'central' if cell.get('threshold_index', 0) == 0 else 'sensitivity')
            usage[cell['uid']].add(phase)
        cell['phase'] = ('memory' if cell['study'] == 'memory' else
                         'central' if cell.get('threshold_index', 0) == 0 else 'sensitivity')
    for row in table['records']:
        row['phase'] = min(usage[row['uid']], key=PHASES.index)
        row['analysis_phases'] = sorted(usage[row['uid']], key=PHASES.index)
    return table


def ready(campaign, observed, want):
    """Score and deliver memory before dispatching central, then diagnostics."""
    from .controller import _stage_state, _success, _failure, terminal
    root = Path(campaign['root'])
    for phase_index, phase in enumerate(PHASES):
        finished = True
        for model in campaign['models']:
            path = root/model/'EXECUTION_SHARDS.json'
            if not path.exists():
                continue
            plan = load(path)
            indices = [i for i, p in enumerate(plan['phases']) if p == phase]
            for i in indices:
                want('execute', model, i, minutes=1440)
            if not all(_stage_state(root, 'execute', model, i) for i in indices):
                finished = False
                continue
            table = load(root/model/'TASKS.json')
            tasks = [r for r in table['records'] if r['phase'] == phase]
            units = Units(root/model/'executions')
            pending = [r['task_index'] for r in tasks if units.state(r['uid']) is None]
            # Only untouched original identities may be continued after a shard
            # reaches its deadline. A start receipt is never a redraw license.
            fragments = sorted((root/model/'execution_fragments').glob('*.json'))
            owned = {i for p in fragments for i in load(p)['indices']}
            missing = [i for i in pending if i not in owned]
            next_index = len(plan['shards'])+len(fragments)
            for i in missing:
                publish(root/model/'execution_fragments'/f'{next_index:06d}.json',
                        {'shard_index': next_index, 'indices': [i], 'phase': phase,
                         'only_proven_unstarted_frozen_units': True})
                next_index += 1
            fragments = [load(p) for p in sorted((root/model/'execution_fragments').glob('*.json'))
                         if load(p)['phase'] == phase]
            for fragment in fragments:
                want('execute', model, fragment['shard_index'], minutes=1440)
                if _failure(root,'execute',model,fragment['shard_index']):
                    for i in fragment['indices']:
                        row=table['records'][i]
                        if units.state(row['uid']) is None and units.begin(row,'terminal-untouched-disposition'):
                            units.commit(row,{'status':'upstream_unavailable','generation_started':False,
                                'reason':'bounded untouched-assignment recovery exhausted','redraw':False})
            pending = [r['task_index'] for r in tasks if units.state(r['uid']) is None]
            if pending or not all(_stage_state(root,'execute',model,p['shard_index']) for p in fragments):
                finished = False
                continue
            scoreplan = root/model/f'SCORE_SHARDS-{phase}.json'
            if not scoreplan.exists():
                publish(scoreplan, {'phase': phase, 'shards': [[r['task_index'] for r in tasks[i:i+256]]
                         for i in range(0, len(tasks), 256)], 'original_strict_contract': True})
            scoring = load(scoreplan)['shards']
            offset = (phase_index+1)*100000
            for i in range(len(scoring)):
                want('score', model, offset+i+1, minutes=180)
            if not all(_stage_state(root,'score',model,offset+i+1) for i in range(len(scoring))):
                finished = False
            if phase != 'memory':
                want('contexts', model, phase_index, minutes=180)
                if not _success(root,'contexts',model,phase_index):
                    if not _failure(root,'contexts',model,phase_index):finished = False
                    continue
                tasks = load(root/model/f'ANNOTATION_TASKS-{phase}.json')['records']
                price = load(root/'ANNOTATOR_PRICE.json')
                count = max(1, int((22*3600-price.get('load_seconds',0))/price['seconds_per_context_upper']))
                slices = [list(range(i,min(i+count,len(tasks)))) for i in range(0,len(tasks),count)]
                publish(root/model/f'ANNOTATION_SHARDS-{phase}.json', {'phase':phase,'shards':slices})
                for i in range(len(slices)):
                    want('annotate',model,offset+i+1,minutes=1440)
                finished &= all(_stage_state(root,'annotate',model,offset+i+1) for i in range(len(slices)))
        if not finished:
            return False
        want('phase_export', shard=phase_index+1, minutes=180)
        if not _success(root,'phase_export','all',phase_index+1):
            if _failure(root,'phase_export','all',phase_index+1):
                terminal(campaign,'upstream_unavailable',{'phase':phase,'reason':'local phase export exhausted its bounded deterministic recovery'})
            return False
    return True


def shard_phase(stage, shard, root=None, model=None):
    if stage == 'phase_export': return PHASES[shard-1]
    if stage == 'contexts' and shard in (1,2): return PHASES[shard]
    if stage in ('score','annotate') and shard >= 100000: return PHASES[shard//100000-1]
    if stage == 'execute' and root and model:
        path = Path(root)/model/'EXECUTION_SHARDS.json'
        if path.exists():
            plan = load(path)
            if shard < len(plan['shards']): return plan['phases'][shard]
            path = Path(root)/model/'execution_fragments'/f'{shard:06d}.json'
            if path.exists(): return load(path)['phase']
    return 'preparation' if stage not in ('closeout','snapshot') else 'final' if stage == 'closeout' else 'snapshot'
