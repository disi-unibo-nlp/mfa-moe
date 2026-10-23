"""Paired policy contrasts and simultaneous bounds conditional on frozen policies."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from moe_exp.io import BENCHMARKS, CLASSES, MODELS, Units, digest, load, publish, require
from moe_exp.final_routing.endpoints import difference

CONTRASTS = ('activate:first', 'activate:occupancy', 'deactivate:first', 'deactivate:occupancy')


def joint_intervals(values, slots, *, draws=10000, seed=20261015):
    """Joint max-t envelope for all eight bound endpoints; shared family draws."""
    require(values.shape == (len(slots), 4, 2), 'primary component shape differs')
    families = sorted({r['family'] for r in slots})
    indices = {f: i for i, f in enumerate(families)}
    groups = {b: [i for i, r in enumerate(slots) if r['benchmark'] == b] for b in BENCHMARKS}
    require(all(groups.values()), 'cannot silently drop a benchmark')
    point = sum(values[ids].mean(0) for ids in groups.values())/len(BENCHMARKS)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(len(families), np.full(len(families), 1/len(families)), size=draws)
    boot = np.zeros((draws, 4, 2))
    undefined = np.zeros(draws, dtype=bool)
    for ids in groups.values():
        weights = multiplicity[:, [indices[slots[i]['family']] for i in ids]]
        denominators = weights.sum(1)
        undefined |= denominators == 0
        boot += np.einsum('di,icb->dcb', weights, values[ids])/np.maximum(1, denominators)[:, None, None]/6
    # Undefined draws are retained as unavailable inference, never discarded
    # to obtain a conditional, apparently narrower interval.
    se = boot.std(0, ddof=1)
    standardized = np.divide(np.abs(boot-point), se, out=np.zeros_like(boot), where=se > 0)
    critical = float(np.quantile(standardized.max(axis=(1, 2)), .95))
    rows = []
    for i, name in enumerate(CONTRASTS):
        rows.append({'contrast': name, 'lower_bound': float(point[i, 0]), 'upper_bound': float(point[i, 1]),
                     'ci_lower': None if undefined.any() else max(-1., float(point[i, 0]-critical*se[i, 0])),
                     'ci_upper': None if undefined.any() else min(1., float(point[i, 1]+critical*se[i, 1])),
                     'simultaneous_components': 8, 'critical_value': critical,
                     'bootstrap_draws': draws, 'undefined_draws': int(undefined.sum()),
                     'family_units': len(families), 'policies_held_fixed': True,
                     'undefined_draws_discarded':0, 'status':'unestimable' if undefined.any() else 'estimable'})
    return rows


def primary(binding, assignments, outcomes):
    allocation = load(Path(binding['root'])/'ALLOCATION.json')
    slots = allocation['records']
    slot_index = {(r['benchmark'], r['family']): i for i, r in enumerate(slots)}
    lookup = {(r['model'], r['benchmark'], r['family'], r['class'], r['depth_third'], r['arm']): r
              for r in assignments if r['dose'] == 'central' and r['arm'] != 'native'}
    values = np.zeros((len(slots), 4, 2))
    usable, cells = 0, []
    for model in MODELS:
        for c in CLASSES:
            for depth in range(3):
                cell = {'model': model, 'class': c, 'depth_third': depth, 'contrasts': {}}
                for direction in ('activate', 'deactivate'):
                    for endpoint in ('first', 'occupancy'):
                        contrasts = []
                        for slot in slots:
                            pair = []
                            for kind in ('target', 'random'):
                                row = lookup.get((model, slot['benchmark'], slot['family'], c, depth, kind+'_'+direction))
                                outcome = outcomes.get(row['uid']) if row else None
                                observed = (outcome['first'][c] if endpoint == 'first' else
                                            outcome['occupancy']['classes'][c]) if outcome else [0., 1.]
                                pair.append(observed)
                            contrast = difference(pair[0], pair[1], -1 if direction == 'deactivate' else 1)
                            i = CONTRASTS.index(direction+':'+endpoint)
                            values[slot_index[slot['benchmark'], slot['family']], i] += np.asarray(contrast)/(8*7*3)
                            contrasts.append((slot['benchmark'], contrast))
                        cell['contrasts'][direction+':'+endpoint] = (
                            sum(np.mean([v for b, v in contrasts if b == benchmark], axis=0) for benchmark in BENCHMARKS)/6).tolist()
                cells.append(cell)
    for row in assignments:
        if row['dose'] == 'central' and row['uid'] in outcomes:
            usable += 1
    result = joint_intervals(values, slots, draws=binding['config']['design']['bootstrap_draws'],
                             seed=binding['config']['seed'])
    return {'contrasts': result, 'cells': cells, 'usable_central_executions': usable,
            'estimand': 'benchmark-balanced, equal fixed model/class/depth, conditional frozen-policy ITT',
            'all_missing_components_retained': True}


def assemble(binding, mode):
    from .contexts import measured
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    assignments, outcomes, models = [], {}, {}
    for model in MODELS:
        path = root/model/('ASSIGNMENTS-'+mode+'.json')
        if not path.exists():
            models[model] = {'status': 'pending', 'observed': 0}
            continue
        rows = load(path)['records']
        assignments.extend(rows)
        count = 0
        for row in rows:
            context_path = bulk/model/'contexts'/(row['uid']+'.json')
            if context_path.exists() and row.get('pulse_tokens') and row.get('horizon'):
                outcomes[row['uid']] = measured(binding, row, load(context_path))
                count += 1
        models[model] = {'status': 'measured' if count else 'pending', 'observed': count,
                         'assigned': len(rows)}
    primary_result = primary(binding, assignments, outcomes)
    responses, quality, secondary = [], [], []
    for row in assignments:
        out = outcomes.get(row['uid'])
        if not out:
            for observed in CLASSES:
                for endpoint in ('occupancy','first_complete_sentence'):
                    responses.append({**{k:row[k] for k in ('uid','model','benchmark','family','fold','arm','dose','class','depth_third')},
                        'observed_class':observed,'lower':0.,'upper':1.,'endpoint':endpoint,'unknown':1.,'observed':False})
            continue
        shared = {k: row[k] for k in ('uid', 'model', 'benchmark', 'family', 'fold', 'arm', 'dose', 'class', 'depth_third')}
        quality.append({**shared, **{k: out[k] for k in ('strict_correct', 'new_tokens', 'reasoning_tokens',
                        'total_tokens', 'finish_reason', 'reasoning_end')},
                        'accuracy_lower': out['accuracy_bounds'][0], 'accuracy_upper': out['accuracy_bounds'][1]})
        for observed, bounds in out['occupancy']['classes'].items():
            responses.append({**shared, 'observed_class': observed, 'lower': bounds[0], 'upper': bounds[1],
                              'endpoint':'occupancy', 'reasoning_ended': out['occupancy']['reasoning_ended'], 'unknown': out['occupancy']['unknown']})
            first = out['first'][observed]
            responses.append({**shared,'observed_class':observed,'lower':first[0],'upper':first[1],
                              'endpoint':'first_complete_sentence'})
        secondary.append({**shared, 'sentences': load(bulk/row['model']/'contexts'/(row['uid']+'.json'))['sentences'],
                          'role': 'diagnostic', 'complete_three_dose_curve': False})
    return {'assignments': assignments, 'outcomes': outcomes, 'models': models,
            'primary': primary_result, 'responses': responses, 'quality': quality, 'secondary': secondary}


def paired_summaries(binding, data):
    """Paired within-prefix differences with frozen model/class/depth weighting."""
    from moe_exp.stats import bootstrap_panel
    rows, outcomes = data['assignments'], data['outcomes']
    lookup = {(r['model'],r['benchmark'],r['family'],r['class'],r['depth_third'],r['dose'],r['arm']):r for r in rows}
    pairs = []
    for row in rows:
        if not row['arm'].startswith('target_'):
            continue
        key = tuple(row[k] for k in ('model','benchmark','family','class','depth_third','dose'))
        other = lookup.get((*key,row['arm'].replace('target_','random_')))
        a, b = outcomes.get(row['uid']), outcomes.get((other or {}).get('uid'))
        shared = {k:row[k] for k in ('model','benchmark','family','fold','class','depth_third','dose','arm')}
        cap = binding['models'][row['model']]['native_context_limit']
        metrics = {}
        for metric in ('accuracy','new_tokens','reasoning_tokens','total_tokens'):
            def bounds(out):
                if out is None:
                    return [0.,1. if metric=='accuracy' else float(cap)]
                return out['accuracy_bounds'] if metric=='accuracy' else [float(out[metric])]*2
            lo, hi = difference(bounds(a),bounds(b))
            metrics[metric+'_lower'],metrics[metric+'_upper'] = lo,hi
        target = row['class']
        for metric in ('first','occupancy'):
            def endpoint(out):
                if out is None:return [0.,1.]
                return out['first'][target] if metric=='first' else out['occupancy']['classes'][target]
            lo,hi = difference(endpoint(a),endpoint(b),-1 if row['arm']=='target_deactivate' else 1)
            metrics[metric+'_lower'],metrics[metric+'_upper']=lo,hi
        pairs.append({**shared,**metrics,'both_observed':a is not None and b is not None})
    columns = [m+'_'+b for m in ('accuracy','new_tokens','reasoning_tokens','total_tokens','first','occupancy') for b in ('lower','upper')]
    summaries=[]
    # Missing components retain their known bounds. All components of a family
    # share the same resample; no label/outcome is used to reselect an assignment.
    for dose in ('central','low','high'):
        for arm in ('target_activate','target_deactivate'):
            subset=[r for r in pairs if r['dose']==dose and r['arm']==arm]
            if not subset:continue
            grouped=defaultdict(list)
            for r in subset:grouped[r['benchmark'],r['family']].append(r)
            records=[{'benchmark':b,'family':f,**{c:float(np.mean([r[c] for r in rs])) for c in columns}}
                     for (b,f),rs in grouped.items()]
            value=bootstrap_panel(records,columns,panel='final-routing:quality-dose:'+dose+':'+arm,
                resample_key='final-routing:shared-family',draws=10000,benchmarks=list(BENCHMARKS))
            summaries.append({'dose':dose,'arm':arm,'inference':value,'paired_cells':len(subset),
                'observed_pairs':sum(r['both_observed'] for r in subset),
                'scope':'preassigned dose population; fixed model/class/depth weights',
                'primary':False})
    return {'pairs':pairs,'summaries':summaries}
