"""One declared central steering family and a separate joint utility family."""
from collections import defaultdict
from pathlib import Path
from .contracts import MODELS, CLASSES, BENCHMARKS, load
from .statistics import bootstrap_panel


def central_panels(rows, campaign):
    root = Path(campaign['root'])
    if not (root/'COHORTS.json').exists() or not any(r['study']=='causal' and r.get('threshold_index')==0 and r.get('arm') in ('target','random') for r in rows): return []
    cohort = load(root/'COHORTS.json')['causal']
    universe = sorted({r['family'] for r in cohort})
    identities = sorted({(r['benchmark'], r['family'], r['question']) for r in cohort})
    lookup = {}
    for r in rows:
        if r['study'] == 'causal' and r.get('threshold_index') == 0 and r.get('arm') in ('target','random'):
            lookup[(r['benchmark'], r['family'], r.get('question'), r['model'], r['target'], r['depth_third'], r['sign'], r['arm'])] = r
    data = {key:{'benchmark':key[0], 'family':key[1], 'question':key[2]} for key in identities}
    primary = []; utility = []; observed = defaultdict(set)
    for model in MODELS:
        for target in CLASSES:
            for depth in range(3):
                for sign in (-1,1):
                    for control in ('native', 'random'):
                        base = f'{model}|{target}|{depth}|{sign}|target-minus-{control}|'
                        for endpoint, metric in (('first','first:'+target), ('recovery_w2','recovery:w2:'+target)):
                            names = [base+endpoint+'|'+bound for bound in ('lower','upper')]
                            primary.extend(names)
                            prefix, suffix = metric.split(':',1)
                            for key in identities:
                                treatment = lookup.get((*key, model,target,depth,sign,'target'), {})
                                random = lookup.get((*key, model,target,depth,sign,'random'), {})
                                if control == 'native':
                                    lo = treatment.get('native_difference_lower:'+metric)
                                    hi = treatment.get('native_difference_upper:'+metric)
                                else:
                                    tl,tu = treatment.get(prefix+'_lower:'+suffix,0.), treatment.get(prefix+'_upper:'+suffix,1.)
                                    cl,cu = random.get(prefix+'_lower:'+suffix,0.), random.get(prefix+'_upper:'+suffix,1.)
                                    lo,hi = tl-cu,tu-cl
                                lo,hi = (-1.,1.) if lo is None or hi is None else (lo,hi)
                                data[key].update(zip(names,(lo,hi)))
                                
                                if lo != -1. or hi != 1.:observed[base+endpoint].add(key[1])
                        for metric in ('strict_correct','new_tokens','unknown_lower','unknown_upper'):
                            name = base+metric; utility.append(name)
                            for key in identities:
                                treatment = lookup.get((*key,model,target,depth,sign,'target'), {})
                                random = lookup.get((*key,model,target,depth,sign,'random'), {})
                                if control == 'native':
                                    native_metric = ('native_difference_'+('lower' if metric=='unknown_lower' else 'upper')+':accuracy'
                                        if metric.startswith('unknown_') else 'native_difference:'+metric)
                                    value = treatment.get(native_metric)
                                elif metric.startswith('unknown_'):
                                    lower = metric == 'unknown_lower'
                                    a = treatment.get(metric, 0. if lower else 1.)
                                    b = random.get('unknown_upper' if lower else 'unknown_lower',1. if lower else 0.)
                                    value = a-b if a is not None and b is not None else None
                                else:
                                    a,b = treatment.get(metric),random.get(metric)
                                    value = a-b if a is not None and b is not None else None
                                data[key][name] = value
    panel_rows = list(data.values())
    panels = []
    for label, columns in (('steering-primary',primary), ('steering-utility',utility)):
        result = bootstrap_panel(panel_rows, columns, panel=label,
            resample_key='central:all-models-all-classes-all-depths-both-signs-both-controls',
            benchmarks=BENCHMARKS, family_universe=universe)
        result.update(identity=[label,'MMM'], declared_contrasts=1176 if label=='steering-primary' else 2352,
            bound_components=len(columns), diagnostic_sensitivities_excluded=True,
            coverage_scope='global within this declared family; primary and utility families separate')
        if label == 'steering-primary':
            result['identified_effects'] = {}
            for lower, upper in zip(primary[::2],primary[1::2]):
                a,b = result['results'][lower],result['results'][upper]
                interval = [a['interval'][0],b['interval'][1]] if a['interval'] and b['interval'] else None
                key = lower.rsplit('|',1)[0]
                result['identified_effects'][key] = {'sample_bounds':[a['estimate'],b['estimate']],
                    'simultaneous_range':interval, 'partly_observed_families':len(observed[key]),
                    'direction_identified':bool(interval and (interval[0]>0 or interval[1]<0))}
        panels.append(result)
    return panels


SUPPORTING = {
    'dwell':'restricted_target_dwell_tokens',
    'transitions':'target_transition_count',
    'reentry':'target_reentry_count',
    'recovery':'first_postpulse_matches_source',
    'target_exits':'observed_target_exits',
}


def supporting_panels(rows,campaign):
    """One registered simultaneous family per named endpoint, regardless of primary significance."""
    root=Path(campaign['root'])
    if not (root/'COHORTS.json').exists() or not any(r['study']=='causal' and r.get('threshold_index')==0 and r.get('arm') in ('target','random') for r in rows):return []
    cohort=load(root/'COHORTS.json')['causal']
    keys=sorted({(r['benchmark'],r['family'],r['question']) for r in cohort})
    lookup={(r['benchmark'],r['family'],r.get('question'),r['model'],r['target'],r.get('depth_third'),r.get('sign'),r.get('arm')):r
        for r in rows if r['study']=='causal' and r.get('threshold_index')==0}
    panels=[]
    for endpoint,metric in SUPPORTING.items():
        data={k:{'benchmark':k[0],'family':k[1],'question':k[2]} for k in keys};columns=[]
        for model in MODELS:
            for target in CLASSES:
                for depth in range(3):
                    for sign in (-1,1):
                        for control in ('native','random'):
                            col=f'{model}|{target}|{depth}|{sign}|target-minus-{control}|{metric}'
                            columns.append(col)
                            for key in keys:
                                t=lookup.get((*key,model,target,depth,sign,'target'),{})
                                if control=='native':value=t.get('native_difference:'+metric)
                                else:
                                    a=t.get(metric);b=lookup.get((*key,model,target,depth,sign,'random'),{}).get(metric)
                                    value=a-b if a is not None and b is not None else None
                                data[key][col]=value
        result=bootstrap_panel(list(data.values()),columns,panel='steering-supporting:'+endpoint,
            resample_key='central:all-models-all-classes-all-depths-both-signs-both-controls',
            benchmarks=BENCHMARKS,family_universe=sorted({k[1] for k in keys}))
        result.update(identity=['steering-supporting',endpoint,'MMM'],declared_contrasts=588,
            primary_significance_gate=False,window='w2',censored_trajectories_retained=True)
        panels.append(result)
    return panels
