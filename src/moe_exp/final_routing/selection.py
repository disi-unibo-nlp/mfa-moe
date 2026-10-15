"""Cross-fitted COUNT nominations; legacy mass selectors are never substituted."""
from collections import Counter
import numpy as np
from .common import CLASSES, digest, require
from .design import strengths


def count_set(excess, n):
    order = sorted((e for e, v in enumerate(excess) if np.isfinite(v) and v > 0),
                   key=lambda e: (-float(excess[e]), e))
    return order[:n] if len(order) >= n else []


def matched_controls(sets, frequency, identity, seed):
    """A nested control draws each additional expert from the target's quintile."""
    E = len(frequency)
    require(np.isfinite(frequency).all(), 'unsupported native routing frequency')
    ranks = sorted(range(E), key=lambda e: (float(frequency[e]), e))
    quintile = {e: min(4, i * 5 // E) for i, e in enumerate(ranks)}
    excluded = set().union(*(set(s) for s in sets))
    chosen, result = [], []
    for targets in sets:
        if not targets:
            result.append([])
            continue
        demand = Counter(quintile[e] for e in targets)
        have = Counter(quintile[e] for e in chosen)
        for q, count in sorted(demand.items()):
            pool = [e for e in ranks if quintile[e] == q and e not in excluded and e not in chosen]
            pool.sort(key=lambda e: digest([seed, 'control', identity, q, e]))
            missing = count - have[q]
            if len(pool) < missing:
                return None
            chosen.extend(pool[:missing])
        result.append(sorted(chosen))
    return result


def nominate(model, counts, tokens, sources, layers, k, sentence_rows, seed):
    from moe_exp.research_campaign_v3_r6.design import inclusion_statistics, family_weights, weighted_quantile
    folds = np.array([r['fold'] for r in sources])
    require(set(folds) <= set(range(5)), 'invalid discovery fold')
    sizes = strengths(k)
    other_counts = counts[:, :7].sum(1)[:, None] - counts[:, :7]
    other_tokens = tokens[:, :7].sum(1)[:, None] - tokens[:, :7]
    records, atlas = [], []
    for fold in range(5):
        train = folds != fold
        rate, support = inclusion_statistics(counts, tokens, sources, train)
        other, osupport = inclusion_statistics(other_counts, other_tokens, sources, train)
        excess = rate[:7] - other
        excess[(support[:7] < 2) | (osupport < 2)] = np.nan
        stability = np.zeros((7, len(layers)))
        all_stability=np.zeros((7,len(layers),3))
        for omitted in set(range(5)) - {fold}:
            sub = train & (folds != omitted)
            sr, ss = inclusion_statistics(counts, tokens, sources, sub)
            so, sos = inclusion_statistics(other_counts, other_tokens, sources, sub)
            sx = sr[:7] - so
            sx[(ss[:7] < 2) | (sos < 2)] = np.nan
            for c in range(7):
                for i in range(len(layers)):
                    a, b = set(count_set(excess[c, i], sizes[1])), set(count_set(sx[c, i], sizes[1]))
                    stability[c, i] += (len(a & b) / len(a | b) if a | b else 0) / 4
                    for d,size in enumerate(sizes):
                        aa,bb=set(count_set(excess[c,i],size)),set(count_set(sx[c,i],size))
                        all_stability[c,i,d]+=(len(aa&bb)/len(aa|bb) if aa|bb else 0)/4
        sent = [r for r in sentence_rows if r['fold'] != fold]
        q50, q75 = weighted_quantile([r['n_tokens'] for r in sent], family_weights(sent), (.5, .75))
        pulse = max(1, round(q50)) if q50 is not None else None
        horizon = 16 * max(1, round(q75)) if q75 is not None else None
        for c, name in enumerate(CLASSES):
            for depth, indices in enumerate(np.array_split(np.arange(len(layers)), 3)):
                candidates = []
                for i in indices:
                    ids = count_set(excess[c, i], sizes[1])
                    if ids:
                        score = float(excess[c, i, ids].sum()) / k
                        candidates.append((-score, -stability[c, i], int(layers[i]), int(i)))
                chosen = min(candidates, default=None)
                r = {'model': model, 'fold': fold, 'class': name, 'depth_third': depth,
                     'status': 'insufficient_support', 'layer': None, 'counts': sizes,
                     'pulse_tokens': pulse, 'horizon': horizon, 'expert_sets': [[], [], []],
                     'control_sets': [[], [], []], 'training_folds': sorted(set(range(5)) - {fold})}
                if chosen and pulse and horizon:
                    i = chosen[-1]
                    sets = [count_set(excess[c, i], n) for n in sizes]
                    # An unsupported outer count must not erase the central
                    # assignment. Controls exclude every available target set.
                    controls = matched_controls(sets, rate[7, i], [model, fold, name, depth], seed) if sets[1] else None
                    r.update(layer=int(layers[i]), expert_sets=[sorted(s) for s in sets],
                             control_sets=controls or [[], [], []], association=-chosen[0],
                             stability=float(stability[c, i]),
                             family_support=int(min(support[c, i].min(), osupport[c, i].min())))
                    r['selected_set_association']=[float(excess[c,i,s].sum()/k) if s else None for s in sets]
                    r['selected_set_support']=[int(min(support[c,i,s].min(),osupport[c,i,s].min())) if s else 0 for s in sets]
                    r['selected_set_stability']=all_stability[c,i].tolist()
                    if controls:
                        r['status'] = 'nominated'
                    atlas.append({'model': model, 'fold': fold, 'class': name, 'layer': int(layers[i]),
                                  'excess': excess[c, i].tolist(), 'frequency': rate[7, i].tolist(),
                                  'support': support[c, i].tolist()})
                records.append(r)
    return {'records': records, 'atlas': atlas, 'outcomes_used': False, 'selector': 'positive-inclusion-excess-count-v1'}


def confirmation(nominations,counts,tokens,sources,layers,k):
    from moe_exp.research_campaign_v3_r6.design import inclusion_statistics
    records=[]
    folds=np.asarray([r['fold'] for r in sources])
    other_counts=counts[:,:7].sum(1)[:,None]-counts[:,:7]
    other_tokens=tokens[:,:7].sum(1)[:,None]-tokens[:,:7]
    for fold in range(5):
        rate,support=inclusion_statistics(counts,tokens,sources,folds==fold)
        other,osupport=inclusion_statistics(other_counts,other_tokens,sources,folds==fold)
        for n in nominations['records']:
            if n['fold']!=fold:continue
            r={key:n[key] for key in ('model','fold','class','depth_third','status','layer')}
            r['effects']=[]
            if n['status']=='nominated':
                c,i=CLASSES.index(n['class']),layers.index(n['layer'])
                for dose,ids,controls in zip(('low','central','high'),n['expert_sets'],n['control_sets']):
                    valid=bool(ids and controls and np.isfinite(rate[c,i,ids]).all() and np.isfinite(other[c,i,ids]).all())
                    r['effects'].append({'dose':dose,'expert_ids':ids,'control_ids':controls,
                        'selected_association':n['selected_set_association'][('low','central','high').index(dose)],
                        'heldout_excess':float((rate[c,i,ids]-other[c,i,ids]).sum()/k) if valid else None,
                        'heldout_support':int(min(support[c,i,ids].min(),osupport[c,i,ids].min())) if valid else 0,
                        'selection_recomputed_on_confirmation':False})
            records.append(r)
    return {'records':records,'cross_fitted':True,'new_selection':False}
