"""Intervention effects of the reduced design (CPU): lexical state endpoints, efficiency, inference.

Endpoints per continuation (window [pulse, H) of continuation tokens, H = 16·Q75, as `endpoints.occupancy`):
  occupancy of each class (hard lexical labels; soft = classifier probability mass), fraction ended,
  class of the first complete sentence, generated/total tokens, strict accuracy with unknown bounds.
Contrasts are paired within (model, family, class): targeted vs matched random (primary; deactivation sign
flipped so positive = intended), targeted vs native, random vs native. Pooled estimands weight models and
classes equally within a family and benchmarks equally across families. Inference resamples the shared
question families (10,000 multinomial draws, the same draws for every contrast) and reports percentile
intervals plus a max-|t| simultaneous envelope over each pre-declared family of contrasts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .common import BENCHMARKS, CLASSES, MODELS, load, publish, require
from .driver import DRIVER, ROOT, binding, bulk, log

PRIMARY = ('activate:first', 'activate:occupancy', 'deactivate:first', 'deactivate:occupancy')


def endpoints(outcome, sentences):
    """outcome: dict row of the outcomes table; sentences: rows (token_start, token_end, complete, p_*, lexical_label)."""
    pulse, horizon = int(outcome['pulse_tokens']), int(outcome['horizon'])
    n = horizon-pulse
    end = outcome['reasoning_end']
    natural = outcome['generated_tokens'] if outcome['finish_reason'] == 'stop' else None
    ends = [x for x in (end, natural) if x is not None and x == x]
    stop = min(ends) if ends else None
    nonreasoning = set(json.loads(outcome['nonreasoning'])) if isinstance(outcome['nonreasoning'], str) else set()
    hard, soft = np.zeros(7), np.zeros(7)
    ended = covered = 0
    spans = sorted(sentences, key=lambda s: s['token_start'])
    j = 0
    for p in range(pulse, horizon):
        if stop is not None and p >= stop:
            ended += 1
            continue
        if p in nonreasoning:
            continue
        while j < len(spans) and spans[j]['token_end'] <= p:
            j += 1
        if j < len(spans) and spans[j]['token_start'] <= p:
            s = spans[j]
            hard[CLASSES.index(s['lexical_label'])] += 1
            soft += np.array([s['p_'+c] for c in CLASSES])
            covered += 1
    first = next((s for s in spans if s['complete'] and not s.get('straddles_prefix', False)), None)
    return {'occupancy_hard': hard/n, 'occupancy_soft': soft/n, 'ended': ended/n, 'uncovered': (n-ended-covered)/n,
            'first_hard': (np.eye(7)[CLASSES.index(first['lexical_label'])] if first else np.full(7, np.nan)),
            'first_soft': (np.array([first['p_'+c] for c in CLASSES]) if first else np.full(7, np.nan))}


def continuation_table(models, design='main'):
    """design 'strong': post-hoc strong-dose arms (renamed to target/random_activate) plus the main-design native rows."""
    import pandas as pd
    from .lexical import LEXICAL, predict
    frames = []
    for model in models:
        folder = bulk()/model/'outcomes'
        def read(kind):
            main = [pd.read_parquet(p) for p in sorted(folder.glob(kind+'-[0-9]*.parquet'))]
            if design == 'main':
                return pd.concat(main, ignore_index=True)
            strong = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob(kind+'-strong-*.parquet'))], ignore_index=True)
            native = pd.concat(main, ignore_index=True)
            native = native[native.arm == 'native']
            strong['arm'] = strong.arm.str.replace('strong_', '', regex=False)
            return pd.concat([native, strong], ignore_index=True)
        outcomes, sentences = read('outcomes'), read('sentences')
        if design == 'strong':  # routing forced over the whole window: measure [0, H) for strong and native rows
            outcomes = outcomes.assign(pulse_tokens=0)
        sentences = predict(sentences, LEXICAL/'readout-v1')
        groups = {u: g.to_dict('records') for u, g in sentences.groupby('uid')}
        rows = []
        for o in outcomes.to_dict('records'):
            e = endpoints(o, groups.get(o['uid'], []))
            row = {k: o[k] for k in ('uid', 'model', 'arm', 'class', 'family', 'fold', 'benchmark', 'layer',
                                      'generated_tokens', 'prefix_tokens', 'reasoning_tokens', 'finish_reason',
                                      'strict_correct', 'accuracy_lower', 'accuracy_upper', 'segments')}
            row.update(ended=e['ended'], uncovered=e['uncovered'])
            for i, c in enumerate(CLASSES):
                row['occ_'+c], row['occs_'+c] = e['occupancy_hard'][i], e['occupancy_soft'][i]
                row['first_'+c], row['firsts_'+c] = e['first_hard'][i], e['first_soft'][i]
            rows.append(row)
        frames.append(pd.DataFrame(rows))
        log({'event': 'continuations', 'model': model, 'rows': len(rows), 'sentences': len(sentences)})
    return pd.concat(frames, ignore_index=True)


def paired(table):
    """One row per (model, family, class) with paired differences for every contrast family."""
    import pandas as pd
    t = table.copy()
    t['log_tokens'] = np.log(t.generated_tokens.clip(lower=1))
    t['log_total'] = np.log((t.generated_tokens+t.prefix_tokens).clip(lower=1))
    t['log_reasoning'] = np.log(t.reasoning_tokens.clip(lower=1))
    t['correct'] = t.strict_correct.astype(float)
    native = t[t.arm == 'native'].set_index(['model', 'family'])
    rows = []
    for (model, family, c), g in t[t.arm != 'native'].groupby(['model', 'family', 'class']):
        arms = {r['arm']: r for r in g.to_dict('records')}
        nat = native.loc[(model, family)].to_dict() if (model, family) in native.index else None
        out = {'model': model, 'family': family, 'class': c, 'benchmark': g.benchmark.iloc[0],
               'layer': g.layer.iloc[0]}
        for direction, sign in (('activate', 1), ('deactivate', -1)):
            ta, ra = arms.get('target_'+direction), arms.get('random_'+direction)
            for kind, key in (('occupancy', 'occ_'), ('first', 'first_'), ('soft_occupancy', 'occs_')):
                out[f'{direction}:{kind}'] = sign*(ta[key+c]-ra[key+c]) if ta and ra else np.nan
                out[f'{direction}:{kind}:vs_native'] = sign*(ta[key+c]-nat[key+c]) if ta and nat else np.nan
            for metric in ('log_tokens', 'log_total', 'log_reasoning', 'correct', 'generated_tokens', 'ended'):
                out[f'{direction}:{metric}'] = ta[metric]-ra[metric] if ta and ra else np.nan
                out[f'{direction}:{metric}:vs_native'] = ta[metric]-nat[metric] if ta and nat else np.nan
                out[f'random_{direction}:{metric}:vs_native'] = ra[metric]-nat[metric] if ra and nat else np.nan
            for c2 in CLASSES:  # response matrix: targeted vs random on every observed class
                out[f'{direction}:response:{c2}'] = (ta['occ_'+c2]-ra['occ_'+c2]) if ta and ra else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


class Bootstrap:
    """Shared multinomial family weights; benchmark-balanced means of family-level cell averages."""
    def __init__(self, families, benchmarks, draws=10000, seed=20261015):
        self.families = list(families)
        self.index = {f: i for i, f in enumerate(self.families)}
        self.benchmark = np.array([benchmarks[f] for f in self.families])
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(len(self.families), np.full(len(self.families), 1/len(self.families)), size=draws)

    def estimate(self, frame, column):
        """frame rows (family, value); value averaged within family first (equal model/class weights)."""
        sub = frame[['family', column]].dropna()
        if sub.empty:
            return None
        means = sub.groupby('family')[column].mean()
        v = np.full(len(self.families), np.nan)
        for f, x in means.items():
            v[self.index[f]] = x
        point, boot, used = 0., 0., 0
        for b in BENCHMARKS:
            ids = np.where((self.benchmark == b) & ~np.isnan(v))[0]
            if not len(ids):
                continue
            point += v[ids].mean()
            w = self.weights[:, ids]
            boot = boot+(w*v[ids]).sum(1)/np.maximum(w.sum(1), 1)
            used += 1
        point, boot = point/used, boot/used
        return {'estimate': float(point), 'se': float(boot.std(ddof=1)), 'ci': np.quantile(boot, [.025, .975]).tolist(),
                'families': int((~np.isnan(v)).sum()), 'cells': int(len(sub)), 'benchmarks': used, 'draws': boot}


def summarize(boot, frame, columns, family_name):
    """Estimates for `columns` with a max-|t| simultaneous 95% envelope across them."""
    found = {c: boot.estimate(frame, c) for c in columns}
    found = {c: v for c, v in found.items() if v is not None}
    if not found:
        return {}
    t = np.stack([np.abs(v['draws']-v['estimate'])/max(v['se'], 1e-12) for v in found.values()], 1)
    critical = float(np.quantile(t.max(1), .95))
    out = {}
    for c, v in found.items():
        out[c] = {k: v[k] for k in ('estimate', 'se', 'ci', 'families', 'cells')}
        out[c]['simultaneous'] = [v['estimate']-critical*v['se'], v['estimate']+critical*v['se']]
        out[c]['critical'] = critical
        out[c]['family_of_contrasts'] = family_name
    return out


def analyze(models, out, design='main'):
    import pandas as pd
    allocation = load(ROOT/'ALLOCATION.json')['records']
    benchmarks = {r['family']: r['benchmark'] for r in allocation}
    table = continuation_table(models, design)
    pairs = paired(table)
    boot = Bootstrap(sorted(benchmarks), benchmarks)
    result = {'models': models, 'design': design, 'continuations': int(len(table)), 'pairs': int(len(pairs)),
              'readout': 'lexical-readout-v1 (cross-fitted; no judge on intervention text)'}
    result['primary_pooled'] = summarize(boot, pairs, list(PRIMARY), 'primary-4')
    result['primary_by_model'] = {m: summarize(boot, pairs[pairs.model == m], list(PRIMARY), f'primary-4:{m}')
                                  for m in models}
    result['primary_by_class'] = {c: summarize(boot, pairs[pairs['class'] == c], list(PRIMARY), f'primary-4:{c}')
                                  for c in CLASSES}
    metrics = ('log_tokens', 'log_total', 'log_reasoning', 'correct', 'ended')
    efficiency = [f'{d}:{m}' for d in ('activate', 'deactivate') for m in metrics]
    efficiency += [f'{d}:{m}:vs_native' for d in ('activate', 'deactivate') for m in metrics]
    efficiency += [f'random_{d}:{m}:vs_native' for d in ('activate', 'deactivate') for m in metrics]
    result['efficiency_pooled'] = summarize(boot, pairs, efficiency, 'efficiency')
    result['efficiency_by_class'] = {c: summarize(boot, pairs[pairs['class'] == c], efficiency, f'efficiency:{c}')
                                     for c in CLASSES}
    result['efficiency_by_model_class'] = {f'{m}|{c}': summarize(boot, pairs[(pairs.model == m) & (pairs['class'] == c)],
                                           ['activate:log_tokens', 'activate:correct', 'activate:log_tokens:vs_native',
                                            'activate:correct:vs_native'], f'efficiency:{m}:{c}')
                                           for m in models for c in CLASSES}
    secondary = [f'{d}:{k}' for d in ('activate', 'deactivate') for k in ('soft_occupancy', 'occupancy:vs_native',
                                                                            'first:vs_native')]
    result['secondary_pooled'] = summarize(boot, pairs, secondary, 'secondary')
    result['response_matrix'] = {c: {c2: (boot.estimate(pairs[pairs['class'] == c], f'activate:response:{c2}') or {}).get('estimate')
                                     for c2 in CLASSES} for c in CLASSES}
    readout = load(Path(str(bulk().parent/'lexical'/'readout-v1'/'READOUT.json')))
    corrected = {}
    for c in CLASSES:  # Rogan-Gladen sensitivity: divide by the native out-of-fold Youden J (pooled readout)
        j = readout['native_out_of_fold']['pooled']['youden_j'][c]
        v = result['primary_by_class'][c].get('activate:occupancy')
        if v and j > 0:
            corrected[c] = {'youden_j': j, 'estimate': v['estimate']/j, 'ci': [x/j for x in v['ci']]}
    result['rogan_gladen_sensitivity'] = corrected
    prereg = DRIVER/'EFFICIENCY_LINK_PREREG.json'
    if prereg.exists() and design == 'main':  # the pre-registration excludes other strengths
        from .common import verified
        frozen = verified(prereg)
        result['efficiency_link'] = efficiency_link(frozen, pairs, boot)
        result['efficiency_link']['prereg_sha256'] = frozen['sha256']
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out/'continuations.parquet', index=False)
    pairs.to_parquet(out/'pairs.parquet', index=False)
    return publish(out/'EFFECTS.json', result)


def efficiency_link(prereg, pairs, boot):
    """The pre-registered test (EFFICIENCY_LINK_PREREG.json 'test'): frozen beta(m,c) vs measured Delta(m,c) =
    family mean of targeted-minus-random activation (log total tokens; strict accuracy). Spearman rho and the
    sign-agreement rate on signed cells, each with a 95% percentile interval from a family-cluster bootstrap of
    the intervention side (2,000 draws, seed 20261017, betas fixed), plus an exact binomial test. Secondary:
    tok_total / acc_excl_trunc / between-question (dataset_fe) predictors and the sign-robust restricted rho."""
    from scipy.stats import binomtest, spearmanr
    link = Bootstrap(boot.families, dict(zip(boot.families, boot.benchmark)), draws=2000, seed=20261017)

    def test(cells, outcome, column, restrict=False):
        beta = {(r['model'], r['class']): r[outcome] for r in cells if r.get(outcome) and r[outcome].get('beta') is not None}
        if restrict:
            beta = {k: v for k, v in beta.items() if v.get('sign') in ('+', '-') and not v.get('opposite_sign_in')}
        means = {}
        for k in sorted(beta):
            sub = pairs[(pairs.model == k[0]) & (pairs['class'] == k[1])][['family', column]].dropna()
            if len(sub):
                means[k] = sub.groupby('family')[column].mean()
        keys = sorted(means)
        if len(keys) < 4:
            return {'status': 'insufficient_cells', 'cells': len(keys)}
        cols = {k: np.array([link.index[f] for f in means[k].index]) for k in keys}
        xs = np.array([beta[k]['beta'] for k in keys])
        signed = np.array([beta[k].get('sign') in ('+', '-') for k in keys])
        predicted = np.array([1 if beta[k].get('sign') == '+' else -1 for k in keys])

        def stats(weights):
            if weights is None:
                ys = np.array([means[k].mean() for k in keys])
            else:
                ys = np.array([(weights[cols[k]]*means[k].to_numpy()).sum()/max(weights[cols[k]].sum(), 1) for k in keys])
            agree = ((np.sign(ys) == predicted) & (ys != 0))[signed]
            return spearmanr(xs, ys).statistic, (agree.mean() if signed.any() else np.nan), ys
        rho, rate, ys = stats(None)
        draws = np.array([stats(link.weights[i])[:2] for i in range(len(link.weights))])
        agree = int(round(rate*signed.sum())) if signed.any() else 0
        return {'spearman': float(rho), 'spearman_ci': np.nanquantile(draws[:, 0], [.025, .975]).tolist(),
                'sign_agreement_rate': float(rate) if signed.any() else None,
                'sign_agreement_ci': np.nanquantile(draws[:, 1], [.025, .975]).tolist() if signed.any() else None,
                'signed_cells': int(signed.sum()), 'sign_agreement': agree,
                'binomial_p': float(binomtest(agree, int(signed.sum()), .5).pvalue) if signed.any() else None,
                'supported': bool(np.nanquantile(draws[:, 0], .025) > 0), 'cells': len(keys),
                'bootstrap_draws': len(link.weights), 'seed': 20261017,
                'cells_detail': [{'model': k[0], 'class': k[1], 'beta': float(beta[k]['beta']),
                                  'predicted_sign': beta[k].get('sign'), 'delta': float(y)} for k, y in zip(keys, ys)]}
    cells = prereg['cells']
    out = {'tokens': test(cells, 'tok_reason', 'activate:log_total'),
           'accuracy': test(cells, 'acc', 'activate:correct'),
           'secondary': {'tok_total': test(cells, 'tok_total', 'activate:log_total'),
                         'acc_excl_trunc': test(cells, 'acc_excl_trunc', 'activate:correct'),
                         'tokens_restricted': test(cells, 'tok_reason', 'activate:log_total', restrict=True),
                         'accuracy_restricted': test(cells, 'acc', 'activate:correct', restrict=True)}}
    fe = (prereg.get('sensitivity') or {}).get('dataset_fe', {}).get('cells')
    if fe:
        out['secondary']['dataset_fe_tokens'] = test(fe, 'tok_reason', 'activate:log_total')
        out['secondary']['dataset_fe_accuracy'] = test(fe, 'acc', 'activate:correct')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='moe_exp.study.effects')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', default=str(DRIVER/'analysis'))
    ap.add_argument('--design', choices=['main', 'strong'], default='main')
    args = ap.parse_args(argv)
    models = [m for m in args.models.split(',') if (bulk()/m/'outcomes').exists()]
    require(models, 'no extracted outcomes')
    value = analyze(models, Path(args.out), args.design)
    log({'event': 'effects', 'primary': value['primary_pooled']})


if __name__ == '__main__':
    main()
