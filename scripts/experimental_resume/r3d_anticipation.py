"""R3-D anticipation: clean past-text/history controls, OOF class estimates and noise.

Existing K2 raw caches and A1 histograms only. Missing cells remain incomplete;
no tensor producer is run. The fixed grid/windows are never searched or expanded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
OUT = ROOT / 'forum/tests/r3_context'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SEED = 20260929
STATE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def seal(value):
    return {**value, 'sha256': hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()}


def prepare():
    path = OUT / 'code/rkin'
    path.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / 'reasoning-kinematics/rk2/rkin').glob('*.py')):
        target = path / source.name
        if target.exists() and sha(target) != sha(source):
            raise ValueError('historical helper copy changed')
        if not target.exists():
            shutil.copy2(source, target)
    target = OUT / 'code/r3d_anticipation.py'
    if target.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, target)
    print(json.dumps({'driver': str(target), 'status': 'PREPARED'}))


def augment_baseline(model, pairs, cohort, P, T):
    """Only problem/past emitted text and observed earlier classes; no difficulty."""
    import numpy as np
    import pandas as pd
    att, sent = cohort.attempts, cohort.sent
    ids = {(r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256) for r in att.itertuples()}
    units = P.load_unit_texts(model, ids)
    records = att[['dataset', 'problem_id', 'sample_id', 'trace_sha256']].to_dict('records')
    by_sentence = {(int(cohort.table.attempt[i]), int(row.sentence_index)): (int(row.cls), int(row.segment))
                   for i, row in enumerate(sent.itertuples())}
    text, history = [], []
    for row in pairs.itertuples():
        a, index = int(row.att), int(row.sentence_index)
        record = records[a]
        key = (record['dataset'], record['problem_id'], int(record['sample_id']), record['trace_sha256'])
        current = units.get(key + (index,))
        if current is None or current[1] != P.CLASSES[int(row.a)]:
            raise ValueError('prefix text and source label identity disagree')
        previous = units.get(key + (index - 1,))
        text.append('previous ' + (previous[0] if previous else '<missing>') + '\ncurrent ' + current[0])
        prior = [by_sentence.get((a, index - gap)) for gap in (1, 2, 3)]
        this_segment = by_sentence[a, index][1]
        prior = [item if item is not None and item[1] == this_segment else None for item in prior]
        features = [float(item is None) for item in prior]
        for item in prior:
            features.extend([float(item is not None and item[0] == c) for c in range(7)])
        words = current[0].lower().split()
        features.extend([float(len(words)), float(sum(char.isdigit() for char in current[0])),
            float(current[0].count('=')), float(sum(w in ('check','verify','test') for w in words)),
            float(sum(w in ('try','instead','maybe') for w in words))])
        history.append(features)
    base = np.column_stack([np.log(pairs['n_tokens'].to_numpy(float)),
        *(pairs['abs_bin'].to_numpy() == i for i in range(6)),
        *(pairs['rel_bin'].to_numpy() == i for i in range(5)), np.asarray(history)])
    return base, T.hash_counts(text)


def configure_blocks(K, F, C, Softmax, T, folds):
    import numpy as np
    from scipy import sparse
    original = K.Blocks
    class ControlledBlocks(original):
        def __init__(self, d, tr, evs):
            super().__init__(d, tr, evs)
            self._strong_sparse = self.xs_tr, self.xs_ev
            weak = T.TextTransform.fit(d.text_weak, tr, d.s_text)
            lexical = T.TextTransform.fit(d.lex, tr, d.s_lex)
            self._weak_sparse = (sparse.hstack([weak.transform(d.text_weak,tr),lexical.transform(d.lex,tr)],format='csr'),
                [sparse.hstack([weak.transform(d.text_weak,e),lexical.transform(d.lex,e)],format='csr') for e in evs])
        def dense(self, keys):
            weak = 'weak_sparse' in keys
            self.xs_tr,self.xs_ev = self._weak_sparse if weak else self._strong_sparse
            return super().dense([key for key in keys if key!='weak_sparse'])
        def _raw(self, key):
            d = self.d
            if key == 'prefix_class':
                if key not in self._cache:
                    raw = np.full((d.n, 7), np.nan)
                    # Fixed lambda=1 was declared before execution; no outcome-driven tuning.
                    # Training-row predictions themselves are grouped out of fold.
                    inner = folds(d.groups[self.tr], 3, SEED + 777)
                    for fold in range(3):
                        tr = self.tr[inner != fold]
                        va = self.tr[inner == fold]
                        self._predict_class(tr, [va], raw)
                    self._predict_class(self.tr, self.evs, raw)
                    self._class_raw = raw
                return self._class_raw
            if key in ('composition16', 'composition_antic'):
                if not hasattr(self, '_profiles'):
                    cohort, attempts, _, _ = d.composition
                    mask = np.zeros(cohort.table.n_attempts, bool)
                    mask[np.unique(attempts[self.tr])] = True
                    rows = np.flatnonzero(mask[cohort.table.attempt])
                    table = cohort.table
                    self._profiles = F.fit_profiles(table.h[rows], table.cls[rows], table.attempt[rows],
                                                     table.layers, table.experts)
                if not hasattr(self, '_composition_raw'):
                    self._composition_raw = {}
                if key not in self._composition_raw:
                    h = d.composition[2 if key == 'composition16' else 3]
                    raw = F.d_scores(h, self._profiles)
                    if key == 'composition_antic':
                        raw[~d.has_antic] = np.nan
                    self._composition_raw[key] = raw
                return self._composition_raw[key]
            if key.startswith('noise_'):
                target, draw = key[len('noise_'):].rsplit('_', 1)
                real = self._raw(target)
                rng = np.random.default_rng([SEED, int(draw), 901 if target.startswith('composition') else 902])
                raw = rng.normal(size=real.shape)
                raw[np.isnan(real)] = np.nan
                return raw
            return super()._raw(key)

        def _predict_class(self, train, evaluations, raw):
            d = self.d
            parts_tr, parts_ev = [], [[] for _ in evaluations]
            for counts, scale in ((d.text, d.s_text), (d.lex, d.s_lex)):
                transform = T.TextTransform.fit(counts, train, scale)
                parts_tr.append(transform.transform(counts, train))
                for i, ev in enumerate(evaluations):
                    parts_ev[i].append(transform.transform(counts, ev))
            scaler = C.Scaler.fit(d.dense['base'][train])
            fit = Softmax.fit_softmax(scaler.transform(d.dense['base'][train]),
                sparse.hstack(parts_tr, format='csr'), d.a[train], d.w[train], 1., 7)
            for ev, pieces in zip(evaluations, parts_ev):
                raw[ev] = np.exp(Softmax.predict_log_proba(fit, scaler.transform(d.dense['base'][ev]),
                                                        sparse.hstack(pieces, format='csr')))
    K.Blocks = ControlledBlocks


def run_cv(K, d, models, workers, binding, directory):
    import multiprocessing as mp
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    K._STATE.update(data=d, models=models)
    names = sorted(models)
    chunks = [names[i:i + 8] for i in range(0, len(names), 8)]
    tasks = [(r, f, c) for c in chunks for r in range(5) for f in range(5)]
    predictions = {name: np.full((5, d.n, d.K), np.nan, np.float32) for name in models}
    stats = []
    def path(task):
        r, f, c = task
        return directory / (str(r) + '-' + str(f) + '-' + sha_text('|'.join(c))[:12] + '.npz')
    def accept(result):
        r, f, test, lp, lam, curves, st, seconds = result
        for name, values in lp.items():
            predictions[name][r, test] = values
        stats.extend(st)
    pending = []
    for task in tasks:
        receipt = path(task)
        if not receipt.exists():
            pending.append(task)
            continue
        with np.load(receipt, allow_pickle=False) as z:
            if str(z['binding']) != binding:
                raise ValueError('anticipation checkpoint code/input mismatch')
            r, f, c = task
            accept((r, f, z['test'], {name: z['lp:' + name] for name in c}, {}, {}, z['stats'], 0.))
    with mp.get_context('fork').Pool(workers) as pool:
        for task, result in pool.imap_unordered(task_with_key, pending):
            accept(result)
            r, f, test, lp, lam, curves, st, seconds = result
            receipt = path(task)
            partial = receipt.with_suffix('.pending')
            with partial.open('wb') as handle:
                np.savez_compressed(handle, binding=binding, test=test, stats=np.asarray(st),
                    **{'lp:' + name: values for name, values in lp.items()})
            partial.replace(receipt)
            print(f'{d.model}/{d.target} repeat {r} fold {f}: {len(lp)} models ({seconds:.1f}s)', flush=True)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('missing anticipation predictions')
    with (directory / 'predictions.npz').open('wb') as handle:
        np.savez_compressed(handle, **predictions, questions=d.groups, y=d.y, folds=d.outer_folds)
    return predictions, np.asarray(stats)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def task_with_key(task):
    from rkin import k2
    return task, k2._task(task)


def summarize(d, predictions, targets, families):
    import numpy as np
    loss = {name: -values[:, np.arange(d.n), d.y] for name, values in predictions.items()}
    uq, qi = np.unique(d.groups, return_inverse=True)
    nf = np.array([families[q] for q in uq])
    uf, fi = np.unique(nf, return_inverse=True)
    W = np.random.default_rng(SEED).multinomial(len(uf), np.full(len(uf), 1 / len(uf)), size=5000)
    output = {}
    for target in targets:
        delta = loss[target] - loss['base']
        question_values = np.array([delta[:, qi == i].mean(axis=1) for i in range(len(uq))])
        values = question_values.mean(axis=1)
        sums = np.bincount(fi, weights=values)
        denominators = np.bincount(fi)
        boot = (W @ sums) / (W @ denominators)
        noise = [float(np.average((loss[f'noise_{target}_{draw}']-loss['base']).mean(axis=0),weights=d.w))
                 for draw in range(20)]
        signs=np.random.default_rng([SEED,991]).integers(0,2,size=(10000,len(uf)))*2-1
        null=signs@sums/denominators.sum()
        output[target] = {'P': float(values.mean()), 'ci95': np.quantile(boot, [.025,.975]).tolist(),
            'simultaneous_ci6': np.quantile(boot, [.025/6,1-.025/6]).tolist(),
            'p_bootstrap': float(min(1., 2 * min((1+(boot<=0).sum())/5001,(1+(boot>=0).sum())/5001))),
            'gain_positive_repeats': int((question_values.mean(axis=0)<0).sum()),
            'per_repeat_P': question_values.mean(axis=0).tolist(), 'noise_P': noise,
            'noise_mean_P': float(np.mean(noise)), 'noise_SD_P': float(np.std(noise,ddof=1))}
        output[target]['sign_flip_p']=float((1+(np.abs(null)>=abs(values.mean())).sum())/10001)
        output[target]['sign_flip_assumption']='family-level paired prediction-loss exchangeability; approximate observational inference'
        output[target]['bootstrap_SD']=float(boot.std(ddof=1))
        from scipy.stats import norm
        output[target]['analytic_MDE_nats_per_pair']=float((norm.ppf(1-.05/(2*6))+norm.ppf(.8))*boot.std(ddof=1))
    return output


def run(workers):
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('anticipation fits require CPU Slurm')
    started = time.monotonic()
    sys.path.insert(0, str(OUT / 'code'))
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from r3d_readout import family_folds, verified
    from dynrt import common as D, cv as C
    D.RESULTS = ROOT / 'dynamics-routing/results'
    from dynrt import data as Data, d3a_pairs as P, d3a_text as T, features as F, d3a_softmax as Softmax
    from rkin import common as RK
    RK.DATA = ROOT / 'reasoning-kinematics/rk2/data'
    from rkin import k2 as K
    K.DATA = RK.DATA
    K.K2DATA = RK.DATA / 'k2'
    K.OUTER = 5
    family_path = REPO / 'report/experimental-resume-v1/family-freeze.json'
    family = verified(family_path)
    qf = {q:f for f, qs in family['new_parent_pools']['families'].items() for q in qs}
    split = D.load_split()
    folds = lambda qs, n, seed: family_folds(qs, n, seed if seed>=SEED else SEED*1000+seed, qf)
    C.question_folds = folds
    configure_blocks(K, F, C, Softmax, T, folds)
    inputs = [family_path, ROOT / 'forum/tests/r3_integrity/estimates.v2.json',
              ROOT / 'forum/tests/r3_integrity/out/absent.v2.json']
    for model in ('gpt','qwen36'):
        inputs += list((RK.DATA / 'k2' / model).glob('*'))
        provenance = json.loads((RK.DATA / 'k2' / model / 'provenance.json').read_text())
        inputs += [Path(provenance['text_counts'])]
        inputs += [D.V3R2/model/'A/attempts.parquet', D.V3R2/model/'B/attempts.parquet']
        inputs += list((D.RESULTS/model/'A').glob('*.npz'))
        inputs += [D.RESULTS/model/'A/sentences.parquet']
        inputs += list((P.LABELS_ROOT/model).glob('part-*/annotations.json'))
    inputs += [D.RESULTS/'A1/pairs.parquet',D.RESULTS/'A1/antic/antic_hist.npz']
    populations={}
    for model in ('gpt','qwen36'):
        table=pd.read_parquet(K.K2DATA/model/'pairs.parquet',columns=['question'])
        reserved=set(pd.read_parquet(D.V3R2/model/'B/attempts.parquet',columns=['question'])['question'])
        populations[model]=sorted({q for q in table['question'] if split[q] in ('dev','tune') and q not in reserved})
    frozen = seal({'schema':'r3d-anticipation-v1','inputs':{str(p):sha(p) for p in inputs if p.is_file()},
        'canonical_ids':populations,
        'code':{str(p):sha(p) for p in [Path(__file__),*(OUT/'code/rkin').glob('*.py'),*(OUT/'code/dynrt').glob('*.py')]},
        'split':'exact dev+tune before learned transforms; reserved B excluded',
        'outer':5,'inner':3,'repeats':5,'seeds':list(range(SEED,SEED+5)),
        'fold_group':'frozen duplicate family','current_class':'training-only text classifier; grouped OOF training predictions; fixed lambda=1',
        'text':'s-1,s ngrams, last token/bigram/32-token bag; earlier observed classes with missingness',
        'outcome_difficulty':'excluded from deployable baseline',
        'noise_draws_per_window_response':20,'noise':'same dimensions, standardization/tuning and fold-specific missingness as routing',
        'bootstraps':5000,'CPU_core_hour_ceiling':8.,'no_new_tensor_producer':True,
        'responses':['A1 next class','K2 switch','K2 destination'],
        'contained_sensitivity':'source n_tokens >=32 for both 16 and32 windows; reported exclusions'})
    path = OUT/'FROZEN.anticipation.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('anticipation inputs/code changed')
    path.write_text(json.dumps(frozen,indent=1)+'\n')
    results = {}
    for model in ('gpt','qwen36'):
        pairs_all = pd.read_parquet(K.K2DATA/model/'pairs.parquet')
        keep = np.array([split[q] in ('dev','tune') for q in pairs_all['question']])
        # Prepared K2 already excluded B; verify rather than relying on that history.
        reserved = set(pd.read_parquet(D.V3R2/model/'B/attempts.parquet',columns=['question'])['question'])
        keep &= ~pairs_all['question'].isin(reserved).to_numpy()
        rows = np.flatnonzero(keep)
        pairs = pairs_all.iloc[rows].reset_index(drop=True)
        cohort = Data.load_cohort(model,'A')
        base,text = augment_baseline(model,pairs,cohort,P,T)
        lex = sparse.load_npz(K.K2DATA/model/'lex_counts.npz').tocsr()[rows]
        with np.load(K.K2DATA/model/'kin.npz') as z:
            kin = {'k16':z['k16'][rows],'k32':z['k32'][rows]}
        prov=json.loads((K.K2DATA/model/'provenance.json').read_text())
        weak_text=sparse.load_npz(prov['text_counts']).tocsr()[rows]
        a,b = pairs['a'].to_numpy(int),pairs['b'].to_numpy(int)
        all_targets = [('next',np.arange(len(pairs)),b,7),
                       ('switch',np.arange(len(pairs)),(b!=a).astype(int),2),
                       ('dest',np.flatnonzero(b!=a),b,7)]
        results[model] = {}
        for target,response_rows,response_y,n_class in all_targets:
            for contained in (False,True):
                selected = response_rows[pairs['n_tokens'].to_numpy()[response_rows]>=32] if contained else response_rows
                tag = target + ('-contained' if contained else '')
                groups = pairs['question'].to_numpy()[selected]
                _,inv,cnt = np.unique(groups,return_inverse=True,return_counts=True)
                d=K.KData(model,tag,n_class,response_y[selected],a[selected],groups,1/cnt[inv],
                    dense={'base':base[selected],'weak_base':base[selected,:12],**{name:v[selected] for name,v in kin.items()},
                           'oracle_class':np.eye(7)[a[selected]]},
                    text=text[selected],lex=lex[selected],
                    outer_folds=np.stack([folds(groups,5,s) for s in range(SEED,SEED+5)]),s_text=10.,s_lex=10.)
                d.text_weak=weak_text[selected]
                if target=='next':
                    source_rows=[]
                    index={(int(cohort.table.attempt[i]),int(s.sentence_index)):i for i,s in enumerate(cohort.sent.itertuples())}
                    for row in pairs.iloc[selected].itertuples():
                        source_rows.append(index[int(row.att),int(row.sentence_index)])
                    hist=cohort.hist
                    pos=np.searchsorted(hist['last16_rows'],source_rows)
                    if not np.array_equal(hist['last16_rows'][pos],source_rows):
                        raise ValueError('A1 last16 histogram alignment failed')
                    h16=hist['last16'][pos].reshape(len(selected),-1).astype(np.float32)
                    ha=np.full_like(h16,np.nan)
                    d.has_antic=np.zeros(len(selected),bool)
                    if model=='gpt':
                        original_pairs=pd.read_parquet(D.RESULTS/'A1/pairs.parquet')
                        idx=pd.MultiIndex.from_frame(original_pairs[['attempt','sentence_index']])
                        keys=pd.DataFrame({'attempt':pairs['att'].to_numpy()[selected],
                                           'sentence_index':pairs['sentence_index'].to_numpy()[selected]})
                        ix=idx.get_indexer(pd.MultiIndex.from_frame(keys))
                        if (ix<0).any():raise ValueError('A1 antic alignment failed')
                        with np.load(D.RESULTS/'A1/antic/antic_hist.npz') as z:
                            ha=z['h'][ix].reshape(len(selected),-1).astype(np.float32)
                            d.has_antic=z['has'][ix].astype(bool)
                    d.composition=(cohort,pairs['att'].to_numpy()[selected],h16,ha)
                    targets=['composition16']+(['composition_antic'] if model=='gpt' else [])
                else: targets=['k16','k32']
                models={'base':['base','prefix_class'],'weak':['weak_sparse','weak_base','prefix_class']}
                for route in targets:
                    models[route]=['base','prefix_class',route]
                    models['oracle_'+route]=['base','oracle_class',route]
                    for draw in range(20):
                        models[f'noise_{route}_{draw}']=['base','prefix_class',f'noise_{route}_{draw}']
                predictions,diagnostics=run_cv(K,d,models,workers,frozen['sha256'],OUT/'anticipation'/model/tag)
                results[model][tag]={'status':'CLEAN','questions':len(set(groups)),'pairs':len(selected),
                    'short_sentence_exclusions':len(response_rows)-len(selected),
                    'routing':summarize(d,predictions,targets,qf),
                    'oracle_current_class':'separate saved oracle_* predictions; never an online trigger',
                    'fit_converged_fraction':float(diagnostics[:,1].mean()),
                    'A1_antic_replication':'INCOMPLETE existing cache absent' if model=='qwen36' and target=='next' else None}
                (OUT/'anticipation'/model/tag/'result.json').write_text(json.dumps(seal(results[model][tag]),indent=1)+'\n')
    output=seal({'schema':'r3d-anticipation-results-v1','models':results,'frozen':frozen['sha256'],
                 'seconds':time.monotonic()-started,'job_id':os.environ['SLURM_JOB_ID'],
                 'B1':'PENDING separate clean fold refit; historical B1 is not reused'})
    (OUT/'anticipation-results.json').write_text(json.dumps(output,indent=1)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    prepare() if args.prepare else run(args.workers)
