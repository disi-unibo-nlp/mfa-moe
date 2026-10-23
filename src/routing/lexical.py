"""Lexical state readout for intervention continuations (CPU; replaces judge relabelling).

Classifier (validated on 2026-10-17 by the L1-L4 analyses, scratchpad/lexicon): judge-equivalent text
inputs only (sentence, previous sentence, next sentence; no position), hashed word 1-2-grams and character
3-5-grams of the sentence (`readout_text`), hashed word 1-2-grams of the neighbours at block scale 0.5,
fold-fitted sublinear tf-idf, nine standardised format features at scale 0.3, multinomial L2 logistic
regression (C=10, L-BFGS). Trained on judge-labelled native discovery sentences pooled over models, and
CROSS-FITTED by the frozen family folds: continuations of fold f are read by a classifier that never saw a
family of fold f. Per-class sensitivity/false-positive rates come from the same out-of-fold predictions on
native text (for Rogan-Gladen sensitivity analyses).
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
from scipy import optimize, sparse
from routing import readout_text as RT
from store import CLASSES, check_files, file_sha, publish, require, verified

CIDX = {c: i for i, c in enumerate(CLASSES)}
MAXC = 600
START, END, WEND = '<START OF RESPONSE>', '<END OF RESPONSE>', '<END OF OBSERVED WINDOW>'
MATH_RE = re.compile(r"[\$\\^_=+\-*/()<>|{}]")
BUL_RE = re.compile(r"^\s*([*\-•]|\d+[.)])\s")
LEXICAL = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/final-routing-20261015-r1/lexical')
SPEC = {'version': 'lexical-readout-v1', 'C': 10.0, 'context_scale': 0.5, 'dense_scale': 0.3, 'max_chars': MAXC,
        'per_model_cap': 60000, 'buckets': RT.N_BUCKETS, 'min_df': RT.MIN_DF}


def clip(text):
    return text if len(text) <= MAXC else text[:450]+' '+text[-150:]


def word_ids(text, prefix):
    words = RT.TOKEN_RE.findall(text.lower())
    ids = [RT._bucket(prefix+'1|', w) for w in words]
    return ids+[RT._bucket(prefix+'2|', words[i]+' '+words[i+1]) for i in range(len(words)-1)]


def counts(texts, kind):
    """Hashed counts; identical texts hashed once."""
    uniq, codes = {}, []
    for t in texts:
        codes.append(uniq.setdefault(t, len(uniq)))
    rows, cols = [], []
    for i, t in enumerate(uniq):
        t = clip(t)
        ids = RT.ngrams(t) if kind == 'cur' else word_ids(t, kind[0])
        rows.append(np.full(len(ids), i, np.int64))
        cols.append(np.asarray(ids, np.int64))
    n = len(uniq)
    if not n:
        return sparse.csr_matrix((0, RT.N_BUCKETS), dtype=np.float32)
    r, c = np.concatenate(rows), np.concatenate(cols)
    m = sparse.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, RT.N_BUCKETS)).tocsr()
    m.sum_duplicates()
    m.sort_indices()
    return m[np.asarray(codes)]


def blocks(sentence, previous, following):
    """sentence/previous/following: lists of strings exactly as the judge saw them (sentinels allowed)."""
    pstart = np.array([p == START for p in previous])
    nend = np.array([n in (END, WEND) for n in following])
    previous = ['' if p == START else p for p in previous]
    following = ['' if n in (END, WEND) else n for n in following]
    t = [s or '' for s in sentence]
    nch = np.array([len(s) for s in t], float)
    def frac(pattern):
        return np.array([len(re.findall(pattern, s)) for s in t], float)/np.maximum(nch, 1)
    dense = np.c_[np.log1p(nch)/8, np.log1p(np.array([len(re.findall(r'\w+', s)) for s in t], float))/6,
                  frac(r'\d'), frac(MATH_RE.pattern), [s.rstrip().endswith('?') for s in t],
                  [s.rstrip().endswith(':') for s in t], [bool(BUL_RE.match(s)) for s in t],
                  pstart, nend].astype(np.float32)
    return counts(t, 'cur'), counts(previous, 'p'), counts(following, 'n'), dense


class Featurizer:
    def fit(self, b, idx):
        cur, prv, nxt, dense = b
        self.tc = RT.TextTransform.fit(cur, idx, 1.0)
        self.tp = RT.TextTransform.fit(prv, idx, SPEC['context_scale'])
        self.tn = RT.TextTransform.fit(nxt, idx, SPEC['context_scale'])
        self.mu, self.sd = dense[idx].mean(0), dense[idx].std(0)+1e-6
        return self

    def transform(self, b, idx):
        cur, prv, nxt, dense = b
        idx = np.asarray(idx)
        d = (dense[idx]-self.mu)/self.sd*SPEC['dense_scale']
        return sparse.hstack([self.tc.transform(cur, idx), self.tp.transform(prv, idx), self.tn.transform(nxt, idx),
                              sparse.csr_matrix(d)], format='csr', dtype=np.float64)


class LR:
    """Multinomial L2 logistic regression (weighted), L-BFGS with a wall-clock budget."""
    def __init__(self, C=10.0, k=7):
        self.C, self.k = C, k

    def fit(self, X, y, w, budget=1800, maxiter=500):
        n, d = X.shape
        k = self.k
        sw = w.sum()
        lam = 1.0/(self.C*sw)
        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1.0
        XT = X.T.tocsr()
        x0 = np.zeros(d*k+k)
        x0[d*k:] = np.log(np.bincount(y, weights=w, minlength=k)/sw+1e-9)

        def fg(x):
            W, b = x[:d*k].reshape(d, k), x[d*k:]
            Z = X@W+b
            m = Z.max(1, keepdims=True)
            logp = Z-(m+np.log(np.exp(Z-m).sum(1, keepdims=True)))
            loss = -(w*logp[np.arange(n), y]).sum()/sw+.5*lam*(W*W).sum()
            G = (np.exp(logp)-Y)*(w/sw)[:, None]
            return loss, np.concatenate([(XT@G+lam*W).ravel(), G.sum(0)])
        started, state = time.time(), {'x': x0}

        class Stop(Exception):
            pass

        def callback(xk):
            state['x'] = xk.copy()
            if time.time()-started > budget:
                raise Stop()
        try:
            r = optimize.minimize(fg, x0, jac=True, method='L-BFGS-B', callback=callback,
                                  options={'maxiter': maxiter, 'maxcor': 20, 'ftol': 1e-9, 'gtol': 1e-6})
            self.x, self.converged, self.iterations = r.x, bool(r.success), int(r.nit)
        except Stop:
            self.x, self.converged, self.iterations = state['x'], False, -1
        self.d = d
        return self

    def proba(self, X):
        Z = X@self.x[:self.d*self.k].reshape(self.d, self.k)+self.x[self.d*self.k:]
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E/E.sum(1, keepdims=True)


def training_table(models, cap=SPEC['per_model_cap'], seed=20261017):
    import pandas as pd
    cols = ['model', 'context_hash', 'n_rows_with_context', 'label', 'sentence_text', 'prev_sentence',
            'next_sentence', 'family_id', 'fold_campaign']
    rng = np.random.default_rng(seed)
    parts = []
    for m in models:
        d = pd.read_parquet(LEXICAL/f'native_{m}.parquet', columns=cols)
        d = d[d.label.isin(CLASSES)].drop_duplicates('context_hash')
        d = d.iloc[np.sort(rng.permutation(len(d))[:cap])]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fit_folds(models, out, budget=1800):
    """Cross-fitted classifiers (one per held-out fold) and out-of-fold native confusion per model."""
    import pickle
    table = training_table(models)
    b = blocks(table.sentence_text.fillna('').tolist(), table.prev_sentence.fillna('').tolist(),
               table.next_sentence.fillna('').tolist())
    y = table.label.map(CIDX).to_numpy()
    w = 1.0/table.n_rows_with_context.to_numpy(float)
    fold = table.fold_campaign.to_numpy()
    oof = np.zeros((len(table), 7))
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    for f in range(5):
        train, test = np.where(fold != f)[0], np.where(fold == f)[0]
        tick = time.time()
        fz = Featurizer().fit(b, train)
        model = LR(SPEC['C']).fit(fz.transform(b, train), y[train], w[train], budget=budget)
        oof[test] = model.proba(fz.transform(b, test))
        path = out/f'fold-{f}.pkl'
        with path.open('wb') as stream:
            pickle.dump({'featurizer': fz, 'model': model, 'spec': SPEC, 'fold': f}, stream)
        receipts.append({'fold': f, 'path': str(path), 'sha256': file_sha(path), 'train_rows': len(train),
                         'converged': model.converged, 'iterations': model.iterations, 'seconds': time.time()-tick})
    pred = oof.argmax(1)
    quality = {}
    for m in models + ['pooled']:
        sel = np.ones(len(table), bool) if m == 'pooled' else (table.model.to_numpy() == m)
        cm = np.bincount(y[sel]*7+pred[sel], minlength=49).reshape(7, 7)
        sens = np.diag(cm)/np.maximum(cm.sum(1), 1)
        fpr = (cm.sum(0)-np.diag(cm))/np.maximum(cm.sum()-cm.sum(1), 1)
        quality[m] = {'rows': int(sel.sum()), 'accuracy': float((pred[sel] == y[sel]).mean()),
                      'macro_f1': float(np.mean([2*cm[i, i]/max(1, cm[i].sum()+cm[:, i].sum()) for i in range(7)])),
                      'confusion': cm.tolist(), 'sensitivity': dict(zip(CLASSES, sens.tolist())),
                      'false_positive_rate': dict(zip(CLASSES, fpr.tolist())),
                      'youden_j': dict(zip(CLASSES, (sens-fpr).tolist()))}
    return publish(out/'READOUT.json', {'spec': SPEC, 'models': models, 'folds': receipts, 'native_out_of_fold': quality,
                   'inputs': {m: file_sha(LEXICAL/f'native_{m}.parquet') for m in models},
                   'judge_relabelling_of_interventions': False})


class _Unpickler:
    """Fits made with `python -m routing.lexical` pickled these classes under `__main__`."""
    def __init__(self, stream):
        import pickle
        import sys

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__' and name in ('Featurizer', 'LR'):
                    return getattr(sys.modules[__name__], name)
                return super().find_class(module, name)
        self.inner = Unpickler(stream)

    def load(self):
        return self.inner.load()


def artifact_binding(root: Path) -> dict:
    """Verify the deployed C10 READOUT and every fold before loading pickles.

    Args:
        root: Deployed readout directory; candidate artifacts are not substituted.

    Returns:
        Physical-file binding suitable for an immutable cohort descriptor.
    """
    root = root.resolve()
    receipt = verified(root/'READOUT.json')
    require(receipt['spec'] == SPEC, 'deployed C10 readout specification differs')
    folds = receipt['folds']
    require(len(folds) == 5 and {row['fold'] for row in folds} == set(range(5)),
            'readout requires exactly five valid fold models')
    files = {str(root/'READOUT.json'): file_sha(root/'READOUT.json')}
    for row in folds:
        path = root/f"fold-{row['fold']}.pkl"
        require(path.is_file() and file_sha(path) == row['sha256'], 'readout fold bytes changed or missing')
        files[str(path)] = row['sha256']
    return {'root': str(root), 'readout_sha256': receipt['sha256'], 'files': files}


def predict(sentences, root, *, frozen=None):
    """Add p_<class> and lexical_label to a continuation sentence table (columns: fold, sentence_text,
    prev_sentence, next_sentence); each row read by the classifier of its own held-out fold."""
    binding = artifact_binding(root)
    if frozen is not None:
        require(binding == frozen, 'cohort classifier artifact changed')
    sentences = sentences.reset_index(drop=True)
    require(sentences.fold.notna().all() and sentences.fold.isin(range(5)).all(),
            'sentence rows are not covered by valid classifier folds')
    if sentences.empty:
        for c in CLASSES:
            sentences['p_'+c] = np.array([], dtype=float)
        sentences['lexical_label'] = []
        return sentences
    b = blocks(sentences.sentence_text.fillna('').tolist(), sentences.prev_sentence.fillna('').tolist(),
               sentences.next_sentence.fillna('').tolist())
    P = np.zeros((len(sentences), 7))
    for f in range(5):
        idx = np.where(sentences.fold.to_numpy() == f)[0]
        if not len(idx):
            continue
        with (root/f'fold-{f}.pkl').open('rb') as stream:
            saved = _Unpickler(stream).load()
        require(saved['spec'] == SPEC and saved['fold'] == f, 'lexical readout specification differs')
        P[idx] = saved['model'].proba(saved['featurizer'].transform(b, idx))
    require(P.shape == (len(sentences), len(CLASSES)) and np.isfinite(P).all()
            and (P >= 0).all() and (P <= 1).all() and np.allclose(P.sum(1), 1., atol=1e-7),
            'invalid classifier probabilities; rows cannot be mapped to Read')
    check_files(binding['files'])
    for i, c in enumerate(CLASSES):
        sentences['p_'+c] = P[:, i]
    sentences['lexical_label'] = [CLASSES[i] for i in P.argmax(1)]
    return sentences


def main(argv=None):
    ap = argparse.ArgumentParser(prog='routing.lexical')
    ap.add_argument('command', choices=['fit'])
    ap.add_argument('--models', default='gpt,nemotron,qwen35,qwen36,gemma,glm,qwen330b')
    ap.add_argument('--out', default=str(LEXICAL/'readout-v1'))
    ap.add_argument('--budget', type=float, default=1800)
    args = ap.parse_args(argv)
    value = fit_folds(args.models.split(','), Path(args.out), args.budget)
    print(json.dumps({k: v for k, v in value['native_out_of_fold'].items()
                      if k == 'pooled'} | {'folds': value['folds']}, default=str)[:3000])


if __name__ == '__main__':
    main()
