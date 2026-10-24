"""Present-state readout of the seven captured models: does routing tell a sentence's class beyond its text?

Family-grouped refits from saved caches only: the token caches `_tok` and the saved sentence tables of the
capture. Per model it fits, for each of five outer family folds, a baseline, the routing-augmented model and five
token-permutation nulls, with three inner folds choosing the penalty, twice (token identity only, and with the
judge's previous/current/next sentence text). A model whose cache is incomplete is reported INCOMPLETE and nothing
is produced for it. Takes about 5 hours on 4 CPUs (19 core-hours in all).

    python -m routing.readout --out DIR [--workers 4]

DIR must be new: it gets FROZEN.readout.v3.json (the run's inputs and the code that ran), the fold checkpoints,
one JSON per model and readout-results.v3.json. The inputs are compared with the published run's spec in
PUBLISHED_RUN; that run's own files are bound by the LFM readout and are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import store
from models.roster import READOUT_ORDER
from routing.readout_analysis import FAMILY_FREEZE, SEED, code_hashes, family_folds, freeze, question_families
from store import ANALYSIS, file_sha, require, seal

MODEL_ORDER = READOUT_ORDER  # the seven captured models, in the order of the frozen specification


def contexts(md, attempts, load_unit_texts, CLASSES):
    """Exact sentence indices: never substitute the next labelled row across a gap."""
    keys = {(r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256)
            for r in attempts.itertuples()}
    units = load_unit_texts(md.model, keys)
    records = attempts[['dataset', 'problem_id', 'sample_id', 'trace_sha256']].to_dict('records')
    texts, missing = [], [0, 0, 0]
    for row in md.sent.itertuples():
        record = records[int(row.row)]
        key = (record['dataset'], record['problem_id'], int(record['sample_id']), record['trace_sha256'])
        parts = []
        for j, offset in enumerate((-1, 0, 1)):
            unit = units.get(key + (int(row.sentence_index) + offset,))
            missing[j] += int(unit is None)
            if offset == 0 and unit is None:
                raise FileNotFoundError(f'current sentence text absent: {md.model}/{row.attempt_id}/{row.sentence_index}')
            if offset == 0 and unit[1] != CLASSES[int(row.cls)]:
                raise ValueError('sentence text label identity mismatch')
            parts.append(('previous' if offset == -1 else 'current' if offset == 0 else 'future')
                         + ' ' + (unit[0] if unit is not None else '<missing>'))
        texts.append('\n'.join(parts))
    return texts, missing


def analyse(sr, md, context_counts, mode, workers, family, family_ids, directory):
    import multiprocessing as mp
    import numpy as np
    from scipy import sparse
    from routing.dynrt.d3a_text import TextTransform
    started = time.monotonic()
    original_design = sr.fold_design

    def fold_design(*args, **kwargs):
        data, pv, tr, evs = args[:4]
        xd, xs, ed, es, info = original_design(*args, **kwargs)
        if mode == 'judge_context':
            transform = TextTransform.fit(context_counts, tr, 10.)
            xs = sparse.hstack([xs, transform.transform(context_counts, tr)], format='csr')
            es = [sparse.hstack([x, transform.transform(context_counts, rows)], format='csr')
                  for x, rows in zip(es, evs)]
            info['context_columns'] = transform.n_active
        return xd, xs, ed, es, info

    sr.fold_design = fold_design
    variants = {'aug': sr.path_variant(md, md.top1)}
    for draw in range(5):
        variants[f'null_{draw}'] = sr.path_variant(md, sr.null_paths(md, draw))
    sr._STATE.update(md=md, variants=variants)
    names = ['base', 'aug', *(f'null_{i}' for i in range(5))]
    logp = {name: np.full((md.n, 7), np.nan, np.float32) for name in names}
    checkpoints = directory / 'readout' / md.model / mode
    checkpoints.mkdir(parents=True, exist_ok=True)
    tasks = []
    statistics, penalties = [], {}

    def accept(result):
        name, fold, rows = result['variant'], result['fold'], result['test']
        logp[name][rows] = result['logp']
        statistics.extend(result['stats'])
        penalties[f'{name}:{fold}'] = result['lam']

    for name in names:
        for fold in range(5):
            path = checkpoints / f'{name}-{fold}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z['binding']) != family['sha256']:
                        raise ValueError('checkpoint binding mismatch')
                    accept(dict(variant=name, fold=fold, test=z['test'], logp=z['logp'],
                                stats=z['stats'], lam=float(z['lam'])))
            else:
                tasks.append((name, fold))
    try:
        with mp.get_context('fork').Pool(workers) as pool:
            for result in pool.imap_unordered(sr.run_task, tasks):
                accept(result)
                path = checkpoints / f"{result['variant']}-{result['fold']}.npz"
                pending = path.with_suffix('.pending')
                with pending.open('wb') as handle:
                    np.savez_compressed(handle, binding=family['sha256'], test=result['test'],
                        logp=result['logp'], stats=np.asarray(result['stats']), lam=result['lam'])
                pending.replace(path)
                print(f"{md.model}/{mode}: {result['variant']} fold {result['fold']} complete", flush=True)
    finally:
        sr.fold_design = original_design
        sr._STATE.clear()
    if any(not np.isfinite(x).all() for x in logp.values()):
        raise ValueError('incomplete OOF predictions')
    losses = {name: sr.nll_rows(lp, md.y) for name, lp in logp.items()}
    with (checkpoints / 'losses.npz').open('wb') as handle:
        np.savez_compressed(handle, **losses, questions=md.groups, family_ids=family_ids,
                            classes=md.y, sentence_index=md.sent['sentence_index'].to_numpy(),
                            attempt_id=md.sent['attempt_id'].to_numpy(), folds=sr.outer_folds(md.groups))
    out = {'status': 'CLEAN', 'mode': mode, 'n_questions': len(set(md.groups)),
           'n_families': len(set(family_ids)), 'n_sentences': md.n, 'n_tokens': len(md.tokid),
           'penalties': penalties, 'seconds': time.monotonic() - started,
           'P_candidate_minus_baseline': -sr.question_bootstrap(losses['base'] - losses['aug'],
                                                             md.groups, md.w, 5000)['gain']}
    # Family-cluster uncertainty; the question-weighted mean remains the estimand.
    uniq, inv = np.unique(family_ids, return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=5000)
    denominator = np.bincount(inv, weights=md.w)
    null_mean = np.mean([losses[f'null_{i}'] for i in range(5)], axis=0)
    for label, values in [('routing', losses['aug'] - losses['base']),
                          ('routing_excess_over_lexical_null', losses['aug'] - null_mean)]:
        sums = np.bincount(inv, weights=md.w * values)
        boot = (weights @ sums) / (weights @ denominator)
        out[label] = {'P': float(sums.sum() / denominator.sum()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'simultaneous_ci7': np.quantile(boot, [.025 / 7, 1 - .025 / 7]).tolist(),
            'p': float(min(1., 2 * min((1 + (boot <= 0).sum()) / 5001,
                                      (1 + (boot >= 0).sum()) / 5001)))}
    st = np.asarray(statistics, float)
    out['fit_diagnostics'] = {'n_fits': len(st), 'fraction_converged': float(st[:, 1].mean()),
                              'maximum_gradient': float(st[:, 2].max())}
    (checkpoints / 'result.json').write_text(json.dumps(seal(out), indent=1) + '\n')
    return out


def run(workers, out):
    # SLURMD_NODENAME is set in batch jobs and srun steps (also on the serial partition, whose nodes are login08 and
    # login13), but not in the interactive shell that salloc starts on a login node.
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURMD_NODENAME'), 'Clean readouts require a CPU Slurm batch job')
    out = Path(out)
    require(not (out / 'readout-results.v3.json').exists(), f'{out} already holds readout results; use a new --out')
    started = time.monotonic()
    import numpy as np
    from routing.dp import common as cm, s3_readout as sr
    # The token caches are read from their saved, read-only location, not from the helper's own results folder.
    def token_cache_path(model, row):
        return cm.TOKEN_CACHE / model / f'{int(row):05d}.npz'
    from routing.dynrt.d3a_pairs import load_unit_texts, LABELS_ROOT
    from routing.dynrt.d3a_text import hash_counts
    family, qf = question_families()
    split = cm.load_split()
    load_attempts = cm.load_attempts
    tables, maps, missing, inputs = {}, {}, {}, [FAMILY_FREEZE, cm.SPLIT_PATH,
        ANALYSIS / 'forum/tests/r3_integrity/estimates.v2.json',
        ANALYSIS / 'forum/tests/r3_integrity/out/ids.v2.json',
        ANALYSIS / 'forum/tests/r3_integrity/out/absent.v2.json',
        ANALYSIS / 'forum/tests/r3_dynamics_validity/estimates.json']
    for model in MODEL_ORDER:
        attempts = load_attempts(model, 'A')
        mask = np.array([split[q] in ('dev', 'tune') for q in attempts['question']])
        tables[model] = attempts.loc[mask].reset_index(drop=True)
        maps[model] = np.flatnonzero(mask)
        cache_paths = [token_cache_path(model, int(i)) for i in maps[model]]
        absent = [str(p) for p in cache_paths if not p.exists()]
        missing[model] = absent
        inputs.extend([cm.V3R2 / model / 'A/attempts.parquet',
                       cm.DYN_RESULTS / model / 'A/sentences.parquet',
                       cm.FAST / 'manifests' / model / 'A/ready.json'])
        inputs.extend(p for p in cache_paths if p.exists())
        inputs.extend(sorted((LABELS_ROOT / model).glob('part-*/annotations.json')))
    here = Path(__file__).parent
    specification = seal({'schema': 'r3d-clean-readout-v1', 'part': 'R3-D S3 only',
        'parent_family_freeze': family['sha256'],
        'models': MODEL_ORDER, 'ids': {m: tables[m]['question'].tolist() for m in MODEL_ORDER},
        'cache_row_map': {m: maps[m].tolist() for m in MODEL_ORDER}, 'missing': missing,
        'inputs': {str(p): file_sha(p) for p in inputs if p.exists()},
        'code': code_hashes(__file__, here / 'dp', here / 'dynrt', here / 'readout_analysis.py', store.__file__),
        'folds': {'outer': 5, 'inner': 3, 'seed': SEED, 'group': 'frozen duplicate family',
                  'order': 'SHA256 of canonical JSON forum-v1|seed|family; round-robin'},
        'bootstraps': 5000, 'nulls': 5, 'ceiling_CPU_core_hours': 8.,
        'context': 'retrospective s-1,s,s+1; explicit missing slots; no sparse-gap joins',
        'remaining_R3D_core_hours_reserved': 8.})
    freeze(out, 'FROZEN.readout.v3.json', specification)
    cm.load_attempts = lambda m, c: tables[m] if c == 'A' else load_attempts(m, c)
    sr.tok_path = lambda m, i: token_cache_path(m, int(maps[m][i]))
    sr.cv.question_folds = lambda questions, n, seed: family_folds(questions, n, seed, qf)
    sr.FOLD_SEED = SEED
    results = {}
    for model in MODEL_ORDER:
        if missing[model]:
            results[model] = {'status': 'INCOMPLETE', 'missing_cache_count': len(missing[model])}
            continue
        md = sr.load_model(model)
        if not set(md.groups) <= {q for q, s in split.items() if s in ('dev', 'tune')}:
            raise ValueError('confirm exposure in clean readout')
        texts, missing_context = contexts(md, tables[model], load_unit_texts, cm.CLASSES)
        counts = hash_counts(texts)
        family_ids = np.array([qf[q] for q in md.groups])
        result = {'missing_context_slots': missing_context}
        for mode in ('token_identity', 'judge_context'):
            result[mode] = analyse(sr, md, counts, mode, workers, specification, family_ids, out)
        results[model] = result
        (out / 'readout' / f'{model}.json').write_text(json.dumps(seal(result), indent=1) + '\n')
        del md, counts
    result = seal({'schema': 'r3d-readout-results-v1', 'frozen': specification['sha256'],
        'models': results, 'seconds': time.monotonic() - started,
        'job_id': os.environ['SLURM_JOB_ID'], 'CPUs': workers,
        'core_hours_upper_estimate': workers * (time.monotonic() - started) / 3600,
        'other_R3D_cells': 'PENDING separate anticipation/B1 driver; not satisfied by S3 readout'})
    (out / 'readout-results.v3.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'status': 'S3 readout finished', 'models': list(results), 'job_id': result['job_id']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Seven-model present-state readout.')
    p.add_argument('--out', required=True, help='new directory for the spec, checkpoints and results')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    run(args.workers, args.out)
