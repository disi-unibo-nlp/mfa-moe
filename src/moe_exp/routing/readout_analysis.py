"""Historical family-grouped readout analysis without module patching."""
from pathlib import Path
import json
import time
from moe_exp.io import digest
SEED = 20260929


def seal(value):
    return {**value, 'sha256': digest(value)}


def family_folds(questions, n, seed, question_family):
    """Shared deterministic family blocks; no label/outcome enters fold assignment."""
    import numpy as np
    families = [question_family[str(q)] for q in questions]
    order = sorted(set(families), key=lambda f: digest(f'forum-v1|{seed}|{f}'))
    by_family = {f: i % n for i, f in enumerate(order)}
    return np.array([by_family[f] for f in families], int)


def analyse(sr, md, context_counts, mode, workers, family, family_ids, output_root):
    import multiprocessing as mp
    import numpy as np
    from scipy import sparse
    started = time.monotonic()
    variants = {'aug': sr.path_variant(md, md.top1)}
    for draw in range(5):
        variants[f'null_{draw}'] = sr.path_variant(md, sr.null_paths(md, draw))
    sr._STATE.update(md=md, variants=variants, mode=mode, context_counts=context_counts)
    names = ['base', 'aug', *(f'null_{i}' for i in range(5))]
    logp = {name: np.full((md.n, 7), np.nan, np.float32) for name in names}
    checkpoints = Path(output_root) / 'readout' / md.model / mode
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
        sr._STATE.clear()
    if any(not np.isfinite(x).all() for x in logp.values()):
        raise ValueError('incomplete OOF predictions')
    losses = {name: sr.nll_rows(lp, md.y) for name, lp in logp.items()}
    with (checkpoints / 'losses.npz').open('wb') as handle:
        np.savez_compressed(handle, **losses, questions=md.groups, family_ids=family_ids,
                            classes=md.y, sentence_index=md.sent['sentence_index'].to_numpy(),
                            attempt_id=md.sent['attempt_id'].to_numpy(), folds=sr.outer_folds(md.groups, md.question_family))
    out = {'status': 'CLEAN', 'mode': mode, 'n_questions': len(set(md.groups)),
           'n_families': len(set(family_ids)), 'n_sentences': md.n, 'n_tokens': len(md.tokid),
           'penalties': penalties, 'seconds': time.monotonic() - started,
           'P_candidate_minus_baseline': -sr.question_bootstrap(losses['base'] - losses['aug'],
                                                             md.groups, md.w, 5000)['gain']}
    # Independent family cluster uncertainty; original question weighting remains the estimand.
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
