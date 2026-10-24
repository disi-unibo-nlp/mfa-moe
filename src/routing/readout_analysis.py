"""Family folds, run-spec freezing and the family-grouped fit loop shared by the readout analyses."""
from pathlib import Path
import json
import sys
import time
import store
from store import ANALYSIS, WORK, digest, file_sha, load, require, seal, verified

# The published seven-model run: its frozen specs fix the inputs a rerun must reproduce, and the LFM readout binds its files.
PUBLISHED_RUN = ANALYSIS / 'forum/tests/r3_context'
# The helper packages that read the saved caches; each takes its folder from store.ANALYSIS when first imported.
CACHE_READERS = ('routing.depth_paths.common', 'routing.dynamics_routing.common', 'routing.dynamics_routing.d3a_pairs',
                 'routing.reasoning_kinematics.common')
# Reports are not tracked in git; they live in the main checkout.
FAMILY_FREEZE = WORK / 'mfa-moe/repo/report/experimental-resume-v1/family-freeze.json'
SEED = 20260929


def use_analysis_dir(analysis_dir):
    """The folder of saved analysis caches that a seven-model driver reads (a Path); `store.ANALYSIS` by default.

    The helper packages take their input folders from `store.ANALYSIS` when they are first imported, so a folder
    other than the one they have seen must be chosen before that; a late change is refused. The frozen specs of the
    published runs name their inputs by absolute path, so only the default folder reproduces them: `freeze` stops a
    run on any other path.
    """
    global PUBLISHED_RUN
    analysis = Path(analysis_dir)
    require(analysis == store.ANALYSIS or not any(name in sys.modules for name in CACHE_READERS),
            'a helper package was imported before --analysis-dir was set')
    store.ANALYSIS = analysis
    PUBLISHED_RUN = analysis / 'forum/tests/r3_context'
    return analysis


def question_families():
    """The family freeze receipt and the map question -> duplicate family."""
    freeze = verified(FAMILY_FREEZE)
    return freeze, {q: f for f, qs in freeze['new_parent_pools']['families'].items() for q in qs}


def code_hashes(*paths):
    """sha256 of each file given and of every .py file of each folder given: the code a run spec records."""
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob('*.py')) if path.is_dir() else [path]
    return {str(path): file_sha(path) for path in sorted(files)}


def freeze(out, name, spec, check_published_run=True):
    """Write the run spec `name` into the new folder `out`.

    Its inputs must equal those of the published run (PUBLISHED_RUN/name), which must exist; only the code hashes, and
    so the seal, may differ. A spec already in `out` must be identical, so a resumed run keeps its checkpoints.
    `check_published_run=False` is for tests only."""
    if check_published_run:
        published = PUBLISHED_RUN / name
        require(published.exists(), f'the published run\'s spec is missing: {published}')
        inputs = lambda value: {k: v for k, v in value.items() if k not in ('code', 'sha256')}
        require(inputs(load(published)) == inputs(spec), f'inputs differ from the published run: {published}')
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    if path.exists():
        require(load(path) == spec, f'{path} differs from this run')
    path.write_text(json.dumps(spec, indent=1) + '\n')


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
