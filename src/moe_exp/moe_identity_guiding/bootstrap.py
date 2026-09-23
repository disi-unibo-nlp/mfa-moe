"""CPU-only paired, dataset-stratified source-problem bootstrap."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import numpy as np


def default_workers():
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return min(8, available)


def add_arguments(parser):
    parser.add_argument('--bootstrap-replicates', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    parser.add_argument('--bootstrap-workers', type=int, default=default_workers())


def _chunk(task):
    strata, seed, start, stop = task
    values = np.empty((stop-start, len(strata)+1))
    for offset, replicate in enumerate(range(start, stop)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        numerator = denominator = 0
        for j, counts in enumerate(strata):
            selected = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
            values[offset, j+1] = selected[0] / selected[1]
            numerator += selected[0]
            denominator += selected[1]
        values[offset, 0] = numerator / denominator
    return values


def paired_accuracy_bootstrap(baseline, guided, *, replicates=5000, seed=42, workers=None):
    """Inputs are matched ID->record mappings; intervals use accuracy units, not percent.

    Resample whole problems within each dataset, retaining all paired attempts.
    The pooled statistic is the ratio of summed correctness differences to
    summed attempt counts, preserving the existing attempt-weighted estimand.
    Replicate-specific RNG streams make results independent of worker count.
    """
    workers = default_workers() if workers is None else workers
    if replicates < 2 or workers < 1 or seed < 0:
        raise ValueError('Need >=2 replicates, >=1 workers, and a nonnegative seed')
    if not baseline or baseline.keys() != guided.keys():
        raise ValueError('Need nonempty matched attempt IDs')
    grouped = {}
    for key in sorted(baseline):
        a, b = baseline[key], guided[key]
        data = a['input']
        dataset = data.get('dataset')
        problem = data.get('source_problem_id') or data.get('problem_id')
        if not dataset or not problem:
            raise ValueError('Bootstrap requires dataset and source problem identity')
        if data != b['input'] or any(type(r['is_correct']) is not bool for r in (a, b)):
            raise ValueError('Bootstrap requires paired inputs and Boolean outcomes')
        count = grouped.setdefault(dataset, {}).setdefault(problem, [0, 0])
        count[0] += int(b['is_correct']) - int(a['is_correct'])
        count[1] += 1
    names = sorted(grouped)
    strata = [np.array([grouped[d][p] for p in sorted(grouped[d])], dtype=np.int64) for d in names]
    tasks = [(strata, seed, start, min(start+100, replicates)) for start in range(0, replicates, 100)]
    used = min(workers, len(tasks))
    if used == 1:
        draws = np.concatenate([_chunk(task) for task in tasks])
    else:
        with ProcessPoolExecutor(max_workers=used, mp_context=multiprocessing.get_context('spawn')) as pool:
            draws = np.concatenate(list(pool.map(_chunk, tasks)))
    def summary(column, counts):
        n = sum(len(s) for s in counts)
        return {'num_problems': n, 'num_attempts': int(sum(s[:, 1].sum() for s in counts)),
                'accuracy_delta': float(sum(s[:, 0].sum() for s in counts)/sum(s[:, 1].sum() for s in counts)),
                'ci95': np.quantile(draws[:, column], [.025, .975]).tolist() if all(len(s) >= 2 for s in counts) else None,
                'limited_problem_support': any(len(s) < 10 for s in counts)}
    return {'method': 'paired_dataset_stratified_problem_percentile', 'replicates': replicates,
            'seed': seed, 'workers': used, 'confidence_level': .95,
            'weighting': 'attempt_weighted', 'overall': summary(0, strata),
            'datasets': {d: summary(j+1, [strata[j]]) for j, d in enumerate(names)}}
