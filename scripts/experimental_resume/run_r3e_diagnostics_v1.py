"""Post-primary exploratory nested CV diagnostics for marginal-block harm."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
PKG = REPO / 'scripts/experimental_resume/r3e_clean_v2'
PRICE = REPO / 'report/experimental-resume-v1/R3E_POSTPRIMARY_DIAGNOSTIC_PRICE_v1.json'
FEATURES = S / 'runs/resume-v1/r3e-b4-features-v2'
OUT = S / 'runs/resume-v1/r3e-postprimary-diagnostics-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


class MargPCA:
    names = ('PC1', 'PC2', 'PC3')

    def __init__(self, base):
        self.base = base

    def fit_transform(self, train):
        x = self.base.fit_transform(train)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite marginal features')
        mean = x[train].mean(axis=0)
        sd = x[train].std(axis=0)
        z = (x - mean) / np.where(sd > 1e-12, sd, 1.0)
        _, _, vt = np.linalg.svd(z[train], full_matrices=False)
        return z @ vt[:3].T


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('exploratory diagnostics must run in owner Slurm allocation')
    price = sealed(PRICE)
    primary = sealed(S / 'runs/resume-v1/r3e-cv-clean-v1/full.json')
    if sha(__file__) != price['driver_sha256'] or primary['sha256'] != price['registered_primary_seal']:
        raise ValueError('diagnostic source or registered result changed')
    for path, expected in price['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'diagnostic input changed: {path}')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '8':
        raise ValueError('diagnostic CPU shape changed')
    output = OUT / 'result.json'
    if output.exists():
        raise ValueError('diagnostic output exists; inspect before retry')
    sys.path.insert(0, str(PKG))
    sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
    from dynrt import cv, d3c_cv as C
    from run_r3e_cv_clean_v1 import family_comparisons
    frozen = C.verify_frozen()
    if frozen['freeze_sha256'] != price['freeze_seal']:
        raise ValueError('feature freeze changed')
    data = np.load(FEATURES / 'features_A.npz', allow_pickle=True)
    u = C.load_universe()
    if not np.array_equal(data['attempt_id'], u['att']['attempt_id'].to_numpy()):
        raise ValueError('feature/outcome attempt IDs differ')
    keys = data['set_keys'].tolist()
    marg = C.PrecomputedBlock(keys, data['marg'], data['marg_names'].tolist())
    pa = C.PrecomputedBlock(keys, data['pa'], data['pa_names'].tolist())
    blocks = {'controls': u['controls'], 'marg': marg, 'm3': MargPCA(marg), 'pa': pa}
    models = {
        'C': ['controls'],
        'C_PA': ['controls','pa.PA'],
        'C_PAlex': ['controls','pa.PAlex'],
        'C_PAall': ['controls','pa.PA','pa.PAlex','pa.PA_sd'],
        'C_M': ['controls','marg'],
        'C_M_PAall': ['controls','marg','pa.PA','pa.PAlex','pa.PA_sd'],
        'C_M3': ['controls','m3'],
        'C_M3_PAall': ['controls','m3','pa.PA','pa.PAlex','pa.PA_sd'],
    }
    comparisons = {
        'C_plus_PA': ('C','C_PA'),
        'C_plus_PAlex': ('C','C_PAlex'),
        'C_plus_PAall': ('C','C_PAall'),
        'C_plus_M': ('C','C_M'),
        'M_plus_PAall_registered_replication': ('C_M','C_M_PAall'),
        'C_plus_M3': ('C','C_M3'),
        'M3_plus_PAall': ('C_M3','C_M3_PAall'),
        'C_plus_M3_PAall': ('C','C_M3_PAall'),
    }
    family = np.asarray([cv._families()[str(q)] for q in u['groups']])
    spec = cv.CVSpec(blocks=blocks, models=models, y=u['y'], groups=u['groups'],
                     weights=u['weights'], valid=u['valid'], repeats=5, outer=5, inner=3,
                     grid=10.0 ** np.arange(-4,5), seed0=0)
    started = time.time()
    fitted = cv.run_cv(spec, workers=8, log=lambda msg: print(msg, flush=True))
    evals = family_comparisons(fitted['preds'], u['y'], u['valid'], u['groups'], family,
                               comparisons, 5000, 20261002)
    result = {'schema': 'r3e-postprimary-exploratory-diagnostics-v1',
              'status': 'COMPLETE_EXPLORATORY', 'job_id': os.environ['SLURM_JOB_ID'],
              'price_sha256': price['sha256'], 'driver_sha256': sha(__file__),
              'registered_primary_seal': primary['sha256'], 'freeze_sha256': frozen['freeze_sha256'],
              'population': {'assigned': len(u['y']), 'scored': int(u['valid'].sum()),
                             'assigned_families': len(set(family)),
                             'scored_families': len(set(family[u['valid']]))},
              'design': {'models': models, 'comparisons': comparisons, 'repeats': 5,
                         'outer': 5, 'inner': 3, 'ridge_grid': spec.grid.tolist(),
                         'M3': 'train-set mean/SD and three PCA loadings, applied to held-out rows'},
              'family_evaluation': evals,
              'lambda_medians': {k: float(np.median(v)) for k,v in fitted['lambdas'].items()},
              'seconds': time.time() - started,
              'interpretation': 'Exploratory same-cohort diagnostic. All candidates reported; no registered primary replaced or independent gain claimed.'}
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / 'predictions.npz'
    np.savez_compressed(pred_path, attempt_id=u['att']['attempt_id'].to_numpy(),
                        question=u['groups'], family=family, y=u['y'], valid=u['valid'],
                        **{'pred_' + k: v for k,v in fitted['preds'].items()})
    result['predictions_sha256'] = sha(pred_path)
    result['sha256'] = digest(result)
    output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'seconds': result['seconds'], 'gains':
                      {k: v['gain'] for k,v in evals['comparisons'].items()}}), flush=True)


if __name__ == '__main__':
    main()
