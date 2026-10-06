"""Frozen-estimator, family-fold B4 conditional-label precision simulations.

The generator is fitted against observed outcomes only outside each fixed
generator fold. Simulated labels never alter the outcome-blind B4 features.
Five targets x 200 deterministic replicates are split into a five-replicate
smoke and twenty disjoint shards. This models estimator power, not a causal
intervention or a universal minimum detectable effect.
"""
from __future__ import annotations

import argparse
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
FEATURES = S / 'runs/resume-v1/r3e-b4-features-v2'
PRIMARY = S / 'runs/resume-v1/r3e-cv-clean-v1'
OUT = S / 'runs/resume-v1/r3e-b4-precision-v2'
PRICE = REPO / 'report/experimental-resume-v1/R3E_B4_PRECISION_PRICE_v2.json'
TARGETS = (0.0, 0.001, 0.003, 0.005, 0.010)
N_EACH = 200
N_BOOT = 1000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def load_data(price):
    sys.path.insert(0, str(PKG))
    from dynrt import cv, d3c_cv as C, d3c_features as D
    freeze = C.verify_frozen()
    if freeze['freeze_sha256'] != price['freeze_sha256']:
        raise ValueError('changed outcome-blind B4 freeze')
    with np.load(FEATURES / 'features_A.npz', allow_pickle=True) as data:
        keys = data['set_keys'].tolist()
        marg = data['marg'].copy()
        pa = data['pa'].copy()
        marg_names = data['marg_names'].tolist()
        pa_names = data['pa_names'].tolist()
        attempt_id = data['attempt_id'].copy()
    u = C.load_universe()
    if len(u['y']) != 509 or not np.array_equal(attempt_id, u['att']['attempt_id'].to_numpy()):
        raise ValueError('precision universe differs from feature freeze')
    families = np.asarray([cv._families()[str(q)] for q in u['groups']])
    if len(set(families)) != 496 or int(u['valid'].sum()) != 508 or len(set(families[u['valid']])) != 495:
        raise ValueError('precision family/outcome inventory differs')
    blocks = {'controls': u['controls'],
              'marg': C.PrecomputedBlock(keys, marg, marg_names),
              'pa': C.PrecomputedBlock(keys, pa, pa_names)}
    return cv, C, D, u, families, blocks


def bernoulli_kl(p1, p0):
    p0 = np.clip(p0, 1e-8, 1 - 1e-8)
    p1 = np.clip(p1, 1e-8, 1 - 1e-8)
    return p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))


def generator(cv, D, u, blocks, target):
    """Five out-of-generator-fold p0,p1 arrays; no held-out observed y is read."""
    n = len(u['y'])
    # Exact registered first-repeat outer folds, which are included in the
    # outcome-blind 100-set feature freeze. The generator's held-out observed
    # outcomes cannot enter its fold-specific baseline or signal calibration.
    folds = cv.question_folds(u['groups'], 5, 0)
    p0, p1 = np.full(n, np.nan), np.full(n, np.nan)
    calibration = []
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        tr = train[u['valid'][train]]
        design = cv.Design({k: cv._Wrap(v) for k, v in blocks.items()},
                           ['controls', 'marg'], n)
        x = design.matrix(train, {})
        sc = cv.Scaler.fit(x[tr])
        xs = sc.transform(x)
        beta = cv.ridge_logistic(xs[tr], u['y'][tr], u['weights'][tr], 0.01)
        base = np.clip(cv.predict_logistic(beta, xs), 1e-5, 1 - 1e-5)
        key = D.set_key(train)
        z_raw = blocks['pa'].fit_transform(train)[:, 0]
        # Ridge projection removes the signal predictable from the baseline's
        # linear design on generator-training families; standardise in-fold.
        x1 = np.column_stack([np.ones(len(tr)), xs[tr]])
        xtx = x1.T @ x1 + np.diag(np.r_[0.0, np.full(xs.shape[1], 0.1)])
        coef = np.linalg.solve(xtx + np.eye(len(xtx)) * 1e-10, x1.T @ z_raw[tr])
        residual = z_raw - np.column_stack([np.ones(n), xs]) @ coef
        sd = float(np.std(residual[tr]))
        if not np.isfinite(sd) or sd <= 1e-8:
            raise ValueError(f'unattainable conditional signal, fold {fold}: residual SD {sd}')
        z = residual / sd
        logit = np.log(base / (1 - base))
        def shifted(gamma):
            return cv._sigmoid(logit + gamma * z)
        if target == 0:
            gamma = 0.0
        else:
            lo, hi = 0.0, 1.0
            while float(bernoulli_kl(shifted(hi)[tr], base[tr]).mean()) < target and hi < 4096:
                hi *= 2
            if hi >= 4096:
                raise ValueError(f'unattainable target {target}, fold {fold}')
            for _ in range(60):
                mid = (lo + hi) / 2
                if float(bernoulli_kl(shifted(mid)[tr], base[tr]).mean()) < target:
                    lo = mid
                else:
                    hi = mid
            gamma = (lo + hi) / 2
        shifted_p = shifted(gamma)
        p0[test], p1[test] = base[test], shifted_p[test]
        calibration.append({'fold': fold, 'train_n': len(tr), 'test_n': len(test),
                            'train_family_n': len(set(u['groups'][tr])),
                            'gamma': gamma, 'signal_sd_train': sd,
                            'train_target_kl': float(bernoulli_kl(shifted_p[tr], base[tr]).mean()),
                            'heldout_oracle_kl': float(bernoulli_kl(shifted_p[test], base[test]).mean()),
                            'feature_train_key': key})
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError('generator left missing probabilities')
    return p0, p1, calibration


class StaticBlock:
    def __init__(self, values):
        self.values = values
        self.names = ['noise0', 'noise1', 'noise2']
    def fit_transform(self, train):
        return self.values


def evaluate(cv, y, preds, valid, families, seed):
    good = np.flatnonzero(valid)
    gain_r = cv.attempt_loss(y[None, good], preds['base'][:, good]) - cv.attempt_loss(
        y[None, good], preds['B4'][:, good])
    gain = gain_r.mean(0)
    noise_r = cv.attempt_loss(y[None, good], preds['base'][:, good]) - cv.attempt_loss(
        y[None, good], preds['noise'][:, good])
    uniq, inv = np.unique(families[good], return_inverse=True)
    totals = np.bincount(inv, weights=gain, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT)
    for k in range(0, N_BOOT, 200):
        draw = rng.integers(len(uniq), size=(min(200, N_BOOT-k), len(uniq)))
        boot[k:k+len(draw)] = totals[draw].sum(1) / counts[draw].sum(1)
    point = float(gain.mean())
    ci = np.percentile(boot, [2.5, 97.5])
    p_one = float((1 + np.count_nonzero(boot <= 0)) / (N_BOOT + 1))
    positive = int(np.count_nonzero(gain_r.mean(1) > 0))
    return {'gain': point, 'ci95': [float(x) for x in ci], 'one_sided_p': p_one,
            'positive_repeats': positive, 'noise_gain': float(noise_r.mean()),
            'holm_safe_reject': bool(p_one < 0.05/3 and ci[0] > 0),
            'registered_advance': bool(p_one < 0.05/3 and ci[0] > 0 and
                                       positive >= 4 and point >= 0.005)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('smoke', 'shard'))
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('precision simulations require owner CPU Slurm')
    price = sealed(PRICE)
    if sha(__file__) != price['driver_sha256']:
        raise ValueError('precision driver differs from sealed price')
    for name, expected in price['input_sha256'].items():
        if sha(name) != expected:
            raise ValueError(f'precision input differs: {name}')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '8':
        raise ValueError('precision CPU shape differs')
    if args.stage == 'smoke':
        if args.shard is not None:
            raise ValueError('smoke has no shard index')
        ids = list(range(5))
        target_path = OUT / 'smoke.json'
    else:
        if args.shard is None or not 0 <= args.shard < 20:
            raise ValueError('shard index must be 0..19')
        smoke = sealed(OUT / 'smoke.json')
        if smoke['status'] != 'PASS' or smoke['price_sha256'] != price['sha256']:
            raise ValueError('precision smoke prerequisite differs')
        ids = list(range(5 + args.shard*50, min(1000, 5 + (args.shard+1)*50)))
        target_path = OUT / f'shard-{args.shard:02}.json'
    if target_path.exists():
        raise ValueError('precision output already exists; inspect before retry')
    t0 = time.time()
    cv, C, D, u, families, blocks = load_data(price)
    generators = {t: generator(cv, D, u, blocks, t) for t in TARGETS}
    if args.stage == 'smoke':
        with np.load(PRIMARY / 'full-predictions.npz', allow_pickle=False) as saved:
            parity = cv.run_cv(cv.CVSpec(blocks=blocks, models={
                'base': C.MODEL_SPECS['base'], 'B4': C.MODEL_SPECS['B4']},
                y=u['y'], groups=u['groups'], weights=u['weights'], valid=u['valid'],
                repeats=5, outer=5, inner=3, seed0=0), workers=8, log=lambda _: None)
            for name in ('base', 'B4'):
                if not np.allclose(parity['preds'][name], saved['pred_'+name], atol=1e-12, rtol=0):
                    raise ValueError(f'frozen CV parity fails for {name}')
    rows = []
    for sim_id in ids:
        target = TARGETS[sim_id // N_EACH]
        p0, p1, _ = generators[target]
        rng = np.random.default_rng(202610020000 + sim_id)
        y = (rng.random(len(p1)) < p1).astype(float)
        noise = rng.standard_normal((len(y), 3))
        sim_blocks = {**blocks, 'noise': StaticBlock(noise)}
        models = {'base': C.MODEL_SPECS['base'], 'B4': C.MODEL_SPECS['B4'],
                  'noise': C.MODEL_SPECS['base'] + ['noise']}
        spec = cv.CVSpec(blocks=sim_blocks, models=models, y=y, groups=u['groups'],
                         weights=u['weights'], valid=u['valid'], repeats=5,
                         outer=5, inner=3, seed0=0)
        pred = cv.run_cv(spec, workers=8, log=lambda _: None)['preds']
        ev = evaluate(cv, y, pred, u['valid'], families, 202610030000 + sim_id)
        rows.append({'sim_id': sim_id, 'target_kl': target, 'realized_oracle_kl': float(
            bernoulli_kl(p1[u['valid']], p0[u['valid']]).mean()), **ev})
        if len(rows) % 10 == 0:
            print(json.dumps({'completed': len(rows), 'last_sim': sim_id,
                              'seconds': time.time()-t0}), flush=True)
    body = {'schema': 'r3e-b4-calibrated-precision-v2', 'status': 'PASS',
            'stage': args.stage, 'shard': args.shard, 'job_id': os.environ['SLURM_JOB_ID'],
            'price_sha256': price['sha256'], 'feature_freeze_sha256': price['freeze_sha256'],
            'population': {'assigned': 509, 'families': 496, 'scored': 508,
                           'scored_families': 495, 'missing_original_outcome': 1},
            'generator': {'baseline': 'in-generator-training-fold C+28 ridge logistic lambda=.01',
                          'signal': 'PA residualized against baseline linear design in generator training fold; sign fixed positive',
                          'calibration': 'train-family mean Bernoulli KL(p1||p0) targets',
                          'folds': {str(t): generators[t][2] for t in TARGETS},
                          'outcomes_independent_conditional_on_features': True,
                          'residual_family_dependence_simulated': False},
            'simulation': {'targets': TARGETS, 'replicates_per_target': N_EACH,
                           'bootstrap_per_replicate': N_BOOT,
                           'fixed_original_duplicate_family_folds': True,
                           'noise_control_columns': 3,
                           'rows': rows}, 'seconds': time.time()-t0}
    body['sha256'] = digest(body)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(target_path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    tmp.write_text(json.dumps(body, indent=1) + '\n')
    tmp.replace(target_path)
    print(json.dumps({'result': str(target_path), 'n': len(rows),
                      'seconds': body['seconds']}), flush=True)


if __name__ == '__main__':
    main()
