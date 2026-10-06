"""Confirm-connected-family exclusion B4 nested correctness CV sensitivity."""
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
PKG = REPO / 'scripts/experimental_resume/r3e_clean_v3'
PRICE = REPO / 'report/experimental-resume-v1/R3E_CV_CLEAN_CONFIRM_PRICE_v1.json'
FEATURES = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/resume-v1/r3e-b4-features-confirm-exclusion-v3')
OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/resume-v1/r3e-cv-clean-confirm-v1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def family_comparisons(preds, y, valid, questions, families, comparisons, n_boot, seed):
    """Question-average point, family-resampled ratio CI, simultaneous max-t CI."""
    from dynrt.cv import attempt_loss
    uniq, inv = np.unique(families[valid], return_inverse=True)
    n_fam = len(uniq)
    point = []
    sums = []
    counts = np.bincount(inv, minlength=n_fam).astype(float)
    details = {}
    for name, (base, aug) in comparisons.items():
        diff = (attempt_loss(y[None, valid], preds[base][:, valid]) -
                attempt_loss(y[None, valid], preds[aug][:, valid])).mean(0)
        val = float(np.mean(diff))
        point.append(val)
        sums.append(np.bincount(inv, weights=diff, minlength=n_fam))
        details[name] = {'gain': val, 'per_repeat_gain': [float(np.mean(
            attempt_loss(y[valid], preds[base][r, valid]) -
            attempt_loss(y[valid], preds[aug][r, valid]))) for r in range(preds[base].shape[0])]}
    totals = np.stack(sums, axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(comparisons)))
    for start in range(0, n_boot, 250):
        stop = min(start + 250, n_boot)
        draw = rng.integers(n_fam, size=(stop - start, n_fam))
        boot[start:stop] = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None]
    point = np.asarray(point)
    sd = np.std(boot, axis=0, ddof=1)
    max_t = np.max(np.abs((boot - point) / np.maximum(sd, 1e-12)), axis=1)
    critical = float(np.percentile(max_t, 95))
    for j, name in enumerate(comparisons):
        details[name].update(ci95=[float(x) for x in np.percentile(boot[:, j], [2.5, 97.5])],
                             simultaneous_ci95=[float(point[j] - critical * sd[j]),
                                                 float(point[j] + critical * sd[j])],
                             bootstrap_sd=float(sd[j]),
                             bootstrap_one_sided_p=float((1 + (boot[:, j] <= 0).sum()) / (n_boot + 1)),
                             positive_repeats=int(sum(x > 0 for x in details[name]['per_repeat_gain'])))
    return {'family_count': n_fam, 'question_count': int(valid.sum()), 'bootstrap': n_boot,
            'seed': seed, 'simultaneous_family_size': len(comparisons),
            'simultaneous_max_t_critical': critical, 'comparisons': details,
            'inference_note': 'Family-clustered percentile and max-t bootstrap of cross-fitted prediction-loss differences; not exact randomization inference.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('smoke', 'full'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('clean correctness CV must run under owner Slurm allocation')
    price = sealed(PRICE)
    if sha(__file__) != price['driver_sha256']:
        raise ValueError('CV driver differs from sealed price')
    for path, expected in price['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'CV input changed: {path}')
    stage = price['stages'][args.stage]
    if str(stage['cpus']) != os.environ.get('SLURM_CPUS_PER_TASK'):
        raise ValueError('CV Slurm CPU shape differs from price')
    if args.stage == 'full':
        smoke = sealed(OUT / 'smoke.json')
        if smoke['status'] != 'PASS' or smoke['freeze_sha256'] != price['freeze_seal']:
            raise ValueError('CV smoke prerequisite differs')
    target = OUT / (args.stage + '.json')
    if target.exists():
        raise ValueError('CV stage output exists; inspect rather than silently rerunning')
    sys.path.insert(0, str(PKG))
    from dynrt import cv, d3c_cv as C
    freeze = C.verify_frozen()
    if freeze['freeze_sha256'] != price['freeze_seal']:
        raise ValueError('B4 feature freeze differs from CV price')
    # The frozen pandas attempt IDs and string feature names are object arrays;
    # verify_frozen has checked the file SHA before this trusted local load.
    data = np.load(FEATURES / 'features_A.npz', allow_pickle=True)
    u = C.load_universe()
    if not np.array_equal(data['attempt_id'], u['att']['attempt_id'].to_numpy()):
        raise ValueError('CV outcome rows differ from frozen feature rows')
    if len(u['y']) != 483 or len(set(u['groups'])) != 483:
        raise ValueError('confirm-connected-family exclusion population changed')
    keys = data['set_keys'].tolist()
    blocks = {'controls': u['controls'],
              'marg': C.PrecomputedBlock(keys, data['marg'], data['marg_names'].tolist()),
              'pa': C.PrecomputedBlock(keys, data['pa'], data['pa_names'].tolist())}
    specs = {k: v for k, v in C.MODEL_SPECS.items() if k != 'B4_readingB'}
    comparisons = {k: v for k, v in C.COMPARISONS.items() if k != 'sensitivity_PAlex_reading_B'}
    folds = 5
    repeats = stage['repeats']
    spec = cv.CVSpec(blocks=blocks, models=specs, y=u['y'], groups=u['groups'],
                     weights=u['weights'], valid=u['valid'], repeats=repeats, outer=folds,
                     inner=3, seed0=0)
    t0 = time.time()
    result = cv.run_cv(spec, workers=stage['cpus'], log=lambda msg: print(msg, flush=True))
    families = np.asarray([cv._families()[str(q)] for q in u['groups']])
    evaluation = family_comparisons(result['preds'], u['y'], u['valid'], u['groups'], families,
                                    comparisons, stage['bootstrap'], 20260929)
    p = evaluation['comparisons']['B4_primary']
    output = {'schema': 'r3e-clean-b4-confirm-exclusion-correctness-cv-v1', 'status': 'PASS',
              'stage': args.stage, 'job_id': os.environ['SLURM_JOB_ID'],
              'price_sha256': price['sha256'], 'driver_sha256': sha(__file__),
              'freeze_sha256': freeze['freeze_sha256'],
              'population': {'assigned_questions': 483, 'scored_questions': int(u['valid'].sum()),
                             'missing_outcome_questions': int((~u['valid']).sum()),
                             'families': int(len(set(families))),
                             'capped': int(u['att']['capped'].sum())},
              'design': {'repeats': repeats, 'outer': folds, 'inner': 3,
                         'family_folds': True, 'models': specs,
                         'comparisons': comparisons,
                         'lexical_method': 'prospective training-fold-only reading-B score'},
              'evaluation': evaluation,
              'sensitivity': {'gain_nats_per_attempt': p['gain'],
                          'family_ci95': p['ci95'],
                          'simultaneous_ci95': p['simultaneous_ci95'],
                          'positive_repeats': p['positive_repeats'],
                          'primary_reference': '509-question exact-ID-clean B4 is the registered primary; this estimate is a family-exclusion sensitivity'},
              'seconds': time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / (args.stage + '-predictions.npz')
    np.savez_compressed(pred_path, attempt_id=u['att']['attempt_id'].to_numpy(),
                        question=u['groups'], family=families, y=u['y'], valid=u['valid'],
                        **{'pred_' + k: v for k, v in result['preds'].items()})
    output['predictions_sha256'] = sha(pred_path)
    output['sha256'] = digest(output)
    target.write_text(json.dumps(output, indent=1) + '\n')
    print(json.dumps({'stage': args.stage, 'freeze': freeze['freeze_sha256'][:16],
                      'gain': p['gain'], 'ci95': p['ci95'], 'seconds': output['seconds']}), flush=True)


if __name__ == '__main__':
    main()
