"""Verify and summarize 1,000 registered B4 conditional-gain simulations."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
OUT = S / 'runs/resume-v1/r3e-b4-precision-v3'
PRICE = REPO / 'report/experimental-resume-v1/R3E_B4_PRECISION_PRICE_v3.json'
AGG_PRICE = REPO / 'report/experimental-resume-v1/R3E_B4_PRECISION_AGGREGATE_PRICE_v2.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z/den * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [mid-half, mid+half]


def mean_mc(values):
    vals = [float(v) for v in values]
    avg = statistics.mean(vals)
    sd = statistics.stdev(vals)
    half = 1.959963984540054*sd/math.sqrt(len(vals))
    return {'mean': avg, 'sd': sd, 'mc_ci95': [avg-half, avg+half]}


def summarize(rows, target):
    n = len(rows)
    if n != 200:
        raise ValueError(f'target {target} has {n}, expected 200')
    gains = [r['gain'] for r in rows]
    algorithm_mean = statistics.mean(gains)
    # Leave-one-replicate-out empirical algorithm target avoids comparing an
    # interval to a mean partly defined by its own outcome.
    loo = [(algorithm_mean*n - g)/(n-1) for g in gains]
    algo_covered = sum(r['ci95'][0] <= m <= r['ci95'][1]
                       for r, m in zip(rows, loo))
    oracle_covered = sum(r['ci95'][0] <= r['realized_oracle_kl'] <= r['ci95'][1]
                         for r in rows)
    positive = sum(r['holm_safe_reject'] for r in rows)
    advance = sum(r['registered_advance'] for r in rows)
    return {'target_train_oracle_kl': target, 'replicates': n,
            'heldout_realized_oracle_kl': mean_mc([r['realized_oracle_kl'] for r in rows]),
            'fitted_gain': mean_mc(gains),
            'same_dimension_noise_gain': mean_mc([r['noise_gain'] for r in rows]),
            'holm_safe_positive_rejection': {'count': positive, 'rate': positive/n,
                                              'wilson_ci95': wilson(positive,n)},
            'registered_advance': {'count': advance, 'rate': advance/n,
                                   'wilson_ci95': wilson(advance,n)},
            'ci95_coverage_empirical_fitted_algorithm_mean': {'count': algo_covered,
                                                               'rate': algo_covered/n,
                                                               'wilson_ci95': wilson(algo_covered,n)},
            'ci95_coverage_heldout_oracle_KL_diagnostic_only': {'count': oracle_covered,
                                                                 'rate': oracle_covered/n,
                                                                 'wilson_ci95': wilson(oracle_covered,n)}}


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('precision aggregation requires owner CPU Slurm')
    price = sealed(PRICE)
    agg_price = sealed(AGG_PRICE)
    if agg_price['driver_sha256'] != sha(__file__) or agg_price['parent_price_sha256'] != price['sha256']:
        raise ValueError('precision aggregation source/parent price differs')
    paths = [OUT / 'smoke.json'] + [OUT / f'shard-{i:02}.json' for i in range(20)]
    receipts = [sealed(p) for p in paths]
    all_rows = []
    gen = receipts[0]['generator']
    for j, receipt in enumerate(receipts):
        if (receipt['status'] != 'PASS' or receipt['schema'] != 'r3e-b4-calibrated-precision-v3' or
            receipt['price_sha256'] != price['sha256'] or receipt['generator'] != gen or
            receipt['feature_freeze_sha256'] != price['freeze_sha256'] or
            receipt['stage'] != ('smoke' if j == 0 else 'shard') or
            receipt['shard'] != (None if j == 0 else j-1)):
            raise ValueError(f'precision receipt {paths[j]} differs')
        all_rows.extend(receipt['simulation']['rows'])
    ids = [r['sim_id'] for r in all_rows]
    if sorted(ids) != list(range(1000)) or len(ids) != 1000:
        raise ValueError('duplicate or missing precision replicate IDs')
    targets = tuple(price['design']['targets_oracle_KL_nats_per_question'])
    for row in all_rows:
        if row['target_kl'] != targets[row['sim_id']//200]:
            raise ValueError(f'replicate target differs: {row["sim_id"]}')
    summary = {str(t): summarize([r for r in all_rows if r['target_kl'] == t],t)
               for t in targets}
    body = {'schema': 'r3e-b4-calibrated-precision-summary-v3', 'status': 'COMPLETE',
            'aggregation_job_id': os.environ['SLURM_JOB_ID'],
            'price_sha256': price['sha256'],
            'aggregation_price_sha256': agg_price['sha256'],
            'source_receipt_sha256': {p.name: r['sha256'] for p,r in zip(paths,receipts)},
            'population': receipts[0]['population'],
            'generator': gen, 'summary': summary,
            'interpretation': {'target': 'finite-sample power/precision of the frozen B4 estimator under this cross-fitted conditional-label generator',
                               'algorithm_coverage': '95% family-bootstrap intervals compared with the leave-one-replicate-out mean fitted gain at each target; empirical simulation target, not exact coverage theorem',
                               'oracle_coverage': 'diagnostic comparison to Bayes KL, which is a distinct estimand from fitted predictive gain',
                               'families': 'simulated question labels independent conditional on fixed features; no shared family random effect; original duplicate-family CV and family bootstrap preserved',
                               'calibration': 'held-out original outcomes excluded; held-out covariates used solely to set exact scored-population oracle KL target. The v2 train-only calibration is a separate sensitivity and overshot held-out targets.',
                               'equivalence': 'non-rejection is not equivalence or accuracy retention',
                               'causality': 'observational prediction calibration, not expert-routing steering'},
            'prior_versions': {'v1_failed_smoke_job': '59201923',
                               'v2_train_calibration_summary': str(S / 'runs/resume-v1/r3e-b4-precision-v2/summary.json')},
            'incomplete_v3_replicates': 0}
    body['sha256'] = digest(body)
    target = OUT / 'summary.json'
    if target.exists():
        if json.loads(target.read_text()) != body:
            raise ValueError('existing precision summary differs')
    else:
        tmp = OUT / ('summary.json.part-' + os.environ['SLURM_JOB_ID'])
        tmp.write_text(json.dumps(body, indent=1) + '\n')
        tmp.replace(target)
    print(json.dumps({'summary': str(target), 'replicates': len(all_rows),
                      'targets': list(summary)}), flush=True)


if __name__ == '__main__':
    main()
