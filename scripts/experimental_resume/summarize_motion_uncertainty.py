"""Family-clustered descriptive intervals for the three native motion scalars."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/fixed-window-routes-84a85c92-363dbe1b')
INPUT = ROOT / 'SUMMARY.json'
OUTPUT = ROOT / 'MOTION_UNCERTAINTY.json'
METRICS = {
    'gate_velocity_TV_per_64_token_step': 'mean_gate_velocity_by_layer',
    'gate_acceleration_second_difference_L1_over_2': 'mean_gate_acceleration_by_layer',
    'top8_frequent_expert_turnover': 'mean_top8_turnover_by_layer',
}
REPLICATES = 5000
ALPHA = .05
SEED = 20261001


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed native motion seal')
    return value


def scalar_intervals(families, *, n_boot=REPLICATES, seed=SEED, alpha=ALPHA):
    if len(families) < 2 or len(set(families)) != len(families):
        raise ValueError('at least two unique families required')
    rows = []
    for name, field in METRICS.items():
        array = np.array([np.asarray(families[f][field], float) for f in sorted(families)])
        if array.shape != (len(families), 40) or not np.isfinite(array).all():
            raise ValueError('complete finite 40-layer profiles required')
        rows.append(array.mean(axis=1))
    values = np.stack(rows, axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), size=(n_boot, len(families)))
    boot = values[draws].mean(axis=1)
    tail = alpha / (2 * len(METRICS))
    return {name: {'estimate': float(values[:, j].mean()),
                   'simultaneous_interval': np.quantile(boot[:, j], [tail, 1-tail]).tolist(),
                   'family_mean_sd': float(values[:, j].std(ddof=1))}
            for j, name in enumerate(METRICS)}


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native motion uncertainty requires CPU Slurm')
    source = sealed(INPUT)
    if source['schema'] != 'native-fixed-window-motion-v1' or source['families'] != 48:
        raise ValueError('incomplete native discovery motion source')
    body = {'schema': 'native-fixed-window-motion-uncertainty-v1',
            'job_id': os.environ['SLURM_JOB_ID'], 'source_sha256': source['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'population': '48 deterministic, family-disjoint discovery native Qwen3.6 traces',
            'intervention': 'none; observational native routing',
            'estimates': scalar_intervals(source['family_summaries']),
            'multiplicity_family': 'three prespecified descriptive motion scalars',
            'interval_method': 'Bonferroni percentile family bootstrap, approximate',
            'replicates': REPLICATES, 'alpha': ALPHA, 'seed': SEED,
            'scope': 'Conditional heterogeneity across frozen discovery families; no causal or population-representative claim'}
    result = {**body, 'sha256': digest(body)}
    if OUTPUT.exists():
        if sealed(OUTPUT) != result:
            raise ValueError('native motion uncertainty output already differs')
    else:
        OUTPUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'estimates': body['estimates']}), flush=True)


if __name__ == '__main__':
    main()
