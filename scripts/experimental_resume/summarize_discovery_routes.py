"""Family-balanced native expert exposure and adjacent-sentence route displacement."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
ROOT = R / 'steering-v1/runs/routing-control-v1/dense-discovery'
INPUT = ROOT / 'route-profiles-84a85c92-40c3d3b0'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed route receipt: ' + str(path))
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def summarize():
    binding = sealed(INPUT / 'BINDING.json')
    complete = sealed(INPUT / 'SUMMARY.json')
    if complete['binding_sha256'] != binding['sha256'] or complete['families'] != 48 or complete['sentences'] != 5659:
        raise ValueError('native discovery routes incomplete')
    expected_families = set(complete['family_counts'])
    families, changes = [], []
    for family in sorted(expected_families):
        receipt = sealed(INPUT / f'{family}.json')
        path = INPUT / f'{family}.npz'
        if receipt['binding_sha256'] != binding['sha256'] or receipt['family'] != family or sha(path) != receipt['npz_sha256']:
            raise ValueError('family profile binding differs')
        with np.load(path, allow_pickle=False) as z:
            freq = z['frequency'].astype(np.float64)
            gate = z['gate'].astype(np.float64)
            count = z['n_tokens'].astype(np.int64)
            index = z['sentence_index'].astype(np.int64)
            start = z['token_start'].astype(np.int64)
            end = z['token_end'].astype(np.int64)
            if str(z['binding_sha256']) != binding['sha256'] or str(z['attempt_id']) != receipt['attempt_id']:
                raise ValueError('NPZ identity differs')
        if freq.shape != gate.shape or freq.shape != (receipt['sentences'], 40, 256) or len(count) != len(freq):
            raise ValueError('profile array shape differs')
        if not np.isfinite(freq).all() or not np.isfinite(gate).all() or (freq < 0).any() or (gate < 0).any():
            raise ValueError('nonfinite or negative route profile')
        if not np.allclose(freq.sum(axis=2), 1., atol=1e-6) or not np.allclose(gate.sum(axis=2), 1., atol=1e-6):
            raise ValueError('native route normalization differs')
        if (count <= 0).any() or (start >= end).any() or not np.all(index[1:] > index[:-1]):
            raise ValueError('native sentence order or ownership differs')
        # Top-k=8 means frequencies sum to one; multiply by eight for the
        # probability that a token selects each expert in native routing.
        exposure = np.average(freq * 8., axis=0, weights=count)
        if (exposure > 1. + 1e-5).any():
            raise ValueError('native expert exposure exceeds one')
        families.append(exposure)
        adjacent = (index[1:] == index[:-1] + 1) & (start[1:] >= end[:-1])
        for metric, profile in (('selection', freq), ('gate', gate)):
            tv = np.abs(np.diff(profile, axis=0)).sum(axis=2).mean(axis=1) / 2
            changes.append({'family': family, 'metric': metric, 'n_adjacent': int(adjacent.sum()),
                            'mean_TV': float(tv[adjacent].mean()) if adjacent.any() else None})
    rates = np.mean(np.stack(families), axis=0)
    result = {'schema': 'dense-discovery-native-route-summary-v1',
              'binding_sha256': binding['sha256'], 'route_summary_sha256': complete['sha256'],
              'families': len(families), 'sentences': complete['sentences'],
              'native_expert_selection_exposure': rates.tolist(),
              'adjacent_sentence_displacement': changes,
              'exposure_weighting': 'token-weighted within family, then equal family weights; each expert selected at most once per token',
              'displacement_scope': 'adjacent observed sentence profiles only; fixed-token velocity and acceleration require separate arrays',
              'status': 'OBSERVATIONAL_ONLY_NO_SEMANTIC_TARGET_SELECTION'}
    result['sha256'] = digest(result)
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('route summary requires CPU Slurm')
    result = summarize()
    result['job_id'] = os.environ['SLURM_JOB_ID']
    result['sha256'] = digest({k: v for k, v in result.items() if k != 'sha256'})
    path = INPUT / 'NATIVE_ROUTE_SUMMARY.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'path': str(path), 'families': result['families'],
                      'sentences': result['sentences']}))


if __name__ == '__main__':
    main()
