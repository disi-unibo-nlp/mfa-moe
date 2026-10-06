"""Exact fixed-token native routing velocity and acceleration on discovery traces.

Each full 64-reasoning-token window is computed from the saved routed tensor,
not interpolated from variable-length sentence profiles. This is observational
and reads no class labels, correctness, or intervention outcome.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BASE = R / 'steering-v1/runs/routing-control-v1/dense-discovery'
UNITS = BASE / 'UNITS.json'
ROUTES = BASE / 'route-profiles-84a85c92-40c3d3b0'
ATTEMPTS = R / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
V3AN = R / 'v3_analysis'
WIDTH = 64


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed native route source seal: ' + str(path))
    return value


def window_profiles(ids, weights, reasoning, width=WIDTH, experts=256):
    ids, weights, reasoning = np.asarray(ids), np.asarray(weights), np.asarray(reasoning)
    if ids.ndim != 3 or ids.shape != weights.shape or ids.shape[2] != 8:
        raise ValueError('expected native [layer, token, top8] arrays')
    if reasoning.ndim != 1 or len(reasoning) < width or not np.all(np.diff(reasoning) > 0):
        raise ValueError('invalid ordered reasoning tokens')
    if reasoning[0] < 0 or reasoning[-1] >= ids.shape[1]:
        raise ValueError('reasoning positions exceed routed tensor')
    if ids.min() < 0 or ids.max() >= experts or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid native expert identity or gate weight')
    full = len(reasoning) // width
    layers = ids.shape[0]
    freq = np.empty((full, layers, experts), np.float32)
    gate = np.empty_like(freq)
    starts = np.empty(full, np.int32)
    ends = np.empty(full, np.int32)
    for index in range(full):
        positions = reasoning[index * width:(index + 1) * width]
        starts[index], ends[index] = positions[0], positions[-1] + 1
        for layer in range(layers):
            chosen = ids[layer, positions, :].astype(np.int64)
            mass = weights[layer, positions, :].astype(np.float64)
            denom = mass.sum(axis=1, keepdims=True)
            if (denom <= 0).any():
                raise ValueError('zero native selected-gate mass')
            freq[index, layer] = np.bincount(chosen.ravel(), minlength=experts) / (width * 8)
            gate[index, layer] = np.bincount(chosen.ravel(), weights=(mass / denom).ravel(),
                                             minlength=experts) / width
    if not np.allclose(freq.sum(axis=2), 1., atol=1e-6) or not np.allclose(gate.sum(axis=2), 1., atol=1e-6):
        raise ValueError('fixed-window native profile normalization differs')
    return freq, gate, starts, ends, len(reasoning) - full * width


def motion(freq, gate):
    """One row per window step; acceleration uses three complete windows."""
    if freq.shape != gate.shape or freq.ndim != 3:
        raise ValueError('frequency and gate window shapes differ')
    delta = np.diff(gate.astype(np.float64), axis=0)
    velocity = np.abs(delta).sum(axis=2) / 2
    acceleration = np.abs(np.diff(delta, axis=0)).sum(axis=2) / 2
    signed_speed_change = np.diff(velocity, axis=0)
    top = np.argsort(-freq, axis=2, kind='stable')[:, :, :8]
    turnover = np.empty((max(len(freq) - 1, 0), freq.shape[1]), np.float64)
    for step in range(len(turnover)):
        for layer in range(freq.shape[1]):
            turnover[step, layer] = 1 - len(set(top[step, layer]) & set(top[step + 1, layer])) / 8
    return velocity, acceleration, signed_speed_change, turnover


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native fixed-window routing extraction requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(V3AN))
    sys.path.insert(0, str(REPO / 'src'))
    from v3an.extract import _load_tensors
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import trace_digest
    from extract_discovery_routes import trace_at

    units, binding, source = sealed(UNITS), sealed(ROUTES / 'BINDING.json'), sealed(ROUTES / 'SUMMARY.json')
    if units['sha256'] != binding['units_sha256'] or source['binding_sha256'] != binding['sha256'] or source['families'] != 48:
        raise ValueError('native discovery route source is incomplete or rebound')
    out = BASE / f"fixed-window-routes-{units['sha256'][:8]}-{sha(__file__)[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    bind_body = {'schema': 'native-fixed-window-binding-v1',
                 'units_sha256': units['sha256'], 'source_route_summary_sha256': source['sha256'],
                 'driver_sha256': sha(__file__), 'attempts_sha256': sha(ATTEMPTS),
                 'width_reasoning_tokens': WIDTH, 'class_labels_used': False,
                 'definition': 'nonoverlapping full 64-reasoning-token windows; L1/2 of gate changes and second differences'}
    bind = {**bind_body, 'sha256': digest(bind_body)}
    if (out / 'BINDING.json').exists():
        if sealed(out / 'BINDING.json') != bind:
            raise ValueError('fixed-window output directory bound to changed inputs')
    else:
        (out / 'BINDING.json').write_text(json.dumps(bind, indent=1) + '\n')
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id','source_location','trace_sha256','tensor_path'])
    by_attempt = {str(r['attempt_id']): r for r in table.to_dict('records')}
    grouped = defaultdict(list)
    for unit in units['records']:
        grouped[unit['family']].append(unit)
    if len(grouped) != 48:
        raise ValueError('fixed discovery families differ')
    summaries = {}
    per_family = []
    for family in sorted(grouped):
        original = sealed(ROUTES / f'{family}.json')
        attempt = grouped[family][0]['attempt_id']
        if original['attempt_id'] != attempt or original['family'] != family or original['binding_sha256'] != binding['sha256']:
            raise ValueError('source native family receipt differs')
        old_array = ROUTES / f'{family}.npz'
        if sha(old_array) != original['npz_sha256']:
            raise ValueError('source native family array differs')
        target, receipt = out / f'{family}.npz', out / f'{family}.json'
        if receipt.exists():
            previous = sealed(receipt)
            if previous['binding_sha256'] != bind['sha256'] or previous['source_family_sha256'] != original['sha256'] or sha(target) != previous['npz_sha256']:
                raise ValueError('completed fixed-window family receipt differs')
            summaries[family] = previous['summary']
            continue
        if target.exists():
            raise ValueError('unreceipted fixed-window array needs manual recovery')
        source_row = by_attempt[attempt]
        if sha(source_row['tensor_path']) != original['tensor_sha256']:
            raise ValueError('native routed tensor changed after family extraction')
        trace = TraceRecord(**trace_at(source_row['source_location']))
        if trace_digest(trace) != source_row['trace_sha256']:
            raise ValueError('native trace differs from source attempt')
        layout = saved_layout(trace)
        ids, weights, _, reasoning = _load_tensors(source_row['tensor_path'])
        if (ids.shape[0] != 40 or ids.shape[1] != layout['token_count'] or
            len(reasoning) != len(layout['reasoning_tokens']) or
            not np.array_equal(reasoning, layout['reasoning_tokens'])):
            raise ValueError('native reasoning token positions differ')
        freq, gate, starts, ends, omitted = window_profiles(ids, weights, reasoning)
        velocity, acceleration, signed, turnover = motion(freq, gate)
        temp = target.with_name(target.name + '.part-' + os.environ['SLURM_JOB_ID'])
        with temp.open('wb') as stream:
            np.savez_compressed(stream, frequency=freq, gate=gate, token_start=starts,
                                token_end=ends, binding_sha256=bind['sha256'], attempt_id=attempt)
        os.replace(temp, target)
        summary = {'windows': len(freq), 'reasoning_tokens': len(reasoning),
                   'omitted_tail_tokens': omitted,
                   'mean_gate_velocity_by_layer': velocity.mean(axis=0).tolist() if len(velocity) else None,
                   'mean_gate_acceleration_by_layer': acceleration.mean(axis=0).tolist() if len(acceleration) else None,
                   'mean_signed_speed_change_by_layer': signed.mean(axis=0).tolist() if len(signed) else None,
                   'mean_top8_turnover_by_layer': turnover.mean(axis=0).tolist() if len(turnover) else None}
        body = {'schema': 'native-fixed-window-family-v1', 'binding_sha256': bind['sha256'],
                'family': family, 'attempt_id': attempt, 'source_family_sha256': original['sha256'],
                'tensor_sha256': original['tensor_sha256'], 'npz_sha256': sha(target), 'summary': summary}
        (receipt).write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
        summaries[family] = summary
        print(json.dumps({'complete_families': len(summaries), 'windows': len(freq)}), flush=True)
    metrics = ('mean_gate_velocity_by_layer','mean_gate_acceleration_by_layer',
               'mean_signed_speed_change_by_layer','mean_top8_turnover_by_layer')
    equal_family = {}
    for key in metrics:
        valid = [s[key] for s in summaries.values() if s[key] is not None]
        if len(valid) != len(summaries):
            raise ValueError('a family lacks enough complete windows for the registered motion profile')
        equal_family[key] = np.mean(valid, axis=0).tolist()
    body = {'schema': 'native-fixed-window-motion-v1', 'binding_sha256': bind['sha256'],
            'job_id': os.environ['SLURM_JOB_ID'], 'families': len(summaries),
            'total_full_windows': sum(s['windows'] for s in summaries.values()),
            'total_omitted_tail_tokens': sum(s['omitted_tail_tokens'] for s in summaries.values()),
            'equal_family_layer_profiles': equal_family, 'family_summaries': summaries,
            'interpretation': 'Observational native routes only; velocity and acceleration are gate-distribution descriptions, not controlled semantic outcomes'}
    summary = {**body, 'sha256': digest(body)}
    path = out / 'SUMMARY.json'
    if path.exists():
        if sealed(path) != summary:
            raise ValueError('existing native fixed-window summary differs')
    else:
        path.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'summary': str(path), 'families': len(summaries),
                      'full_windows': body['total_full_windows']}), flush=True)


if __name__ == '__main__':
    main()
