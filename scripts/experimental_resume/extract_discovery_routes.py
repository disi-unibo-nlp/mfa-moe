"""Extract exact native expert-use profiles for fixed discovery sentence windows.

No correctness, gold answers, judge labels, or future semantic outcomes are read.
The saved trace's exact token ownership is matched to the frozen window manifest.
One atomic NPZ and receipt per family make the CPU stage resumable.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
UNITS = R / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'
ATTEMPTS = R / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
V3AN = R / 'v3_analysis'
EXPECTED_UNITS = '84a85c92b595a7829446b6a53b376e1126df3b6c36eabaddb3b4e884eb52fa72'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed JSON seal: {path}')
    return value


def trace_at(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location['path']).open('rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('native trace location changed')
    return json.loads(raw)


def profiles(ids, weights, owners, *, experts=256):
    """One sentence's native selection frequency and selected-gate distribution."""
    import numpy as np
    L, T, k = ids.shape
    tokens = np.asarray(owners, dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens) or (tokens < 0).any() or (tokens >= T).any() or len(set(tokens.tolist())) != len(tokens):
        raise ValueError('invalid owned token indices')
    chosen = ids[:, tokens, :].astype(np.int64)
    mass = weights[:, tokens, :].astype(np.float64)
    if chosen.min() < 0 or chosen.max() >= experts or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError('invalid saved native router output')
    denom = mass.sum(axis=2, keepdims=True)
    if (denom <= 0).any():
        raise ValueError('zero selected gate mass')
    mass /= denom
    freq = np.zeros((L, experts), dtype=np.float32)
    gate = np.zeros((L, experts), dtype=np.float32)
    for layer in range(L):
        flat = chosen[layer].ravel()
        freq[layer] = np.bincount(flat, minlength=experts) / (len(tokens) * k)
        gate[layer] = np.bincount(flat, weights=mass[layer].ravel(), minlength=experts) / len(tokens)
    if not np.allclose(freq.sum(axis=1), 1., atol=1e-6) or not np.allclose(gate.sum(axis=1), 1., atol=1e-6):
        raise ValueError('profile normalization differs from native top-k')
    return freq, gate


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('route extraction requires CPU Slurm')
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(V3AN))
    sys.path.insert(0, str(REPO / 'src'))
    from v3an.extract import _load_tensors
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import trace_digest

    units = sealed(UNITS)
    if units['sha256'] != EXPECTED_UNITS or units['attempts'] != 48:
        raise ValueError('discovery sentence freeze differs')
    driver_sha = sha(__file__)
    out = UNITS.parent / f"route-profiles-{units['sha256'][:8]}-{driver_sha[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    binding_body = {'schema': 'dense-discovery-native-route-binding-v1',
                    'units_sha256': units['sha256'], 'driver_sha256': driver_sha,
                    'extractor_sha256': sha(V3AN / 'v3an/extract.py'),
                    'layout_sha256': sha(Path(sys.modules[saved_layout.__module__].__file__)),
                    'attempts_sha256': sha(ATTEMPTS), 'class_labels_used': False,
                    'population': '48 fixed Qwen3.6 discovery native traces only'}
    binding = {**binding_body, 'sha256': digest(binding_body)}
    if (out / 'BINDING.json').exists():
        if sealed(out / 'BINDING.json') != binding:
            raise ValueError('route output directory bound to different code or inputs')
    else:
        atomic_json(out / 'BINDING.json', binding)
    grouped = defaultdict(list)
    for row in units['records']:
        grouped[row['attempt_id']].append(row)
    if len(grouped) != 48:
        raise ValueError('fixed 48 discovery attempts missing')
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id','question','trace_sha256',
                                                 'completion_tokens','source_location','tensor_path'])
    table = table[table['attempt_id'].astype(str).isin(grouped)]
    if len(table) != 48 or len(set(table['attempt_id'])) != 48:
        raise ValueError('native route attempt table is incomplete or duplicated')
    counts = {}
    for attempt in sorted(grouped):
        record = table.loc[table['attempt_id'].astype(str) == attempt].iloc[0]
        rows = sorted(grouped[attempt], key=lambda row: row['sentence_index'])
        family = rows[0]['family']
        if any(r['family'] != family or r['question'] != record['question'] or
               r['trace_sha256'] != record['trace_sha256'] for r in rows):
            raise ValueError('route attempt identity differs from discovery sentences')
        npz = out / f'{family}.npz'
        receipt = out / f'{family}.json'
        if receipt.exists():
            previous = sealed(receipt)
            if previous['binding_sha256'] != binding['sha256'] or previous['attempt_id'] != attempt or previous['family'] != family or not npz.is_file() or sha(npz) != previous['npz_sha256']:
                raise ValueError('completed route profile receipt changed')
            counts[family] = previous['sentences']
            continue
        if npz.exists():
            raise ValueError('unreceipted route array requires manual recovery: ' + str(npz))
        trace = TraceRecord(**trace_at(record['source_location']))
        if trace_digest(trace) != record['trace_sha256']:
            raise ValueError('native trace digest differs')
        layout = saved_layout(trace)
        ids, weights, _, reasoning = _load_tensors(record['tensor_path'])
        L, T, k = ids.shape
        if T != int(record['completion_tokens']) or layout['token_count'] != T or k != 8 or L != 40:
            raise ValueError('native route tensor shape/token alignment differs')
        if list(layout['reasoning_tokens']) != reasoning.tolist():
            raise ValueError('native reasoning positions differ from tensor capture')
        reasoning_mask = np.zeros(T, bool)
        reasoning_mask[reasoning] = True
        freq = np.zeros((len(rows), L, 256), np.float32)
        gate = np.zeros_like(freq)
        n_tokens = np.zeros(len(rows), np.int32)
        for j, row in enumerate(rows):
            index = int(row['sentence_index'])
            unit = layout['units'][index]
            owned = layout['unit_tokens'][index]
            if (unit['text'] != row['inputs']['sentence'] or
                unit['start'] != row['char_start'] or unit['end'] != row['char_end'] or
                min(owned) != row['token_start'] or max(owned) + 1 != row['token_end']):
                raise ValueError('sentence text/offset/ownership differs')
            if not reasoning_mask[np.asarray(owned, dtype=np.int64)].all():
                raise ValueError('selected sentence includes a nonreasoning token')
            freq[j], gate[j] = profiles(ids, weights, owned)
            n_tokens[j] = len(owned)
        tmp = npz.with_name(npz.name + '.part-' + os.environ['SLURM_JOB_ID'])
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, frequency=freq, gate=gate, sentence_index=np.array([r['sentence_index'] for r in rows]),
                                token_start=np.array([r['token_start'] for r in rows]), token_end=np.array([r['token_end'] for r in rows]),
                                n_tokens=n_tokens, binding_sha256=binding['sha256'], attempt_id=attempt)
        os.replace(tmp, npz)
        body = {'schema': 'dense-discovery-native-route-family-v1', 'binding_sha256': binding['sha256'],
                'family': family, 'attempt_id': attempt, 'question': record['question'],
                'trace_sha256': record['trace_sha256'], 'tensor_sha256': sha(record['tensor_path']),
                'npz_sha256': sha(npz), 'sentences': len(rows), 'routed_layers': L,
                'experts': 256, 'top_k': k, 'owned_tokens': int(n_tokens.sum())}
        atomic_json(receipt, {**body, 'sha256': digest(body)})
        counts[family] = len(rows)
        print(json.dumps({'families_complete': len(counts), 'sentences': sum(counts.values())}), flush=True)
    if sum(counts.values()) != len(units['records']):
        raise ValueError('incomplete discovery route profiles')
    summary_body = {'schema': 'dense-discovery-native-routes-v1', 'binding_sha256': binding['sha256'],
                    'families': len(counts), 'sentences': sum(counts.values()),
                    'family_counts': counts, 'job_id': os.environ['SLURM_JOB_ID'],
                    'interpretation': 'Native observational expert use only; proposals require semantic eligibility and causal validation.'}
    summary = {**summary_body, 'sha256': digest(summary_body)}
    if (out / 'SUMMARY.json').exists():
        if sealed(out / 'SUMMARY.json') != summary:
            raise ValueError('route summary differs')
    else:
        atomic_json(out / 'SUMMARY.json', summary)
    print(json.dumps({'path': str(out / 'SUMMARY.json'), 'families': len(counts),
                      'sentences': sum(counts.values())}), flush=True)


if __name__ == '__main__':
    main()
