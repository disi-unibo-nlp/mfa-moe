"""Seal Qwen3.8 reader output lengths without copying completion text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BATCHES = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/micro-screen-semantic-v22-1e56570b050828e3/batches')
OUTPUT = REPO / 'report/experimental-resume-v1/CAUSAL_READER_OUTPUT_LENGTH_AUDIT_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = sorted(BATCHES.glob('*.json'))
    if len(files) != 10:
        raise ValueError(f'expected ten complete reader batches, found {len(files)}')
    rows = []
    provenance = {}
    for path in files:
        raw = path.read_bytes()
        batch = json.loads(raw)
        if batch.get('sha256') != digest({k: v for k, v in batch.items() if k != 'sha256'}):
            raise ValueError(f'changed saved reader batch: {path}')
        provenance[str(path)] = hashlib.sha256(raw).hexdigest()
        rows.extend(batch['records'])
    valid = [r for r in rows if isinstance(r['rating'], dict) and r['finish_reason'] == 'stop']
    counts = {
        'all_assigned_ratings': len(rows),
        'natural_stops': sum(r['finish_reason'] == 'stop' for r in rows),
        'length_stops_at_1024': sum(r['finish_reason'] == 'length' and r['generated_tokens'] == 1024 for r in rows),
        'valid_stopped_ratings': len(valid),
        'valid_stopped_over_256_tokens': sum(r['generated_tokens'] > 256 for r in valid),
        'all_ratings_over_256_tokens': sum(r['generated_tokens'] > 256 for r in rows),
    }
    if counts != {'all_assigned_ratings': 104, 'natural_stops': 101,
                  'length_stops_at_1024': 3, 'valid_stopped_ratings': 97,
                  'valid_stopped_over_256_tokens': 44,
                  'all_ratings_over_256_tokens': 49}:
        raise ValueError(f'prior reader tail counts changed: {counts}')
    body = {
        'schema': 'causal-reader-output-length-audit-v1',
        'status': 'COMPLETE_METADATA_ONLY',
        'reader_model': 'Qwen3.8-27B',
        'population': 'prior micro-screen independent arm-blind reader batches; not new eligible deactivation ratings',
        'counts': counts,
        'batch_file_sha256': provenance,
        'interpretation': 'A 256-output-token price underestimates the observed reader tail. Three prior ratings exhausted the 1,024-token cap. These counts are pricing/measurement evidence, not a causal semantic effect; no raw completion text is copied.'}
    OUTPUT.write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    print(json.dumps({'path': str(OUTPUT), 'sha256': digest(body), 'counts': counts}))


if __name__ == '__main__':
    main()
