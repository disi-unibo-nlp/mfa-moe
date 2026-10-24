"""Acceptance of a finished capture: every shard of a cohort has complete, intact and consistent artifacts.

For the manifest folder of one model and cohort (`ready.json` plus one `SHARD.json` per shard) it checks that
  - the shard list is exactly the one `ready.json` froze, and each manifest carries the same model, spec and bindings;
  - the verification receipt of the model passed (`validate_verification`);
  - every trace of every shard has its two files (sizes and hashes match the marker), no trace is missing or twice;
  - the shard `result.json` agrees with the markers, and the token and trace totals equal the planned ones.

    python -m capture.gate MANIFEST_DIR OUTPUT.json [--captures ROOT]     (a CPU job: it hashes every captured file)

The captures are where the manifests say (`capture_dir`). With `--captures ROOT` they are in a repeat tree instead:
`ROOT/<shard number>` for the shards and `ROOT/verify` for the verification.

A second run on unchanged captures must give the same acceptance (the `code` field aside).
"""
import argparse
import json
import os
from pathlib import Path

from capture.manifest import atomic, capture_folders, load_manifest, resume_records, validate_verification
from provenance import code_provenance
from store import file_sha

SHARED_KEYS = ('source', 'cohort', 'spec', 'validation_binding', 'code_binding', 'cohort_binding', 'cohort_model',
               'cohort_manifest_path', 'maximum_input_tokens', 'storage_root')


def gate(manifest_dir, captures=None):
    root = Path(manifest_dir)
    ready = json.loads((root / 'ready.json').read_text())
    if ready.get('status') != 'ready' or not ready.get('shards'):
        raise ValueError('Production capture planning is incomplete')
    expected = {str(s['index']) for s in ready['shards']}
    if len(expected) != len(ready['shards']):
        raise ValueError('Duplicate ready shard indices')
    actual = {p.stem for p in root.glob('*.json')} - {'ready', 'verify'}
    if actual != expected:
        raise ValueError('Unexpected/missing capture manifests')
    manifests = [load_manifest(root / (i + '.json')) for i in sorted(expected)]
    for manifest in manifests:
        for key in SHARED_KEYS:
            if manifest.get(key) != ready.get(key):
                raise ValueError('Ready/shard provenance differs: ' + key)
        planned = next(s for s in ready['shards'] if s['index'] == manifest['shard'])
        if planned['binding'] != manifest['binding']:
            raise ValueError('Ready shard binding differs')
    ids = []
    total_tokens = 0
    inputs = {}
    for manifest in manifests:
        out, receipt = capture_folders(manifest, captures)
        validate_verification(manifest, json.loads(receipt.read_text()))
        records, missing, used = resume_records(manifest, out)
        result_path = out / 'result.json'
        result = json.loads(result_path.read_text())
        if missing or len(records) != manifest['expected_attempts'] or result.get('status') != 'complete_native_collection':
            raise ValueError('Incomplete capture shard')
        tokens = sum(r['tokens'] for r in records)
        if (result.get('manifest_binding') != manifest['binding'] or result.get('records') != records
                or tokens != manifest['expected_tokens'] or result.get('tokens') != tokens
                or result.get('artifact_bytes') != used):
            raise ValueError('Stale/inconsistent shard result')
        ids.extend(r['id'] for r in records)
        total_tokens += tokens
        inputs[str(result_path)] = file_sha(result_path)
    if len(ids) != len(set(ids)) or len(ids) != ready['attempts'] or total_tokens != ready['tokens']:
        raise ValueError('Campaign cohort coverage mismatch')
    return dict(status='complete', attempts=len(ids), tokens=total_tokens,
                ready_sha256=file_sha(root / 'ready.json'), results=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--captures', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Production capture gate requires a Slurm CPU allocation')
    result = gate(args.manifest_dir, args.captures)
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if {k: v for k, v in saved.items() if k != 'code'} != result:
            raise ValueError('Existing acceptance differs; use fresh output')
    atomic(args.output, {**result, 'code': code_provenance()})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
