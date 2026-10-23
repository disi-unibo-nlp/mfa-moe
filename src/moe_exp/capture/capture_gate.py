"""Read-only gate: every planned v3 shard must have accepted exact artifacts."""
import argparse
import json
import os
from pathlib import Path
from moe_exp.capture.production_plan import sha, atomic
from moe_exp.capture.campaign_v3_capture import load_manifest, resume_records, validate_verification
from moe_exp.capture.bundle import worker_init, verify_loaded_imports


def gate(manifest_dir, bundle):
    root = Path(manifest_dir)
    ready = json.loads((root / 'ready.json').read_text())
    if ready.get('status')!='ready' or not ready.get('shards'):
        raise ValueError('Production capture planning is incomplete')
    expected = {str(s['index']) for s in ready['shards']}
    if len(expected)!=len(ready['shards']):raise ValueError('Duplicate ready shard indices')
    actual = {p.stem for p in root.glob('*.json')} - {'ready', 'verify'}
    if actual != expected:
        raise ValueError('Unexpected/missing capture manifests')
    manifests = [load_manifest(root / (i+'.json'), bundle) for i in sorted(expected)]
    for manifest in manifests:
        for key in ('source','cohort','spec','validation_binding','code_binding','cohort_binding','cohort_model',
                    'cohort_manifest_path','maximum_input_tokens','storage_root'):
            if manifest.get(key)!=ready.get(key):raise ValueError('Ready/shard provenance differs: '+key)
        expected_shard=next(s for s in ready['shards'] if s['index']==manifest['shard'])
        if expected_shard['binding']!=manifest['binding']:raise ValueError('Ready shard binding differs')
    ids = []
    total_tokens = 0
    inputs = {}
    for manifest in manifests:
        validate_verification(manifest, json.loads(Path(manifest['verification_path']).read_text()))
        out = Path(manifest['capture_dir'])
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
        ids.extend(r['id'] for r in records); total_tokens += tokens
        inputs[str(result_path)] = sha(result_path)
    if len(ids) != len(set(ids)) or len(ids) != ready['attempts'] or total_tokens != ready['tokens']:
        raise ValueError('Campaign cohort coverage mismatch')
    return dict(status='complete', attempts=len(ids), tokens=total_tokens,
                ready_sha256=sha(root/'ready.json'), results=inputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Production capture gate requires a Slurm CPU allocation')
    worker_init(a.bundle)
    result = gate(a.manifest_dir, a.bundle)
    result['imports'] = verify_loaded_imports(a.bundle)
    if a.output.exists() and json.loads(a.output.read_text()) != result:
        raise ValueError('Existing acceptance differs; use fresh output')
    atomic(a.output, result)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
