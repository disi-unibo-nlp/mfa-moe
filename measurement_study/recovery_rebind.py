"""Prepare new recovery manifests; retain valid captures with explicit parent receipts."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from measurement_study.production_plan import ROOT, atomic, digest, sha


def rebind_manifest(parent, spec, code_binding, cap, parent_path):
    if digest({k: v for k, v in parent.items() if k != 'binding'}) != parent['binding']:
        raise ValueError('Parent manifest digest changed')
    if parent['spec'] != spec:
        raise ValueError('Recovery cannot silently change checkpoint or replay semantics')
    if len(parent['attempts']) != parent['expected_attempts'] or sum(r['completion_tokens'] for r in parent['attempts']) != parent['expected_tokens']:
        raise ValueError('Parent manifest coverage changed')
    result = copy.deepcopy(parent)
    result.update(code_binding=code_binding, artifact_cap_bytes=cap, transient_cap_bytes=7*1024**3,
                  parent_manifest=dict(path=str(parent_path), sha256=sha(parent_path), binding=parent['binding']))
    result['binding'] = digest({k: v for k, v in result.items() if k != 'binding'})
    return result


def reuse_packet(record, old_binding, manifest, output, acceptance_path):
    if record.get('manifest_binding', old_binding) != old_binding:
        raise ValueError('Saved completion binding changed')
    rows = {r['id']: r for r in manifest['attempts']}
    if record['id'] not in rows:
        raise ValueError('Saved completion outside manifest')
    row = rows[record['id']]
    if record.get('tokens', record.get('completion_tokens')) != row['completion_tokens']:
        raise ValueError('Saved completion token count changed')
    tensor, packet_info = record['files']
    for info in record['files']:
        path = Path(info['path'])
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Saved artifact changed')
    raw = gzip.decompress(Path(packet_info['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != packet_info['raw_sha256']:
        raise ValueError('Saved packet raw hash changed')
    packet = json.loads(raw)
    if packet['id'] != record['id'] or packet['manifest_binding'] != old_binding:
        raise ValueError('Saved packet provenance changed')
    packet['parent_capture'] = dict(manifest_binding=old_binding, files=record['files'], acceptance_path=str(acceptance_path), acceptance_sha256=sha(acceptance_path))
    packet['manifest_binding'] = manifest['binding']
    packet['tensor_artifact'] = tensor
    raw = json.dumps(packet, allow_nan=False).encode()
    path = output / (record['id'] + '.json.gz')
    if path.exists():
        raise ValueError('Refuse to replace an existing recovery artifact')
    path.write_bytes(gzip.compress(raw, mtime=0))
    info = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), raw_sha256=hashlib.sha256(raw).hexdigest())
    done = dict(id=record['id'], manifest_binding=manifest['binding'], files=[tensor, info],
                tokens=row['completion_tokens'], seconds=record['seconds'],
                job=record.get('job', '58518118'), reused=True, parent_capture=packet['parent_capture'])
    atomic(output / (record['id'] + '.complete.json'), done)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', choices=['gemma', 'nemotron', 'qwen330b'])
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--cap-gib', type=int, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm CPU allocation required')
    if a.input_root.resolve() == a.output_root.resolve():
        raise ValueError('Recovery must preserve parent manifests')
    for rel, expected in json.loads((a.bundle/'SHA256.json').read_text()).items():
        if sha(a.bundle/rel) != expected:
            raise ValueError('Frozen recovery code changed')
    campaign = json.loads((a.bundle/'campaign.json').read_text())
    for path, expected in campaign['runtime_code_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Shared runtime changed')
    spec = campaign['sources'][a.source]
    if a.source == 'gemma':
        acceptance_path = a.input_root/'capture/gemma/3/acceptance.json'
    elif a.source == 'nemotron':
        acceptance_path = ROOT/'results/measurement-study/recovery-storage/validation/nemotron-confirmation-v1/acceptance.json'
    else:
        acceptance_path = ROOT/'results/measurement-study/long-storage-sizing/result.json'
    acceptance = json.loads(acceptance_path.read_text())
    if a.source != 'qwen330b' and acceptance['status'] != 'passed':
        raise ValueError('Native GPU acceptance required')
    frozen_parents = campaign['recovery_parents'][a.source]
    for path, expected in frozen_parents['hashes'].items():
        if sha(path) != expected:
            raise ValueError('Validated parent code or manifest changed')
    if a.source == 'nemotron':
        fixtures = json.loads(Path(frozen_parents['fixtures']).read_text())
    total_reused = []
    manifests = []
    for i in range(4):
        path = a.input_root/'manifests'/a.source/f'{i}.json'
        parent = json.loads(path.read_text())
        manifest = rebind_manifest(parent, spec, sha(a.bundle/'SHA256.json'), a.cap_gib*1024**3//4, path)
        manifest_path = a.output_root/'manifests'/a.source/f'{i}.json'
        if manifest_path.exists():
            raise ValueError('Recovery manifest already exists')
        output = a.output_root/'capture'/a.source/str(i)
        output.mkdir(parents=True, exist_ok=True)
        atomic(manifest_path, manifest)
        manifests.append(manifest)
        reused = []
        if a.source == 'gemma':
            for marker in sorted((a.input_root/'capture/gemma'/str(i)).glob('*.complete.json')):
                record = json.loads(marker.read_text())
                reused.append(reuse_packet(record, parent['binding'], manifest, output, acceptance_path))
        elif a.source == 'nemotron':
            fixture = next(f for f in fixtures if f['shard'] == i)
            row = next(r for r in manifest['attempts'] if r['id'] == fixture['row']['id'])
            if digest(row) != digest(fixture['row']):
                raise ValueError('Validated Nemotron replay input changed')
            historical = copy.deepcopy(fixture['manifest'])
            historical['code_binding'] = acceptance['capture_bundle_sha256']
            historical['binding'] = digest({k: v for k, v in historical.items() if k != 'binding'})
            record = next(r for r in acceptance['records'] if r['shard'] == i)
            reused.append(reuse_packet(record, historical['binding'], manifest, output, acceptance_path))
        total_reused.extend(reused)
        atomic(output/'progress.json', dict(completed=len(reused), expected=manifest['expected_attempts'],
            tokens=sum(r['tokens'] for r in reused), artifact_bytes=sum(f['bytes'] for r in reused for f in r['files'])))
    ready = dict(status='ready', source=a.source, attempts=sum(m['expected_attempts'] for m in manifests),
        tokens=sum(m['expected_tokens'] for m in manifests), code_binding=sha(a.bundle/'SHA256.json'),
        replay_variant=spec.get('replay_variant',''), reused_capture_ids=[r['id'] for r in total_reused],
        acceptance_path=str(acceptance_path), acceptance_sha256=sha(acceptance_path),
        manifests=[dict(path=str(a.output_root/'manifests'/a.source/f'{i}.json'),binding=m['binding']) for i,m in enumerate(manifests)],
        verification_shard=max(range(4),key=lambda i:max(r['total_input_tokens'] for r in manifests[i]['attempts'])))
    atomic(a.output_root/'manifests'/a.source/'ready.json',ready)
    print(json.dumps(ready),flush=True)


if __name__ == '__main__':
    main()
