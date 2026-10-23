"""Frozen capture manifests: loading, checking and the accepted artifacts of a shard.

A manifest lists the traces of one shard of one model (with the byte position of each saved trace line and its
sentence selection), the model spec (checkpoint, routed layers, experts, top-k), the cohort file it was cut from and
the label files it uses. `binding` is the hash of all other fields. The manifests of the paper were written by a
planner that is not part of this repository and are the inputs of a capture run:

    /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/{A,B}/SHARD.json   one per shard
    /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/{A,B}/ready.json   the shard list
    /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/MODEL/verify.json        the 2-trace check

Every file a manifest names (cohort file, label files, source trace lines) is checked against the hash it was
frozen with before it is used.
"""
import hashlib
import json
from pathlib import Path

from labeling.spans import digest  # the JSON form the manifest bindings were computed with
from store import file_sha

DIFFERENTIAL_MODELS = ('gpt', 'glm')  # verified against a second capture of the short trace


def atomic(path, value):
    """Write a JSON file so that a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False) + '\n')
    tmp.replace(path)


def write_new(path, value):
    """Write a receipt that must not exist yet: a finished capture or verification is never overwritten."""
    if path.exists():
        raise ValueError('Refusing to overwrite an existing receipt: ' + str(path))
    atomic(path, value)


def read_location(info):
    """One saved trace line, read from its byte position and checked against the frozen size, time and hash."""
    path = Path(info['path'])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (info['size'], info['mtime_ns']):
        raise ValueError('Source changed: ' + str(path))
    with path.open('rb') as stream:
        stream.seek(info['byte_offset'])
        raw = stream.read(info['line_bytes'])
    if hashlib.sha256(raw).hexdigest() != info['line_sha256']:
        raise ValueError('Source line checksum changed')
    trace = json.loads(raw)
    if 'sentence_schema' in info:
        trace.setdefault('metadata', {})['measurement_sentence_schema'] = info['sentence_schema']
    if 'sentence_selection' in info:
        trace.setdefault('metadata', {})['sentence_selection'] = info['sentence_selection']
    return trace


def validate_cohort_binding(binding, path):
    """The cohort file the manifest was cut from is unchanged, byte for byte and in its embedded identity."""
    if not isinstance(binding, dict) or not binding.get('path'):
        raise ValueError('Missing cohort binding')
    if Path(path).resolve() != Path(binding['path']).resolve():
        raise ValueError('Cohort manifest path differs from its binding')
    cohort_path = Path(binding['path'])
    if not cohort_path.is_file() or file_sha(cohort_path) != binding.get('file_sha256'):
        raise ValueError('Cohort manifest changed since planning')
    cohort = json.loads(cohort_path.read_text())
    if cohort.get('schema_version') != 1 or cohort.get('corpus') != 'card-v3' or not cohort.get('model'):
        raise ValueError('Expected v1 card-v3 cohort manifest')
    core = ('schema_version', 'model', 'seed', 'label_total', 'per_b', 'floor', 'traces_sha256')
    if any(key not in cohort for key in core):
        raise ValueError('Cohort manifest core fields missing')
    identity = digest({key: cohort[key] for key in core})
    if identity != cohort.get('manifest_sha256') or identity != binding.get('manifest_sha256'):
        raise ValueError('Cohort manifest identity changed since planning')


def verify_annotation_inputs(binding):
    """The label files used by cohort A are unchanged since the manifest was cut."""
    if binding is None:
        return
    inputs = binding.get('inputs')
    if not isinstance(inputs, list) or not inputs or not binding.get('annotations_sha256'):
        raise ValueError('Missing annotation input binding')
    checked = {}
    for item in inputs:
        path = str(Path(item['path']).resolve())
        if path in checked:
            if checked[path] != item['sha256']:
                raise ValueError('Conflicting annotation input hashes')
            continue
        if file_sha(path) != item['sha256']:
            raise ValueError('Annotation input changed since capture planning: ' + path)
        checked[path] = item['sha256']


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if digest({k: v for k, v in manifest.items() if k != 'binding'}) != manifest.get('binding'):
        raise ValueError('Capture manifest binding mismatch')
    if manifest.get('spec', {}).get('capture_version') != 2:
        raise ValueError('V3 native capture spec required')
    if 'outcome_binding' in manifest:
        raise ValueError('Capture manifests must not bind outcomes; replan with the cohort-only schema')
    validate_cohort_binding(manifest.get('cohort_binding'), manifest['cohort_manifest_path'])
    verify_annotation_inputs(manifest.get('annotation_binding'))
    if len({r['id'] for r in manifest['attempts']}) != len(manifest['attempts']):
        raise ValueError('Duplicate attempt IDs in capture manifest')
    return manifest


def resume_records(manifest, output):
    """Split the manifest's traces into finished (marker present and artifacts intact) and remaining."""
    completed, remaining, used = [], [], 0
    for row in manifest['attempts']:
        marker = Path(output) / (row['id'] + '.complete.json')
        if not marker.exists():
            remaining.append(row)
            continue
        record = json.loads(marker.read_text())
        if (record.get('manifest_binding') != manifest['binding'] or
                record.get('id') != row['id'] or record.get('tokens') != row['completion_tokens'] or
                record.get('trace_sha256') != row['trace_sha256']):
            raise ValueError('Incompatible capture resume: ' + row['id'])
        if len(record.get('files', [])) != 2:
            raise ValueError('Incomplete artifact marker')
        for artifact in record['files']:
            path = Path(artifact['path']).resolve()
            if not path.is_relative_to(Path(output).resolve()):
                raise ValueError('Artifact escapes immutable capture output')
            if path.stat().st_size != artifact['bytes'] or file_sha(path) != artifact['sha256']:
                raise ValueError('Corrupt accepted capture: ' + str(path))
            used += artifact['bytes']
        completed.append(record)
    return completed, remaining, used


def capture_folders(manifest, captures=None):
    """(folder of this shard, verification receipt) in the paper's layout or in a repeat tree.

    The paper's are the two paths the manifest names. A repeat tree `captures` holds one folder per shard number and
    one folder `verify`."""
    if captures is None:
        return Path(manifest['capture_dir']), Path(manifest['verification_path'])
    return Path(captures) / str(manifest['shard']), Path(captures) / 'verify' / 'verification.json'


def validate_verification(manifest, receipt):
    """The 2-trace verification of this model passed, on the longest input, with every routed layer."""
    if receipt.get('status') != 'passed' or receipt.get('validation_binding') != manifest.get('validation_binding'):
        raise ValueError('No accepted verify-only receipt for this spec/cohort')
    if not manifest.get('validation_binding'):
        raise ValueError('Missing verification binding')
    if receipt.get('maximum_input_tokens') != manifest['maximum_input_tokens']:
        raise ValueError('Longest input was not validated')
    if receipt.get('all_routed_layers') != manifest['spec']['layers'] or not receipt.get('exact_saved_token_coverage'):
        raise ValueError('Incomplete native verification coverage')
    if not receipt.get('unchanged_resume_verified'):
        raise ValueError('Unchanged resume was not validated')
    if manifest['source'] in DIFFERENTIAL_MODELS and receipt.get('differential', {}).get('status') != 'passed':
        raise ValueError('Native differential verification required')
