"""Attach independently saved class labels to native captures without replay.

New immutable attachment files bind the original capture, annotation inputs and
source manifest. Existing packets, tensors and outcome labels are never edited.
"""
import argparse
import gzip
import json
import os
from pathlib import Path

from moe_exp.capture.contract import CLASSES
from moe_exp.capture.production_plan import atomic, digest, read_location, sha


def annotation_key(identity):
    return (identity['dataset'], identity['problem_id'], identity['sample_id'], identity['trace_sha256'])


def load_annotation_records(paths):
    """Read annotation_batch's annotations.json lists or checkpoint documents."""
    groups, inputs = {}, []
    for raw_path in paths:
        path = Path(raw_path)
        before = sha(path)
        value = json.loads(path.read_text())
        if isinstance(value, list):
            records = value
            summary_path = path.with_name('summary.json')
            if path.name == 'annotations.json' and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary.get('status') != 'complete' or summary.get('annotations_sha256') != digest(records):
                    raise ValueError('Annotation summary binding mismatch')
                inputs.append(dict(path=str(summary_path.resolve()), sha256=sha(summary_path)))
        elif isinstance(value, dict) and isinstance(value.get('records'), list) and 'binding' in value:
            records = value['records']
        else:
            raise ValueError('Expected annotation_batch records')
        for record in records:
            identity, unit = record['identity'], record['unit']
            if identity['sentence_index'] != unit['index'] or any(identity[k] != unit[k] for k in ('start', 'end')):
                raise ValueError('Annotation identity/unit offsets mismatch')
            key = annotation_key(identity)
            index = unit['index']
            if type(index) is not int or index < 0:
                raise ValueError('Invalid sentence index')
            if record.get('status') == 'unknown':
                if record.get('label') is not None:
                    raise ValueError('Unknown record has a canonical label')
            elif record.get('label') not in CLASSES:
                raise ValueError('Invalid canonical class label')
            target = groups.setdefault(key, {})
            if index in target and target[index] != record:
                raise ValueError('Conflicting duplicate annotation sentence identity')
            # A/B selections can overlap in the common per-model label export.
            # Reuse only byte-equivalent decoded records; a conflict is never voted away.
            target[index] = record
        if sha(path) != before:
            raise ValueError('Annotation input changed during read')
        inputs.append(dict(path=str(path.resolve()), sha256=before))
    return groups, inputs


def assemble_annotation(row, membership, records, source_trace):
    """Require every selected sentence to be present, including unknown records."""
    identity = row['identity']
    key = (identity['dataset'], identity['problem_id'], identity['sample_id'], row['trace_sha256'])
    saved = records.get(key, {})
    selected = membership['sentence_selection']['indices']
    if not set(selected) <= set(saved):
        raise ValueError('Annotation coverage misses selected sentences')
    # Other records can belong to the other cohort. Attach this exact selection only.
    units = []
    for index in sorted(selected):
        record = saved[index]
        unit = dict(record['unit'])
        if record.get('status') == 'unknown':
            unit.update(status='unknown', raw_completion=record.get('raw_completion'), failure=record.get('failure'))
        else:
            unit['label'] = record['label']
        units.append(unit)
    return dict(schema_version=1, source_model=source_trace['model_id'], dataset=identity['dataset'],
                problem_id=identity['problem_id'], sample_id=identity['sample_id'],
                trace_sha256=row['trace_sha256'], sentence_selection=membership['sentence_selection'], units=units)


def count_experts_by_class(ids, sentences, experts, *, layers=None):
    """Count dispatched IDs only; unknown tokens never enter either class arm."""
    import numpy as np
    ids = np.asarray(ids)
    if ids.ndim != 3 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Expected integer native IDs [layers,tokens,top_k]')
    if layers is not None and len(layers) != ids.shape[0]:
        raise ValueError('Native routed-layer coverage mismatch')
    if experts < 1 or ids.size == 0 or ids.min() < 0 or ids.max() >= experts:
        raise ValueError('Invalid native expert IDs')
    if ids.shape[-1] > 1 and np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError('Duplicate expert in native dispatch')
    groups = {label: set() for label in CLASSES}
    owners = {}
    for sentence in sentences:
        label = sentence['label']
        if label is not None and label not in CLASSES:
            raise ValueError('Invalid class label')
        indices = sentence.get('token_indices')
        if indices is None:
            raise ValueError('Exact token/label alignment required')
        for token in indices:
            if type(token) is not int or not 0 <= token < ids.shape[1]:
                raise ValueError('Sentence token outside native capture')
            if token in owners:
                raise ValueError('Sentence tokens overlap')
            owners[token] = label
        if label in groups:
            groups[label].update(indices)
    def counts(tokens):
        ix = sorted(tokens)
        return [np.bincount(layer[ix].reshape(-1).astype(np.int64), minlength=experts).tolist() for layer in ids]
    observed = set().union(*groups.values())
    return dict(population='labeled reasoning tokens; class gaps excluded', tokens=len(observed),
                counts=counts(observed), classes={label: dict(tokens=len(tokens), counts=counts(tokens))
                                                 for label, tokens in groups.items()})


def attach_one(row, membership, done, manifest, records, annotation_inputs):
    """Load a verified saved native packet and create an attachment value."""
    if done.get('id') != row['id'] or done.get('manifest_binding') != manifest['binding']:
        raise ValueError('Native capture completion binding mismatch')
    if done.get('tokens') != row['completion_tokens']:
        raise ValueError('Native token coverage mismatch')
    files = done['files']
    if len(files) != 2 or any(sha(info['path']) != info['sha256'] for info in files):
        raise ValueError('Native capture artifact changed')
    with gzip.open(files[1]['path'], 'rt') as stream:
        packet = json.load(stream)
    if packet['id'] != row['id'] or packet['manifest_binding'] != manifest['binding']:
        raise ValueError('Native packet binding mismatch')
    import torch
    with gzip.open(files[0]['path'], 'rb') as stream:
        tensors = torch.load(stream, map_location='cpu', weights_only=True)
    ids = tensors['ids']
    expected = (len(manifest['spec']['layers']), row['completion_tokens'], manifest['spec']['top_k'])
    if tuple(ids.shape) != expected:
        raise ValueError('Native tensor coverage mismatch')
    from moe_exp.capture.sentence_schema import install
    install()
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest, validate_annotation
    from moe_exp.correlation_pipeline.dynamics.classes import sentence_table, saved_layout
    source = read_location(membership['locations'][0])
    trace = TraceRecord(**source)
    if trace_digest(trace) != row['trace_sha256']:
        raise ValueError('Attachment source trace mismatch')
    annotation = assemble_annotation(row, membership, records, source)
    validate_annotation(trace, annotation)
    sentences = sentence_table(trace, annotation, saved_layout(trace))
    counts = count_experts_by_class(ids.numpy(), sentences, manifest['spec']['experts'], layers=manifest['spec']['layers'])
    binding = dict(manifest_binding=manifest['binding'], capture_files=files,
                   annotations=annotation_inputs, cohort=membership['id'],
                   code_sha256=sha(Path(__file__)))
    return dict(schema_version=1, status='complete', id=row['id'], cohort=membership['id'],
                binding=binding, binding_sha256=digest(binding), source_trace=source,
                annotation=annotation, expert_class=counts,
                coverage=dict(selected_sentences=len(annotation['units']),
                              unknown_sentences=sum(u.get('status') == 'unknown' for u in annotation['units']),
                              labeled_reasoning_tokens=counts['tokens']),
                limitations=['Class attachment does not reconstruct discarded hidden/router tensors or class views'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, nargs='+', required=True)
    parser.add_argument('--cohort', choices=('A', 'B'), default='B')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Label attachment requires a Slurm CPU allocation')
    from moe_exp.capture.campaign_v3_plan import validate_storage_root
    output = validate_storage_root(args.output_root)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 3 or digest({k:v for k,v in manifest.items() if k != 'binding'}) != manifest['binding']:
        raise ValueError('Changed/unsupported capture manifest')
    records, inputs = load_annotation_records(args.annotations)
    results = []
    for row in manifest['attempts']:
        membership = next((c for c in row['cohorts'] if c['id'] == args.cohort), None)
        if membership is None:
            continue
        marker = Path(manifest['capture_dir'])/(row['id']+'.complete.json')
        done = json.loads(marker.read_text())
        value = attach_one(row, membership, done, manifest, records, inputs)
        path = output/args.cohort/(row['id']+'.labels.json')
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing label attachment binding changed')
        if not path.exists():
            atomic(path, value)
        results.append(dict(id=row['id'], path=str(path), sha256=sha(path), coverage=value['coverage']))
    if not results:
        raise ValueError('No capture attempts belong to requested label cohort')
    result = dict(status='complete', manifest_sha256=sha(args.manifest), manifest_binding=manifest['binding'],
                  cohort=args.cohort, annotation_inputs=inputs, records=results)
    path = output/args.cohort/(str(manifest['shard'])+'.result.json')
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Existing attachment summary changed')
    atomic(path, result)
    print(json.dumps(dict(status='complete', attempts=len(results), path=str(path))), flush=True)


if __name__ == '__main__':
    main()
