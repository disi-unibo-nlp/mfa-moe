"""Validate and import the tutor's September 2026 GPT-OSS/Gemma label bundle.

Run from the repository root with PYTHONPATH=src python3 -m
moe_exp.correlation_pipeline.import_stratified_labels [--apply].
Without --apply, only staged outputs are written. No inference is performed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from moe_exp.correlation_pipeline import spans
from moe_exp.gepaLLMAsJudge.data import SENTENCE_LABELS


class Trace(SimpleNamespace):
    """Minimal saved-trace interface used by the production span validators."""

    def model_copy(self, *, update):
        return Trace(**{**vars(self), **update})


def sha256(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stage_source(root, bundle, manifest, source, run, slug, stamp):
    base = root / 'results/correlation_pipeline' / run
    reasoning = base / 'reasoning-vllm-v1'
    stage = reasoning / f'annotations-import-{stamp}'
    backup = reasoning / f'annotations-backup-{stamp}'
    if stage.exists() or backup.exists():
        raise FileExistsError(f'Import destination already exists: {stage} or {backup}')
    rows = json.loads((bundle / 'merged' / source / 'annotations.json').read_text())
    summary = json.loads((bundle / 'merged' / source / 'summary.json').read_text())
    if len(rows) != summary['rows']:
        raise ValueError('Source row count mismatch')
    groups = defaultdict(list)
    for row in rows:
        identity = row['identity']
        groups[identity['dataset'], identity['problem_id']].append(row)
    output = defaultdict(list)
    failures = []
    versions = Counter()
    generation_hashes = {}
    for trace_file in sorted((base / 'generation' / slug).glob('*/traces.jsonl')):
        generation_hashes[str(trace_file.relative_to(root))] = sha256(trace_file)
        with trace_file.open() as handle:
            for line in handle:
                raw = json.loads(line)
                key = raw['dataset'], raw['problem_id']
                records = groups.pop(key, None)
                if records is None:
                    continue
                trace = Trace(**raw)
                trace_hash = spans.trace_digest(trace)
                units, selected = [], []
                for row in records:
                    ident, unit = row['identity'], row['unit']
                    if (ident['trace_sha256'] != trace_hash
                            or ident['sample_id'] != trace.sample_id
                            or ident['source_problem_id'] != trace.source_problem_id
                            or any(ident[a] != unit[b] for a, b in (
                                ('sentence_index', 'index'), ('start', 'start'), ('end', 'end')))
                            or row['inputs']['sentence'] != unit['text']
                            or trace.cot_text[unit['start']:unit['end']] != unit['text']):
                        raise ValueError(f'Source identity/text mismatch: {key}')
                    selected.append(unit['index'])
                    if row.get('label') in SENTENCE_LABELS:
                        units.append({**unit, 'label': row['label']})
                    elif row.get('status') == 'unknown' and 'label' not in row:
                        failures.append(row)
                    else:
                        raise ValueError(f'Invalid label: {key}')
                if len(selected) != len(set(selected)):
                    raise ValueError(f'Duplicate sentence: {key}')
                annotation = {
                    'schema_version': spans.SPAN_SCHEMA_VERSION,
                    'trace_sha256': trace_hash,
                    'classifier': {'imported_from': str(bundle.relative_to(root)),
                                   **manifest['judge']},
                    'dataset': trace.dataset, 'problem_id': trace.problem_id,
                    'reasoning_span': list(spans.reasoning_bounds(trace)),
                    'sentence_selection': {'schema_version': 1, 'indices': sorted(selected)},
                    'units': sorted(units, key=lambda unit: unit['index']),
                    'status': 'complete' if len(units) == len(selected) else 'partial',
                }
                # The export has offsets but no per-trace splitter version.
                # Accept a version only when every supplied unit matches exactly,
                # including unknown units; do not silently reindex sentences.
                for version in (1, 2):
                    annotation['splitter_version'] = version
                    candidate = trace.model_copy(update={'metadata': {
                        **trace.metadata, 'reasoning_annotation': annotation}})
                    expected = spans.sentence_spans(candidate)
                    if all(0 <= row['unit']['index'] < len(expected)
                           and expected[row['unit']['index']] == row['unit'] for row in records):
                        spans.validate_available_annotation(trace, annotation)
                        break
                else:
                    raise ValueError(f'No supported splitter matches: {key}')
                versions[version] += 1
                output[trace.dataset].append(annotation)
        print(f'{source}: validated {trace_file.parent.name}', flush=True)
    if groups:
        raise ValueError(f'Missing local traces: {list(groups)[:5]}')
    counts = Counter(row['label'] for rows_ in output.values()
                     for row in (unit for ann in rows_ for unit in ann['units']))
    if dict(counts) != summary['labels'] or len(failures) != summary['unknown']:
        raise ValueError('Validated label totals disagree with supplied summary')
    if sum(map(len, output.values())) != summary['traces']:
        raise ValueError('Validated trace count disagrees with supplied summary')
    report = {
        'source': source, 'bundle': str(bundle.relative_to(root)),
        'imported_at': datetime.now(timezone.utc).isoformat(),
        'source_file_sha256': sha256(bundle / 'merged' / source / 'annotations.json'),
        'labels': sum(counts.values()), 'unknown': len(failures),
        'traces': sum(map(len, output.values())), 'splitter_versions': dict(versions),
        'backup': str(backup.relative_to(root)), 'datasets': {},
        'generation_sha256': generation_hashes,
        'downstream_status': 'Existing forward/analysis outputs predate this import; recompute label-dependent outputs.',
    }
    for dataset, annotations in sorted(output.items()):
        dest = stage / slug / dataset / 'annotations.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w') as handle:
            for annotation in sorted(annotations, key=lambda ann: ann['problem_id']):
                handle.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        report['datasets'][dataset] = {
            'traces': len(annotations), 'labels': sum(len(a['units']) for a in annotations),
            'partial_traces': sum(a['status'] == 'partial' for a in annotations),
            'sha256': sha256(dest),
        }
    for source_path, name in [(bundle / 'manifest.json', 'source_manifest.json'),
                              (bundle / 'merged' / source / 'summary.json', 'source_summary.json'),
                              (bundle / 'sampling' / source / 'sampling_manifest.json', 'sampling_manifest.json')]:
        shutil.copyfile(source_path, stage / name)
    write_json(stage / 'failures.json', failures)
    write_json(stage / 'import_manifest.json', report)
    return stage, backup, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    bundle = root / 'data/mfa-moe-labels-stratified'
    for line in (bundle / 'SHA256SUMS.txt').read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (bundle / relative.lstrip('*')).resolve()
        if bundle not in path.parents or sha256(path) != expected:
            raise ValueError(f'Bundle checksum mismatch: {relative}')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    staged = []
    for source, run, slug in [
        ('gpt', 'gpt-oss-20b', 'openai--gpt-oss-20b'),
        ('gemma', 'gemma-nvfp4-nf4', 'nvidia--Gemma-4-26B-A4B-NVFP4'),
    ]:
        staged.append(stage_source(root, bundle, manifest, source, run, slug, stamp))
    # Both sources must validate before either active directory is changed.
    for stage, backup, report in staged:
        if args.apply:
            active = stage.parent / 'annotations'
            active.rename(backup)
            try:
                stage.rename(active)
            except Exception:
                backup.rename(active)
                raise
            (active.parent / 'ANNOTATION_IMPORT.md').write_text(
                f'# Stratified annotation import {stamp}\n\n'
                f'Active annotations: {report["labels"]:,} labels, {report["traces"]:,} traces; '
                f'{report["unknown"]} missing labels.\n\n'
                f'Previous annotations, including shards: `{backup.name}`.\n\n'
                'The annotations directory contains import/source/sampling manifests and failures.json. '
                'Selections include unknown units; affected traces are partial. '
                'The older sampling directory describes the previous experiment.\n\n'
                '**Existing forward and analysis outputs have not been recomputed.** '
                'They must not be reported as results of the new stratified sample. '
                'Use fresh output directories for replay/analysis, reuse the original generation '
                'directory and these annotations, and skip annotation generation. '
                'Qwen3.5 guiding/report outputs are unrelated and unchanged.\n')
        print(json.dumps(report, indent=2), flush=True)
    print('Imported both sources with backups.' if args.apply else 'Validated staged outputs; active data unchanged.')


if __name__ == '__main__':
    main()
