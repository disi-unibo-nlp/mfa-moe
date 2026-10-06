"""Guarded operational adapter for the frozen FRESH dense source schema.

The original source, scientific pipeline and measurement plan stay unchanged.
Checked metadata normalization has its own seal and explicit provenance.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path

import fresh_frame_recovery_v1 as context
import generated_dense_pipeline_v1 as native

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
SCRIPTS = REPO / 'scripts/experimental_resume'
AMENDMENT = DOC / 'GENERATED_DENSE_FRESH_METADATA_RECOVERY_AMENDMENT_v1.json'
ORIGINAL_PLAN = DOC / 'GENERATED_DENSE_MEASUREMENT_PLAN_v1.json'
OPERATIONAL_FILES = tuple(SCRIPTS / name for name in (
    'fresh_frame_recovery_v1.py', 'generated_dense_fresh_metadata_recovery_v1.py',
    'dispatch_generated_dense_fresh_metadata_recovery_v1.py',
    'prepare_generated_dense_fresh_metadata_recovery_v1.sbatch',
    'dispatch_generated_dense_fresh_metadata_recovery_v1.sbatch',
    'label_generated_dense_fresh_metadata_recovery_v1.sbatch',
    'analyze_generated_dense_fresh_metadata_recovery_v1.sbatch'))
UNCHANGED_DISPATCH_FILES = tuple(SCRIPTS / name for name in (
    'dispatch_generated_dense_v1.py', 'dispatch_overnight_readers_v1.py'))
_native_measurement_code = native.measurement_code
_native_inputs = native.inputs
_native_save = native.save
require = native.require
sealed = native.sealed
file_sha = native.file_sha
digest = native.digest


def measurement_code():
    return {**_native_measurement_code(), **{str(p.resolve()): file_sha(p) for p in OPERATIONAL_FILES}}


def validate_amendment(amendment, manifest):
    context.check_seal(amendment)
    context.check_seal(manifest)
    plan = sealed(ORIGINAL_PLAN)
    require(amendment['schema'] == 'generated-dense-fresh-metadata-recovery-amendment-v1' and
            amendment['manifest_path'] == str(context.MANIFEST.resolve()) and
            amendment['manifest_sha256'] == manifest['sha256'] == sealed(context.MANIFEST)['sha256'] and
            amendment['source_frame_sha256'] == manifest['source_frame_sha256'] and
            amendment['source_enrollment_sha256'] == manifest['source_enrollment_sha256'] and
            amendment['original_plan_sha256'] == plan['sha256'] and
            amendment['frozen_code_files'] == plan['code_files'] == _native_measurement_code() and
            amendment['operational_code_files'] == {str(p.resolve()): file_sha(p) for p in OPERATIONAL_FILES} and
            amendment['unchanged_dispatch_code_files'] ==
            {str(p.resolve()): file_sha(p) for p in UNCHANGED_DISPATCH_FILES} and
            amendment['measurement_code_digest'] == digest(measurement_code()) and
            all(file_sha(p) == sha for p, sha in plan['code_files'].items()) and
            all(file_sha(p) == sha for p, sha in manifest['code_files'].items()),
            'dense recovery amendment or unchanged frozen source binding differs')
    for path, sha in amendment['frozen_artifacts'].items():
        require(sealed(path)['sha256'] == sha, 'dense recovery frozen artifact changed')
    parent = Path(amendment['dense_parent']).resolve()
    require(str(parent).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/') and
            parent.name == 'generated-dense-fresh-metadata-recovery-v1',
            'dense recovery output parent is outside its dedicated owned location')
    return amendment


def recovery_fields(amendment, provenance):
    return {'amendment_sha256': amendment['sha256'],
            'operational_adapter_sha256': file_sha(__file__),
            'source_normalizer_sha256': file_sha(context.__file__),
            'original_measurement_plan_sha256': amendment['original_plan_sha256'],
            'scientific_functions_unchanged': True, **provenance}


def checked_recovery(manifest):
    amendment = validate_amendment(sealed(AMENDMENT), manifest)
    derived, contexts, provenance = context.load_contexts(manifest)
    require(provenance == amendment['source_context_provenance'],
            'dense recovery normalized source provenance differs from amendment')
    return amendment, derived, contexts, recovery_fields(amendment, provenance)


def output_path(manifest, parent):
    amendment = validate_amendment(sealed(AMENDMENT), manifest)
    require(Path(parent).resolve() == Path(amendment['dense_parent']).resolve(),
            'dense recovery cannot reuse a failed original output parent')
    return native.output_path(manifest, parent)


def build_frame(manifest, price, run_out, directory, tokenizer):
    amendment, derived, contexts, provenance = checked_recovery(manifest)
    require(directory == output_path(manifest, directory.parent), 'dense recovery directory binding differs')
    require(Path(run_out).resolve() == Path(amendment['generation_out']).resolve(),
            'dense recovery generation output path differs')
    stage, assigned, outputs, arrays, receipts = native.source_outputs(manifest, price, run_out)
    originals = {r['uid']: r for r in derived['records']}
    natives = {r['uid']: r for r in manifest['rows']}
    normalized = _native_save(directory / 'NORMALIZED_SOURCE_CONTEXT.json',
                              {k: v for k, v in derived.items() if k != 'sha256'})
    require(normalized['sha256'] == provenance['normalized_context_sha256'],
            'normalized source has a different derived seal')
    map_path = directory / 'ARM_MAP.json'
    salt = bytes.fromhex(sealed(map_path)['blind_salt_hex']) if map_path.exists() else os.urandom(32)
    records, blind_rows = [], []
    for expected, result in zip(assigned, outputs, strict=True):
        start = natives[expected['prefix_uid']]
        original = originals[start['uid']]
        data, meta = contexts[start['uid']], original['analysis_meta']
        require(original['family'] == start['family'] and original['transition'] == start['transition'] and
                original['prefix_tokens'] == len(start['prefix_ids']) and
                meta['prefix_ids_sha256'] == start['prefix_ids_sha256'] == digest(start['prefix_ids']) and
                meta['prompt_ids_sha256'] == start['prompt_ids_sha256'] == digest(start['prompt_ids']),
                'normalized exact prefix and generated assignment are not aligned')
        tokens = result['tokens']
        require(len(tokens) <= manifest['horizon'], 'generation exceeds assigned horizon')
        layout = native.sentence_layout(tokenizer, start['prompt_ids'], tokens)
        blind_rows.extend(native.make_blind_inputs(layout, data['problem'], data['triggering_sentence'],
                                                   expected['uid'], salt))
        records.append({**expected, **layout, 'emitted_token_ids': tokens,
                        'emitted_token_ids_sha256': digest(tokens), 'array': arrays[expected['uid']],
                        'finish_reason': result['finish'], 'error': result['error'],
                        'hit_cap': len(tokens) == manifest['horizon'],
                        'action_dose': result.get('action_dose'),
                        'inactive_native_checks': result.get('inactive_native_checks'),
                        'missing_telemetry_ranks': result.get('missing_telemetry_ranks')})
    blind_rows.sort(key=lambda r: r['blind_id'])
    require(len({r['blind_id'] for r in blind_rows}) == len(blind_rows), 'blind sentence identity collision')
    codes = measurement_code()
    shared = {'generation_manifest_sha256': manifest['sha256'], 'generation_stage_sha256': stage['sha256'],
              'source_frame_path': manifest['source_frame_path'],
              'source_frame_sha256': manifest['source_frame_sha256'],
              'normalized_context_sha256': normalized['sha256'], 'recovery_provenance': provenance,
              'code_files': codes, 'code_digest': digest(codes), 'horizon': manifest['horizon'],
              'generation_run_out': str(Path(run_out).resolve())}
    arm_map = _native_save(map_path, {**shared, 'schema': 'generated-dense-arm-map-v1',
                 'assigned': len(assigned), 'blind_salt_hex': salt.hex(),
                 'generation_batch_receipts': receipts,
                 'tokenizer_config_sha256': file_sha(native.TOKENIZER / 'tokenizer_config.json'), 'records': records})
    frame = _native_save(directory / 'BLIND_FRAME.json', {**shared, 'schema': 'generated-dense-blind-frame-v1',
                 'assigned': len(assigned), 'sentences': len(blind_rows), 'records': blind_rows,
                 'input_allowlist': ['problem_statement', 'previous_sentence', 'sentence', 'next_sentence'],
                 'scope': 'Offline dense class audit with observed next sentence; never online detector input.'})
    return arm_map, frame


def validate_recovered_inputs(directory, manifest, frame, price, arm_map, derived, provenance):
    """Check recovery-only joins in addition to unchanged native price/profile checks."""
    for value in (frame, price, arm_map, derived):
        context.check_seal(value)
    require(frame.get('recovery_provenance') == arm_map.get('recovery_provenance') ==
            price.get('recovery_provenance') == provenance and
            frame.get('normalized_context_sha256') == arm_map.get('normalized_context_sha256') ==
            provenance['normalized_context_sha256'] == derived['sha256'] and
            frame['source_frame_sha256'] == arm_map['source_frame_sha256'] == manifest['source_frame_sha256'] and
            frame['source_frame_path'] == arm_map['source_frame_path'] == manifest['source_frame_path'] and
            frame['generation_manifest_sha256'] == arm_map['generation_manifest_sha256'] == manifest['sha256'] and
            frame['generation_stage_sha256'] == arm_map['generation_stage_sha256'] and
            frame['horizon'] == arm_map['horizon'] == manifest['horizon'] and
            arm_map['assigned'] == frame['assigned'] == manifest['expected_requests'] and
            arm_map['code_files'] == frame['code_files'] == measurement_code() and
            arm_map['code_digest'] == frame['code_digest'] and
            arm_map['tokenizer_config_sha256'] == file_sha(native.TOKENIZER / 'tokenizer_config.json'),
            'dense recovered source/map/price provenance differs')
    import overnight_routing_runner_v2 as runner
    expected = [meta for _, meta in runner.request_metadata(manifest, manifest['rows'], manifest.get('arms'))]
    require(len(expected) == len(arm_map['records']) and
            all(all(row.get(k) == v for k, v in assigned.items())
                for assigned, row in zip(expected, arm_map['records'], strict=True)) and
            len({row['uid'] for row in arm_map['records']}) == len(expected) and
            {s['blind_id'] for row in arm_map['records'] for s in row['sentences']} ==
            {row['blind_id'] for row in frame['records']},
            'dense recovered exact assignment/sentence enrollment differs')
    require(all(row['emitted_token_ids_sha256'] == digest(row['emitted_token_ids']) and
                len(row['emitted_token_ids']) <= manifest['horizon'] and
                row['hit_cap'] == (len(row['emitted_token_ids']) == manifest['horizon'])
                for row in arm_map['records']), 'dense recovered continuation token/horizon binding differs')
    original_contexts = {row['uid']: row['reader_input'] for row in derived['records']}
    expected_blind = []
    salt = bytes.fromhex(arm_map['blind_salt_hex'])
    for row in arm_map['records']:
        data = original_contexts[row['prefix_uid']]
        layout = {**row, 'sentences': [dict(sentence) for sentence in row['sentences']]}
        expected_blind.extend(native.make_blind_inputs(layout, data['problem'], data['triggering_sentence'],
                                                       row['uid'], salt))
    require(sorted(expected_blind, key=lambda row: row['blind_id']) == frame['records'],
            'dense recovered blind sentence text/context binding differs')
    require(Path(directory) == output_path(manifest, Path(directory).parent),
            'dense recovered output path differs')


def inputs(directory):
    manifest = sealed(context.MANIFEST)
    _, derived, _, provenance = checked_recovery(manifest)
    directory = Path(directory)
    stored = sealed(directory / 'NORMALIZED_SOURCE_CONTEXT.json')
    require(stored == derived, 'dense normalized context changed after preparation')
    frame, price = _native_inputs(directory)
    validate_recovered_inputs(directory, manifest, frame, price, sealed(directory / 'ARM_MAP.json'),
                              stored, provenance)
    return frame, price


def operational_save(path, body):
    if Path(path).name == 'TRAJECTORY_RESULT.json':
        frame = sealed(Path(path).parent / 'BLIND_FRAME.json')
        body = {**body, 'recovery_provenance': frame['recovery_provenance'],
                'normalized_context_sha256': frame['normalized_context_sha256']}
    return _native_save(path, body)


def install_adapter():
    native.measurement_code = measurement_code
    native.inputs = inputs
    native.save = operational_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'label', 'analyze'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--generation-price', type=Path)
    parser.add_argument('--run-out', type=Path)
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    args = parser.parse_args()
    native.require_step(gpu=args.stage == 'label')
    install_adapter()
    if args.stage == 'prepare':
        require(args.manifest and args.generation_price and args.run_out and args.out_root,
                'dense recovery prepare requires exact source paths')
        require(args.manifest.resolve() == context.MANIFEST.resolve() and
                args.generation_price.resolve() == context.PRICE.resolve() and args.max_wall_seconds == 7200,
                'dense recovery prepare source/price/horizon differs')
        manifest = sealed(args.manifest)
        directory = output_path(manifest, args.out_root)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'PREPARE.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            native.seal_generation_if_needed(manifest, args.manifest, args.run_out)
            tokenizer = AutoTokenizer.from_pretrained(native.TOKENIZER, local_files_only=True)
            _, frame = build_frame(manifest, sealed(args.generation_price), args.run_out, directory, tokenizer)
            judge = AutoTokenizer.from_pretrained(native.qualified.MODEL, local_files_only=True)
            price = _native_save(directory / 'LABEL_PRICE.json', {
                **native.price_frame(frame, judge, args.max_wall_seconds),
                'recovery_provenance': frame['recovery_provenance']})
            inputs(directory)
        import json
        print(json.dumps({'directory': str(directory), 'sentences': frame['sentences'],
                          'shards': len(price['shards']), 'GPU_h': price['complete_stage_GPU_h']}), flush=True)
        return
    require(args.directory is not None, 'prepared dense recovery directory required')
    if args.stage == 'label':
        result, _ = native.label_shard(args.directory, args.shard_index)
    else:
        result = native.analyze(args.directory)
    import json
    print(json.dumps({'status': 'COMPLETE', 'stage': args.stage, 'sha256': result['sha256']}), flush=True)


if __name__ == '__main__':
    main()
