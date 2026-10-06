"""Additive, fail-closed source-schema repair for frozen fresh v2 measurements.

The original frame stays sealed. Derived identities come from its sealed
selection and are checked against enrollment, exact tokens and native units.
Scientific builder/reader/analysis/selection functions remain v2; operational
entry points have separate hashes in the recovery amendment and artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base
import rate_overnight_semantics_v2 as storage

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
SCRIPTS = REPO / 'scripts/experimental_resume'
AMENDMENT = DOC / 'OVERNIGHT_FRESH_FRAME_RECOVERY_AMENDMENT_v1.json'
SELECTION = DOC / 'MECHANISM_EXTENSION_START_SELECTION_v1.json'
UNITS = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
             'steering-v1/runs/routing-control-v1/dense-mechanism-extension-220/'
             'MECHANISM_EXTENSION_UNITS_v1.json')
MANIFEST = DOC / 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'
PRICE = DOC / 'OVERNIGHT_FRESH_COMPARISON_PRICE_v2.json'
ENVELOPE = DOC / 'OVERNIGHT_FRESH_COMPARISON_ENVELOPE_v2.json'
FREEZE = DOC / 'OVERNIGHT_MEASUREMENT_PIPELINE_v2.json'
require = storage.require


def check_seal(value):
    require(value.get('sha256') == base.digest({k: v for k, v in value.items() if k != 'sha256'}),
            'recovery input JSON seal differs')


def unique(rows, key, label):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), 'duplicate ' + label)
    return result


def checked_contexts(manifest, source, enrollment, selection, units):
    """Return a separately sealed derived frame, exact contexts, and provenance.

    No identity is inferred from a question or substituted from the manifest.
    Only explicitly enrolled UIDs are normalized, preserving their order.
    """
    for value in (manifest, source, enrollment, selection, units):
        check_seal(value)
    require(manifest['schema'] == 'overnight-routing-manifest-v2' and
            source['schema'] == 'mechanism-extension-start-frame-v1' and
            enrollment['schema'] == 'extension-overnight-enrollment-v1' and
            enrollment['status'] == 'READY_EXACT_NATIVE_PREFIXES' and
            selection['schema'] == 'mechanism-extension-start-selection-v1' and
            units['schema'] == 'dense-mechanism-extension-units-v1' and
            manifest['source_frame_sha256'] == enrollment['frame_sha256'] == source['sha256'] and
            manifest['source_enrollment_sha256'] == enrollment['sha256'] and
            enrollment['selection_sha256'] == source['selection_sha256'] == selection['sha256'] and
            source['units_sha256'] == selection['units_sha256'] == units['sha256'] and
            source['extension_family_freeze_sha256'] == enrollment['extension_family_freeze_sha256'] ==
            selection['extension_family_freeze_sha256'] == units['extension_family_freeze_sha256'] and
            source['visible_input_allowlist'] == ['problem', 'emitted_prefix', 'triggering_sentence'] and
            source['rows'] == len(source['records']) == len(selection['records']) and
            [r['uid'] for r in source['records']] == [r['uid'] for r in selection['records']],
            'fresh source/selection/units/enrollment provenance differs')
    originals = unique(source['records'], lambda r: r['uid'], 'source UID')
    selected = unique(selection['records'], lambda r: r['uid'], 'selection UID')
    enrolled = unique(enrollment['rows'], lambda r: r['uid'], 'enrollment UID')
    natives = unique(manifest['rows'], lambda r: r['uid'], 'manifest UID')
    by_unit = unique(units['records'], lambda r: (r['attempt_id'], r['sentence_index']), 'native unit')
    require(set(natives) == set(enrolled) and set(natives) <= set(originals),
            'exact enrolled UID set differs')
    normalized, contexts = [], {}
    identity = ('uid', 'family', 'attempt_id', 'transition', 'sentence_index', 'prefix_tokens')
    for uid, native in natives.items():
        original, event = originals[uid], selected[uid]
        require(native == enrolled[uid] and all(native[k] == event[k] for k in identity) and
                uid == base.digest(['mechanism-extension-start-uid-v1', event['transition'],
                                    event['family'], event['attempt_id'], event['sentence_index']]),
                'frozen enrolled/selected start identity differs')
        require(set(original) == {'uid', 'transition', 'reader_input', 'analysis_meta'} and
                original['transition'] == native['transition'], 'fresh reader record schema differs')
        unit = by_unit.get((event['attempt_id'], event['sentence_index']))
        require(unit is not None and unit['family'] == event['family'] and
                unit['segment'] == event['segment'] and unit['token_end'] == event['prefix_tokens'] and
                native['question'] == native['canonical_question'] == unit['question'],
                'selected start differs from frozen native unit')
        data, meta = original['reader_input'], original['analysis_meta']
        require(set(data) == {'problem', 'emitted_prefix', 'triggering_sentence'} and
                all(isinstance(v, str) and v for v in data.values()) and
                set(meta) == {'prefix_tokens', 'prefix_ids_sha256', 'prompt_ids_sha256',
                              'prefix_text_sha256', 'trace_sha256', 'tokenizer_sha256'},
                'fresh source context/meta schema differs')
        require(meta['prefix_tokens'] == native['prefix_tokens'] == len(native['prefix_ids']) and
                1 <= len(native['prefix_ids']) <= 8192 and native['prompt_ids'] and
                all(type(t) is int and t >= 0 for t in native['prefix_ids'] + native['prompt_ids']) and
                all(meta[k + '_sha256'] == native[k + '_sha256'] == base.digest(native[k])
                    for k in ('prefix_ids', 'prompt_ids')),
                'exact native prefix/prompt token binding differs')
        require(meta['trace_sha256'] == unit['trace_sha256'] and
                meta['tokenizer_sha256'] == event['tokenizer_sha256'] and
                all(isinstance(meta[k], str) and len(meta[k]) == 64 and
                    all(c in '0123456789abcdef' for c in meta[k])
                    for k in ('trace_sha256', 'tokenizer_sha256')) and
                meta['prefix_text_sha256'] == hashlib.sha256(data['emitted_prefix'].encode()).hexdigest() and
                len(data['emitted_prefix']) == event['prefix_end_char'] and
                0 <= unit['char_start'] < unit['char_end'] <= event['prefix_end_char'] and
                data['emitted_prefix'][unit['char_start']:unit['char_end']] ==
                data['triggering_sentence'] == unit['inputs']['sentence'] and
                data['problem'] == unit['inputs']['problem_statement'] and
                data['emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()),
                'native trace/tokenizer/text/problem provenance differs')
        normalized.append({**original, **{k: event[k] for k in identity}})
        contexts[uid] = data
    provenance = {'source_frame_sha256': source['sha256'], 'source_enrollment_sha256': enrollment['sha256'],
                  'source_selection_sha256': selection['sha256'], 'source_units_sha256': units['sha256'],
                  'enrolled_uid_order_sha256': base.digest(list(natives)), 'normalized_records': len(normalized)}
    body = {'schema': 'fresh-start-context-normalization-v1', **provenance, 'records': normalized}
    derived = {**body, 'sha256': base.digest(body)}
    return derived, contexts, {**provenance, 'normalized_context_sha256': derived['sha256']}


def load_contexts(manifest):
    return checked_contexts(manifest, base.sealed(manifest['source_frame_path']),
                            base.sealed(manifest['source_enrollment_path']),
                            base.sealed(SELECTION), base.sealed(UNITS))


def validate_amendment(amendment, manifest):
    check_seal(amendment)
    expected_sha = os.environ.get('OVERNIGHT_FRESH_RECOVERY_AMENDMENT_SHA256')
    require(expected_sha is None or amendment['sha256'] == expected_sha,
            'recovery amendment differs from submitted binding')
    require(amendment['schema'] == 'overnight-fresh-frame-recovery-amendment-v1' and
            amendment['manifest_sha256'] == manifest['sha256'] and
            amendment['source_frame_sha256'] == manifest['source_frame_sha256'] and
            amendment['source_enrollment_sha256'] == manifest['source_enrollment_sha256'] and
            amendment['measurement_freeze_sha256'] == base.sealed(FREEZE)['sha256'] and
            all(base.file_sha(path) == sha for path, sha in amendment['frozen_code_files'].items()) and
            all(base.file_sha(path) == sha for path, sha in amendment['operational_code_files'].items()),
            'recovery amendment/source/code binding differs')
    for path, sha in amendment['frozen_artifacts'].items():
        require(base.sealed(path)['sha256'] == sha, 'recovery frozen artifact binding differs')
    require(amendment['operational_code_files'][str(Path(__file__).resolve())] == base.file_sha(__file__),
            'unbound recovery contract')
    return amendment


def recovery_fields(amendment, context_provenance, entry):
    entry = Path(entry).resolve()
    require(amendment['operational_code_files'].get(str(entry)) == base.file_sha(entry),
            'unbound recovery entry point')
    return {'amendment_sha256': amendment['sha256'],
            'context_adapter_sha256': base.file_sha(SCRIPTS / 'build_overnight_blind_frame_fresh_adapter_v1.py'),
            'source_normalizer_sha256': base.file_sha(__file__), **context_provenance,
            'scientific_functions_unchanged': True}


def validate_measurement(manifest, arm_map, frame, price, amendment):
    validate_amendment(amendment, manifest)
    _, _, provenance = load_contexts(manifest)
    expected = recovery_fields(amendment, provenance, SCRIPTS / 'build_overnight_blind_frame_fresh_adapter_v1.py')
    for value in (arm_map, frame, price):
        check_seal(value)
    require(arm_map.get('recovery_provenance') == frame.get('recovery_provenance') == expected and
            arm_map['manifest_sha256'] == frame['generation_manifest_sha256'] == manifest['sha256'] and
            arm_map['source_frame_sha256'] == frame['source_frame_sha256'] == manifest['source_frame_sha256'] and
            arm_map['builder_sha256'] == frame['builder_sha256'] ==
            base.file_sha(SCRIPTS / 'build_overnight_blind_frame_v2.py') and
            arm_map['analysis_driver_sha256'] == frame['analysis_driver_sha256'] ==
            base.file_sha(SCRIPTS / 'analyze_overnight_semantics_v2.py'),
            'recovered frame/map adapter or scientific provenance differs')
    import build_overnight_blind_frame_v2 as builder
    plan = builder.expected_assignments(manifest)
    require(len(plan) == len(arm_map['records']) == frame['assigned'] and
            all(all(row.get(k) == v for k, v in assigned.items())
                for assigned, row in zip(plan, arm_map['records'], strict=True)) and
            {r['blind_id'] for r in frame['records']} ==
            {r['blind_id'] for r in arm_map['records'] if r['measurement_status'] == 'gradeable'},
            'recovered measurement exact assignment IDs differ')
    storage.validate(frame, price)
    return expected


def validate_analysis(manifest, analysis, amendment):
    validate_amendment(amendment, manifest)
    check_seal(analysis)
    paths = amendment['paths']
    amap, frame, price = (base.sealed(Path(paths['measurement']) / name)
                          for name in ('ARM_MAP.json', 'BLIND_FRAME.json', 'READER_PRICE.json'))
    fields = validate_measurement(manifest, amap, frame, price, amendment)
    assigned = base.sealed(Path(paths['analysis']) / 'ASSIGNED_RESULTS.json')
    reader_stage = base.sealed(Path(paths['ratings']) / 'STAGE_SUMMARY.json')
    require(analysis.get('recovery_provenance') == fields and
            analysis.get('analysis_operational_entry_sha256') ==
            base.file_sha(SCRIPTS / 'analyze_overnight_semantics_fresh_adapter_v1.py') and
            analysis['arm_map_sha256'] == amap['sha256'] and analysis['frame_sha256'] == frame['sha256'] and
            analysis['price_sha256'] == price['sha256'] and
            analysis['assigned_results_sha256'] == assigned['sha256'] and
            analysis['reader_stage_sha256'] == assigned['reader_stage_sha256'] == reader_stage['sha256'] and
            assigned.get('recovery_provenance') == fields and
            assigned['frame_sha256'] == reader_stage['frame_sha256'] == frame['sha256'] and
            reader_stage['price_sha256'] == price['sha256'] and
            len(assigned['records']) == manifest['expected_requests'] and
            analysis['analysis_driver_sha256'] == base.file_sha(SCRIPTS / 'analyze_overnight_semantics_v2.py') and
            analysis['manifest_sha256'] == manifest['sha256'] and
            analysis['assigned'] == manifest['expected_requests'], 'recovered analysis adapter or evidence differs')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-measurement', action='store_true')
    parser.add_argument('--check-analysis', type=Path)
    args = parser.parse_args()
    m, a = base.sealed(MANIFEST), base.sealed(AMENDMENT)
    validate_amendment(a, m)
    if args.check_analysis:
        validate_analysis(m, base.sealed(args.check_analysis), a)
    elif args.check_measurement:
        p = Path(a['paths']['measurement'])
        validate_measurement(m, *(base.sealed(p / name) for name in
                             ('ARM_MAP.json', 'BLIND_FRAME.json', 'READER_PRICE.json')), a)
    else:
        _, _, provenance = load_contexts(m)
        print(json.dumps(provenance, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
