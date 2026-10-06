"""Eligible ITT analysis keeps every assignment and joins only through blind IDs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/'
          'analyze_eligible_immediate_semantics_v1.py')
SPEC = importlib.util.spec_from_file_location('eligible_analysis_v1', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

ARMS = ['native', 'native_duplicate', 'target_bias0.5', 'target_bias1',
        'random_bias0.5', 'random_bias1']
ORDER = ['target_bias1', 'native', 'random_bias0.5', 'target_bias0.5',
         'native_duplicate', 'random_bias1']
V4_ARMS = ['native', 'native_duplicate', 'target_bias_minus1', 'target_force_off',
           'random_bias_minus1', 'random_force_off']
V4_ORDER = ['target_force_off', 'native', 'random_bias_minus1', 'target_bias_minus1',
            'native_duplicate', 'random_force_off']
FAMILIES = [('fam-c1', 'candidate_to_verify'), ('fam-c2', 'candidate_to_verify'),
            ('fam-a1', 'approach_to_commit'), ('fam-a2', 'approach_to_commit')]
SEEDS = [0, 1]


def fixture(*, schema='v3', ungradeable=None, parse_fail=None):
    arms = ARMS if schema == 'v3' else V4_ARMS
    order = ORDER if schema == 'v3' else V4_ORDER
    manifest = {
        'schema': ('routing-eligible-micro-serial-v3' if schema == 'v3'
                   else 'routing-eligible-deactivation-serial-v4'),
        'sha256': 'manifest-seal',
        'expected_requests': len(FAMILIES) * len(SEEDS) * len(arms),
        'seeds': SEEDS,
        'arms': [{'name': arm,
                  'role': ('native' if arm in ('native', 'native_duplicate') else
                           'target' if arm.startswith('target') else 'random')}
                 for arm in arms],
        'rows': [{'family': family, 'canonical_question': 'q-' + family,
                  'transition': transition} for family, transition in FAMILIES],
        'arm_order_by_family_seed': {family: [order, order]
                                     for family, _ in FAMILIES},
    }
    map_rows, frame_rows, ratings = [], [], {}
    for family, transition in FAMILIES:
        for seed in SEEDS:
            for arm in arms:
                blind_id = f'blind|{family}|{seed}|{arm}'
                status = ('generation_error' if ungradeable == (family, seed, arm)
                          else 'gradeable')
                map_rows.append({
                    'uid': f'uid|{family}|{seed}|{arm}', 'prefix_uid': f'prefix|{family}',
                    'family': family, 'transition': transition,
                    'canonical_question': 'q-' + family, 'seed': seed, 'arm': arm,
                    'role': next(row['role'] for row in manifest['arms']
                                 if row['name'] == arm),
                    'blind_id': blind_id, 'measurement_status': status,
                    'emitted_tokens': 10 if status == 'gradeable' else 0,
                    'action_dose': {}, 'base_hook_dose': {},
                })
                if status != 'gradeable':
                    continue
                frame_rows.append({'blind_id': blind_id, 'reader_input': {
                    'transition': transition, 'problem': 'problem ' + family,
                    'full_emitted_prefix': 'prefix ' + family,
                    'triggering_sentence': 'trigger ' + family,
                    'continuation': 'continuation ' + family}})
                positive = arm.startswith('target')
                ratings[blind_id] = {}
                for reader in (0, 1):
                    ratings[blind_id][reader] = {
                        'blind_id': blind_id,
                        'rating': None if parse_fail == (family, seed, arm, reader) else positive,
                        'finish_reason': 'stop', 'prompt_tokens': 5, 'generated_tokens': 5}
    batches = []
    for start in range(0, len(frame_rows), analysis.BATCH):
        block = frame_rows[start:start + analysis.BATCH]
        for reader in (0, 1):
            batches.append({'schema': 'eligible-immediate-rating-batch-v1',
                            'sha256': f'batch-{start}-{reader}', 'start': start,
                            'reader': reader,
                            'records': [ratings[row['blind_id']][reader] for row in block]})
    binding = {'schema': 'eligible-immediate-blind-rating-binding-v1',
               'sha256': 'binding-seal', 'frame_sha256': 'frame-seal'}
    frame = {'schema': 'eligible-immediate-blind-frame-v1', 'sha256': 'frame-seal',
             'generation_manifest_sha256': 'manifest-seal',
             'generation_summary_sha256': 'gen-summary',
             'reader_input_allowlist': sorted(analysis.ALLOWLIST),
             'continuation_max_tokens': 256, 'records': frame_rows}
    arm_map = {'schema': 'eligible-immediate-arm-map-v1', 'sha256': 'map-seal',
               'manifest_sha256': 'manifest-seal',
               'generation_summary_sha256': 'gen-summary', 'records': map_rows}
    summary = {'schema': 'eligible-immediate-rating-summary-v1',
               'sha256': 'rating-summary', 'binding_sha256': 'binding-seal',
               'frame_sha256': 'frame-seal', 'rows': len(frame_rows),
               'ratings': 2 * len(frame_rows),
               'batch_sha256': [batch['sha256'] for batch in batches]}
    return manifest, arm_map, frame, binding, summary, batches


def run(fixture_data):
    manifest, arm_map, frame, binding, summary, batches = fixture_data
    return analysis.run_analysis(manifest, arm_map, frame, binding, summary, batches,
                                 n_boot=200, seed=1, driver_sha='driver')


def contrast(body, arm, reference):
    return next(row for row in body['primary']['result']['contrasts']
                if row['arm'] == arm and row['reference'] == reference and
                row['metric'] == 'semantic_success')


def test_primary_contrast_uses_all_assigned_rows():
    body = run(fixture())
    assert body['population']['assigned'] == 48
    assert body['population']['both_readers_positive'] == 16
    assert contrast(body, 'target_bias1', 'native')['estimate'] == 1.0
    assert contrast(body, 'target_bias1', 'random_bias1')['estimate'] == 1.0


def test_ungradeable_row_stays_in_the_itt_denominator():
    body = run(fixture(ungradeable=('fam-c1', 0, 'target_bias1')))
    assert body['population']['assigned'] == 48
    assert body['population']['gradeable'] == 47
    assert contrast(body, 'target_bias1', 'native')['estimate'] == pytest.approx(.875)


def test_unparsed_reader_is_negative_for_primary_and_positive_for_sensitivity():
    data = fixture(parse_fail=('fam-a1', 0, 'target_bias0.5', 1))
    joined = analysis.join_records(*data[:4], data[4], data[5])
    row = next(record for record in joined
               if (record['family'], record['seed'], record['arm']) ==
               ('fam-a1', 0, 'target_bias0.5'))
    assert row['both_readers_positive'] == 0
    assert row['at_least_one_reader_positive'] == 1
    assert row['operational_status'] == 'unscored'
    body = run(data)
    assert body['population']['both_readers_positive'] == 15


def test_duplicate_blind_id_is_rejected():
    data = fixture()
    manifest, arm_map, frame, binding, summary, batches = data
    arm_map['records'][1]['blind_id'] = arm_map['records'][0]['blind_id']
    with pytest.raises(ValueError, match='duplicate blind'):
        analysis.run_analysis(manifest, arm_map, frame, binding, summary, batches,
                              n_boot=200, seed=1, driver_sha='driver')


def test_execution_positions_are_recovered_from_the_sealed_schedule():
    manifest, arm_map, frame, binding, summary, batches = fixture()
    joined = analysis.join_records(manifest, arm_map, frame, binding, summary, batches)
    for row in joined:
        assert row['execution_position'] == ORDER.index(row['arm'])


def test_missing_rating_batch_is_rejected():
    manifest, arm_map, frame, binding, summary, batches = fixture()
    with pytest.raises(ValueError, match='batch count'):
        analysis.run_analysis(manifest, arm_map, frame, binding, summary, batches[:-1],
                              n_boot=200, seed=1, driver_sha='driver')


def test_rating_batch_loader_preserves_start_then_reader_order(tmp_path):
    frame_rows = [{'blind_id': f'id-{index:03d}'} for index in range(13)]
    summary = {'batch_sha256': []}
    batch_dir = tmp_path / 'batches'
    batch_dir.mkdir()
    for start in range(0, len(frame_rows), analysis.BATCH):
        block = frame_rows[start:start + analysis.BATCH]
        for reader in (0, 1):
            body = {'schema': 'eligible-immediate-rating-batch-v1', 'start': start,
                    'reader': reader,
                    'records': [{'blind_id': row['blind_id'], 'rating': True,
                                 'finish_reason': 'stop'} for row in block]}
            value = {**body, 'sha256': analysis.digest(body)}
            (batch_dir / f'target-{start:03d}-reader{reader}.json').write_text(
                json.dumps(value))
            summary['batch_sha256'].append(value['sha256'])
    batches = analysis._load_batches(tmp_path, frame_rows, summary)
    assert [(batch['start'], batch['reader']) for batch in batches] == [
        (0, 0), (0, 1), (12, 0), (12, 1)]


def test_analysis_body_is_json_serializable():
    json.dumps(run(fixture()))


def test_deactivation_manifest_uses_its_own_predeclared_contrasts():
    body = run(fixture(schema='v4'))
    assert body['kind'] == 'deactivation'
    row = next(item for item in body['primary']['result']['contrasts']
               if item['arm'] == 'target_bias_minus1' and item['reference'] == 'native' and
               item['metric'] == 'semantic_success')
    assert row['estimate'] == 1.0
