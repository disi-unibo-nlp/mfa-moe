"""CPU-only checks for the separate prospective 220-family extension."""
import copy
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import freeze_mechanism_extension_220_v1 as freeze  # noqa: E402
import mechanism_extension_strict_veto_v1 as veto  # noqa: E402
import prepare_mechanism_extension_220_v1 as extension  # noqa: E402


class MechanismExtensionTests(unittest.TestCase):
    def test_cpu_stage_write_is_sealed_idempotent_and_collision_safe(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as folder:
            path = Path(folder) / 'STAGE.json'
            stale = path.with_name(path.name + '.partial-stale')
            stale.write_text('interrupted prior write')
            value = extension.write_once(path, {'schema': 'synthetic-stage', 'families': 220})
            self.assertEqual(freeze.sealed(path), value)
            self.assertEqual(extension.write_once(
                path, {'schema': 'synthetic-stage', 'families': 220}), value)
            with self.assertRaisesRegex(ValueError, 'existing extension artifact differs'):
                extension.write_once(path, {'schema': 'synthetic-stage', 'families': 219})
            self.assertEqual(freeze.sealed(path), value)
            self.assertEqual(stale.read_text(), 'interrupted prior write')

    def test_cpu_step_is_slurm_and_partition_bound_without_hostname_gate(self):
        env = {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0',
               'SLURM_JOB_PARTITION': 'lrd_all_serial'}
        with patch.dict(os.environ, env, clear=True):
            extension.require_cpu_step()
        with patch.dict(os.environ, {**env, 'SLURM_JOB_PARTITION': 'other'},
                        clear=True):
            with self.assertRaisesRegex(RuntimeError, 'lrd_all_serial'):
                extension.require_cpu_step()
        with patch.dict(os.environ, {k: v for k, v in env.items()
                                      if k != 'SLURM_STEP_ID'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'CPU Slurm step'):
                extension.require_cpu_step()

    def test_native_units_and_detector_cover_every_frozen_family(self):
        family = freeze.sealed(freeze.OUT)

        class TraceRecord:
            def __init__(self, **value):
                self.metadata = value['metadata']
                self.cot_text = value['cot_text']
                self.generation_messages = [{'role': 'user', 'content': 'problem'}]
                self.prompt = 'problem'

        schema = types.ModuleType('moe_exp.schemas')
        schema.TraceRecord = TraceRecord
        spans = types.ModuleType('moe_exp.correlation_pipeline.spans')
        spans.trace_digest = lambda trace: 'trace-sha'
        spans.reasoning_ranges = lambda trace: [(0, 10)]
        classes = types.ModuleType('moe_exp.correlation_pipeline.dynamics.classes')
        classes.saved_layout = lambda trace: {
            'units': [{'index': 0, 'start': 0, 'end': 5, 'text': 'abcde'},
                      {'index': 1, 'start': 5, 'end': 10, 'text': 'fghij'}],
            'unit_tokens': [[0], [1]]}
        annotation = types.ModuleType('moe_exp.correlation_pipeline.dynamics.annotation_plan')
        annotation.contiguous_blocks = lambda units, width: units
        transitions = sys.modules['moe_exp.routing_control.transitions_v2']

        class Detector:
            def observe(self, row):
                return {'events': [types.SimpleNamespace(
                    transition='candidate_to_verify', evidence_end=5)]}

        rows = [{'question': family['representative_questions'][member],
                 'attempt_id': f'attempt-{i}', 'source_location': str(i),
                 'trace_sha256': 'trace-sha'}
                for i, member in enumerate(family['families'])]
        trace = {'metadata': {'token_replay': {
            'completion_token_ids': [10, 11],
            'completion_offsets': [[0, 5], [5, 10]],
            'tokenizer_sha256': 'tokenizer'}}, 'cot_text': 'abcdefghij'}
        with patch.dict(sys.modules, {
                'moe_exp.schemas': schema,
                'moe_exp.correlation_pipeline.spans': spans,
                'moe_exp.correlation_pipeline.dynamics.classes': classes,
                'moe_exp.correlation_pipeline.dynamics.annotation_plan': annotation}), \
             patch.object(transitions, 'StreamingTransitionDetectorV2', Detector):
            units = extension.build_units(rows, family, trace_reader=lambda _: trace)
            self.assertEqual(units['attempts'], 220)
            self.assertEqual(units['sentences'], 440)
            units['sha256'] = 'synthetic-units'
            audit = extension.audit_detector(
                units, family, {row['attempt_id']: row for row in rows},
                trace_reader=lambda _: trace)
        self.assertEqual(audit['contiguous_pairs'], 220)
        self.assertEqual(audit['fire_family_counts']['candidate_to_verify'], 220)
        self.assertEqual(len(audit['eligible_events']['candidate_to_verify']), 220)

    def test_exact_unused_confirm_disjoint_suffix_is_frozen(self):
        source, saved = freeze.sealed(freeze.SOURCE), freeze.sealed(freeze.OUT)
        self.assertEqual(saved, {**freeze.derive(source),
                                 'sha256': freeze.digest(freeze.derive(source))})
        pools = source['new_parent_pools']['parent_pools']
        used = set().union(*(set(pools[key]) for key in freeze.POOLS))
        self.assertEqual(len(saved['families']), 220)
        self.assertFalse(set(saved['families']) & used)
        self.assertFalse(set(saved['families']) & set(
            source['new_parent_pools']['confirm_connected_excluded']))
        changed = copy.deepcopy(source)
        changed['new_parent_pools']['confirm_connected_excluded'].pop()
        with self.assertRaisesRegex(ValueError, 'confirm-connected'):
            freeze.derive(changed)

    def test_three_hash_ordered_starts_and_no_replacement(self):
        family = freeze.sealed(freeze.OUT)
        records = []
        for i, member in enumerate(family['families']):
            for index in range(4 if i == 0 else 1):
                records.append({'family': member, 'attempt_id': f'attempt{i}',
                                'sentence_index': index, 'segment': 0,
                                'token_end': 100 + index,
                                'inputs': {'problem_statement': 'problem'}})
        units = {'schema': 'dense-mechanism-extension-units-v1',
                 'extension_family_freeze_sha256': family['sha256'],
                 'sha256': 'units', 'records': records}
        events = [{'family': family['families'][0], 'attempt_id': 'attempt0',
                   'sentence_index': i, 'segment': 0, 'prefix_tokens': 100 + i,
                   'prefix_end_char': 1000 + i, 'event_end_char': 990 + i,
                   'tokenizer_sha256': 'tokenizer'} for i in range(4)]
        audit = {'schema': 'mechanism-extension-detector-audit-v1',
                 'sha256': 'audit', 'units_sha256': 'units', 'families': 220,
                 'extension_family_freeze_sha256': family['sha256'],
                 'detector_sha256': extension.file_sha(
                     REPO / 'src/moe_exp/routing_control/transitions_v2.py'),
                 'eligible_events': {'candidate_to_verify': events,
                                     'approach_to_commit': events[:1]}}
        selected = extension.select_events(units, audit, family)
        self.assertEqual(selected['selected_counts'],
                         {'candidate_to_verify': 3, 'approach_to_commit': 1})
        self.assertEqual(len(selected['records']), 4)
        altered = copy.deepcopy(audit)
        altered['eligible_events']['candidate_to_verify'].reverse()
        self.assertEqual(extension.select_events(units, altered, family)['records'],
                         selected['records'])
        changed = copy.deepcopy(audit)
        changed['eligible_events']['candidate_to_verify'][0]['prefix_tokens'] = 999
        with self.assertRaisesRegex(ValueError, 'detector fire differs'):
            extension.select_events(units, changed, family)

    def test_reader_price_includes_primary_and_strict_sensitivity(self):
        family = freeze.sealed(freeze.OUT)
        frame = {'schema': 'mechanism-extension-start-frame-v1',
                 'sha256': 'frame', 'selection_sha256': 'selection',
                 'extension_family_freeze_sha256': family['sha256'], 'rows': 2}
        selected = {'sha256': 'selection', 'records': [{}, {}]}
        prior = freeze.sealed(extension.PRIOR_PRICE)
        price = extension.price_reader(frame, selected, family, prior,
                                       [1000, 2000], [1100, 2100], 3600)
        self.assertEqual(price['primary_start_ratings'], 4)
        self.assertEqual(price['strict_veto_sensitivity_ratings'], 4)
        self.assertEqual(price['ratings'], 8)
        self.assertEqual(price['prompt_tokens_all_ratings_exact'], 12400)
        self.assertEqual(price['max_decode_tokens'], 8 * 1024)
        self.assertTrue(price['status'].startswith('HOLD_'))
        with self.assertRaisesRegex(ValueError, 'exact frame'):
            extension.price_reader(frame, selected, family, prior,
                                   [49000, 2000], [1100, 2100], 3600)

    def test_veto_is_blind_and_sensitivity_only(self):
        row = {'transition': 'candidate_to_verify', 'family': 'hidden-family',
               'arm': 'hidden-arm',
               'reader_input': {'problem': 'Factor 21.',
                                'emitted_prefix': 'We try 3 and 7. Check it.',
                                'triggering_sentence': 'Check it.'}}
        messages = veto.messages(row)
        visible = str(messages)
        self.assertNotIn('hidden-family', visible)
        self.assertNotIn('hidden-arm', visible)
        self.assertIn('Earlier work on a different', visible)
        self.assertEqual(veto.parse_veto('{"already_completed": true}'), True)
        self.assertEqual(veto.parse_veto('<think>private reasoning</think>\n{"already_completed": false}'), False)
        self.assertIsNone(veto.parse_veto('{"already_completed": "true"}'))
        primary = [{'rating': {'start': True}, 'finish_reason': 'stop'}] * 2
        clear = [{'already_completed': False, 'finish_reason': 'stop'}] * 2
        self.assertTrue(veto.strict_valid(primary, clear))
        self.assertFalse(veto.strict_valid(primary, [
            clear[0], {'already_completed': True, 'finish_reason': 'stop'}]))


if __name__ == '__main__':
    unittest.main()
