"""Real frozen fresh schemas plus resealed identity/provenance tamper cases."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_overnight_blind_frame_fresh_adapter_v1 as adapter
import build_overnight_blind_frame_v2 as frozen
import fresh_frame_recovery_v1 as recovery


def seal(value):
    value['sha256'] = recovery.base.digest({k: v for k, v in value.items() if k != 'sha256'})
    return value


class FreshRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = recovery.base.sealed(recovery.MANIFEST)
        cls.source = recovery.base.sealed(cls.manifest['source_frame_path'])
        cls.enrollment = recovery.base.sealed(cls.manifest['source_enrollment_path'])
        cls.selection = recovery.base.sealed(recovery.SELECTION)
        cls.units = recovery.base.sealed(recovery.UNITS)

    def setUp(self):
        self.m, self.s, self.e, self.q, self.u = map(copy.deepcopy,
             (self.manifest, self.source, self.enrollment, self.selection, self.units))
        self.uid = self.m['rows'][0]['uid']
        self.original = next(r for r in self.s['records'] if r['uid'] == self.uid)
        self.event = next(r for r in self.q['records'] if r['uid'] == self.uid)
        self.unit = next(r for r in self.u['records'] if r['attempt_id'] == self.event['attempt_id'] and
                         r['sentence_index'] == self.event['sentence_index'])

    def rebound(self):
        seal(self.u)
        self.q['units_sha256'] = self.u['sha256']
        seal(self.q)
        self.s.update(units_sha256=self.u['sha256'], selection_sha256=self.q['sha256'])
        seal(self.s)
        self.e.update(frame_sha256=self.s['sha256'], selection_sha256=self.q['sha256'])
        seal(self.e)
        self.m.update(source_frame_sha256=self.s['sha256'], source_enrollment_sha256=self.e['sha256'])
        seal(self.m)

    def check(self):
        return recovery.checked_contexts(self.m, self.s, self.e, self.q, self.u)

    def test_actual_source_schema_and_frozen_failure(self):
        self.assertEqual(set(self.original), {'uid', 'transition', 'reader_input', 'analysis_meta'})
        with self.assertRaises(KeyError):
            frozen.reader_context(self.original, self.m['rows'][0], self.s['schema'])
        derived, contexts, provenance = self.check()
        self.assertEqual(len(contexts), 134)
        self.assertEqual(derived['records'][0]['family'], self.m['rows'][0]['family'])
        self.assertEqual(provenance['source_frame_sha256'], self.s['sha256'])
        self.assertNotEqual(derived['sha256'], self.s['sha256'])
        frozen.reader_context(derived['records'][0], self.m['rows'][0], self.s['schema'])

    def test_unsealed_input_rejected(self):
        self.original['reader_input']['problem'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'seal differs'):
            self.check()

    def test_resealed_family_identity_rejected(self):
        self.event['family'] = '0' * 64
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.check()

    def test_resealed_enrollment_omission_rejected(self):
        self.e['rows'] = self.e['rows'][1:]
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'UID set differs'):
            self.check()

    def test_resealed_manifest_token_change_rejected(self):
        self.m['rows'][0]['prefix_ids'][0] += 1
        seal(self.m)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.check()

    def test_resealed_shared_token_hash_change_rejected(self):
        native = self.m['rows'][0]
        enrolled = next(r for r in self.e['rows'] if r['uid'] == self.uid)
        for row in (native, enrolled):
            row['prefix_ids'][0] += 1
            row['prefix_ids_sha256'] = recovery.base.digest(row['prefix_ids'])
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'token binding differs'):
            self.check()

    def test_resealed_trace_digest_rejected(self):
        self.original['analysis_meta']['trace_sha256'] = 'a' * 64
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.check()

    def test_resealed_tokenizer_digest_rejected(self):
        self.original['analysis_meta']['tokenizer_sha256'] = 'b' * 64
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.check()

    def test_resealed_problem_change_rejected(self):
        self.original['reader_input']['problem'] += 'new problem'
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.check()

    def test_resealed_prefix_text_and_hash_change_rejected(self):
        data = self.original['reader_input']
        data['emitted_prefix'] = 'X' + data['emitted_prefix'][1:]
        self.original['analysis_meta']['prefix_text_sha256'] = hashlib.sha256(data['emitted_prefix'].encode()).hexdigest()
        self.rebound()
        # The selected unit establishes the trigger and problem, while the
        # original frozen frame digest pins the earlier text through amendment.
        recovery.validate_amendment(recovery.base.sealed(recovery.AMENDMENT), self.manifest)
        with self.assertRaisesRegex(ValueError, 'amendment/source/code binding'):
            recovery.validate_amendment(recovery.base.sealed(recovery.AMENDMENT), self.m)

    def test_selected_sentence_index_change_rejected(self):
        self.event['sentence_index'] += 1
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.check()

    def test_resealed_unit_trigger_change_rejected(self):
        self.unit['inputs']['sentence'] += '?'
        self.rebound()
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.check()

    def test_adapter_binding_and_all_assignment_ids(self):
        class Tokenizer:
            @staticmethod
            def decode(ids, **kwargs):
                return 'Check.'
        amendment = recovery.base.sealed(recovery.AMENDMENT)
        price = recovery.base.sealed(recovery.PRICE)
        plan = frozen.expected_assignments(self.m)
        outputs = [{**row, 'tokens': [1], 'error': False, 'routed_present': True,
                    'finish': 'stop', 'stop_reason': None} for row in plan]
        stage = {'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory(prefix='.fresh-schema-test-', dir=recovery.REPO) as name:
            with patch.object(frozen, 'complete_outputs', return_value=(stage, plan, outputs, [])):
                amap, frame = adapter.build(self.m, price, self.s, self.e, self.q, self.u,
                                             Path(name) / 'generation', Path(name) / 'measurement', Tokenizer(),
                                             amendment, salt=b'x' * 32)
            self.assertEqual([r['uid'] for r in amap['records']], [r['uid'] for r in plan])
            self.assertEqual(len(frame['records']), 3480)
            self.assertEqual(frame['builder_sha256'], recovery.base.file_sha(frozen.__file__))
            self.assertEqual(frame['recovery_provenance']['context_adapter_sha256'],
                             recovery.base.file_sha(adapter.__file__))
            self.assertEqual(amap['recovery_provenance'], frame['recovery_provenance'])
            self.assertTrue(all(set(r) == {'blind_id', 'reader_input'} for r in frame['records']))
            self.assertTrue(all(set(r['reader_input']) == set(frozen.ALLOWLIST) for r in frame['records']))

    def test_tampered_operational_code_hash_rejected(self):
        amendment = copy.deepcopy(recovery.base.sealed(recovery.AMENDMENT))
        amendment['operational_code_files'][str(Path(adapter.__file__).resolve())] = '0' * 64
        seal(amendment)
        with self.assertRaisesRegex(ValueError, 'amendment/source/code binding'):
            recovery.validate_amendment(amendment, self.m)


if __name__ == '__main__':
    unittest.main()
