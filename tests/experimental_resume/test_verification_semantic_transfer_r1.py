"""Exact zero-runtime cancellation proofs; no scheduler mutations."""
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import verification_study_v3_semantic_transfer_r1 as transfer
from moe_exp.verification_study.contracts import save, load, file_sha


class TransferAccountingTests(unittest.TestCase):
    def setUp(self):
        self.card = {'sha256': 'a' * 64}
        self.row = {'job_id': '123', 'state': 'CANCELLED', 'exit_code': '0:0',
                    'elapsed_seconds': 0, 'allocated_cpus': 0, 'start': 'None',
                    'account': 'iscrc_miosr', 'partition': 'lrd_all_serial',
                    'comment': 'cv3auto:' + self.card['sha256']}

    def test_raw_none_start_is_explicitly_retained(self):
        transfer.validate_cancelled(self.row, '123', self.card)
        self.assertEqual(self.row['start'], 'None')

    def test_raw_unknown_start_is_accepted_without_normalizing(self):
        self.row['start'] = 'Unknown'
        transfer.validate_cancelled(self.row, '123', self.card)
        self.assertEqual(self.row['start'], 'Unknown')

    def test_actual_start_is_never_accepted(self):
        self.row['start'] = '2026-10-08T01:00:00'
        with self.assertRaises(ValueError):
            transfer.validate_cancelled(self.row, '123', self.card)

    def test_any_runtime_rejects(self):
        self.row['elapsed_seconds'] = 1
        with self.assertRaises(ValueError):
            transfer.validate_cancelled(self.row, '123', self.card)

    def test_any_allocated_cpu_rejects(self):
        self.row['allocated_cpus'] = 4
        with self.assertRaises(ValueError):
            transfer.validate_cancelled(self.row, '123', self.card)

    def test_foreign_job_or_comment_rejects(self):
        for field, value in (('job_id', '124'), ('comment', 'foreign'),
                             ('account', 'other'), ('partition', 'boost_usr_prod')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                transfer.validate_cancelled({**self.row, field: value}, '123', self.card)

    def test_failure_cannot_be_laundered_as_unstarted_cancel(self):
        for field, value in (('state', 'FAILED'), ('exit_code', '1:0')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                transfer.validate_cancelled({**self.row, field: value}, '123', self.card)

    def test_accounting_keeps_raw_none_and_full_card_comment(self):
        text = '123|CANCELLED by 133943|0:0|0|None|2026-10-08T02:00:00|0|iscrc_miosr|lrd_all_serial|' + self.row['comment'] + '|'
        with patch.object(transfer.driver, 'command', return_value=text):
            values = transfer.cancellation_accounting(('123',))
        self.assertEqual(values['123']['state'], 'CANCELLED')
        self.assertEqual(values['123']['start'], 'None')
        transfer.validate_cancelled(values['123'], '123', self.card)

    def test_duplicate_accounting_rejects(self):
        line = '123|CANCELLED|0:0|0|None|None|0|iscrc_miosr|lrd_all_serial|c|'
        with patch.object(transfer.driver, 'command', return_value=line + '\n' + line), self.assertRaisesRegex(ValueError, 'ambiguous'):
            transfer.cancellation_accounting(('123',))


class TransferResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cv3-format-transfer-test-',
            dir='/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')
        self.root = Path(self.tmp.name); self.cp = self.root / 'report/CAMPAIGN.json'
        source = self.root / 'fixture.py'; source.write_text('# bounded fixture\n')
        files = {str(source): file_sha(source)}
        self.campaign = save(self.cp, {'schema': 'fixture', 'code_files': files,
                                      'result_root': str(self.root)})
        self.binding = save(self.cp.parent / 'SEMANTIC_FORMAT_BINDING-r1.json', {'schema': 'fixture'})
        save(self.cp.parent / 'AUTOMATION_READINESS-r5.json', {'schema': 'fixture', 'status': 'PASS',
            'campaign_sha256': self.campaign['sha256'], 'code_files': files})
        self.jobs = {}; self.operations = []; self.crash = None; self.crashed = False
        targets = []
        for job in transfer.TARGETS:
            slot = self.root / 'old' / job
            card = save(slot / 'CARD.json', {'schema': 'fixture', 'job': job})
            sub = save(slot / 'SUBMISSION.json', {'schema': 'fixture', 'job_id': job})
            self.jobs[job] = {'card': card, 'held': True, 'state': 'PENDING', 'dependency': '(null)'}
            targets.append({'job_id': job, 'slot': str(slot), 'controller_card_sha256': card['sha256'],
                            'submission_sha256': sub['sha256']})
        save(self.cp.parent / 'SEMANTIC_FORMAT_HOLD-r1.json', {'schema': 'fixture',
            'campaign_sha256': self.campaign['sha256'], 'targets': targets})
        child = None
        for counter, job in ((2, '993'), (1, '992'), (0, '991')):
            slot = self.root / 'new' / job
            parent = '992' if counter == 2 else '991' if counter == 1 else None
            ancestors = ['992', '991'] if counter == 2 else ['991'] if counter else []
            card = save(slot / 'CARD.json', {'schema': 'fixture', 'job': job,
                'consecutive_rescues': counter, 'parent_job_id': parent, 'failure_ancestor_job_ids': ancestors})
            sub = save(slot / 'SUBMISSION.json', {'schema': 'fixture', 'job_id': job})
            proof = save(slot / 'PREARM.json', {'schema': 'fixture', 'controller_card_sha256': card['sha256'],
                'primary_job_id': job, 'submission_sha256': sub['sha256'], 'children': [child] if child else []})
            if counter:
                release = save(slot / 'RELEASE.json', {'schema': 'fixture', 'job_id': job,
                    'controller_card_sha256': card['sha256'], 'submission_sha256': sub['sha256']})
                child = {'controller_card_sha256': card['sha256'], 'submission_path': str(slot / 'SUBMISSION.json'),
                    'submission_sha256': sub['sha256'], 'job_id': job, 'prearm_sha256': proof['sha256'],
                    'release_sha256': release['sha256']}
            else:
                self.card = card; self.sub = sub; self.slot = slot
            dependency = 'afternotok:' + ':'.join(ancestors) if counter else '(null)'
            self.jobs[job] = {'card': card, 'held': not counter, 'state': 'PENDING', 'dependency': dependency}
        self.patches = [patch.object(transfer, 'CAMPAIGN', self.cp),
            patch.object(transfer.fmt, 'load_binding', return_value=self.binding),
            patch.object(transfer.old, 'validate_card'), patch.object(transfer.driver, 'validate_card'),
            patch.object(transfer.driver, 'controller_card', return_value=self.card),
            patch.object(transfer.driver, 'submit_controller', return_value=self.sub),
            patch.object(transfer.driver, 'slot_for', side_effect=lambda c: self.root / 'new' / c['job']),
            patch.object(transfer.driver, 'orchestration_lock', side_effect=lambda root: nullcontext()),
            patch.object(transfer.driver, 'command', side_effect=self.command),
            patch.object(transfer.driver, 'release_controller', side_effect=self.release)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def command(self, args, **kwargs):
        if args[:3] == ['scontrol', 'show', 'job']:
            job = args[3]; value = self.jobs[job]
            reason = 'JobHeldUser' if value['held'] else 'Dependency' if value['dependency'] != '(null)' else 'None'
            return (f"JobId={job} UserId=lmolfett(133943) Account=iscrc_miosr Partition=lrd_all_serial "
                    f"Comment=cv3auto:{value['card']['sha256']} JobState={value['state']} Reason={reason} "
                    f"StartTime=Unknown RunTime=00:00:00 AllocTRES=(null) Dependency={value['dependency']}")
        if args[0] == 'sacct':
            rows = []
            for job in args[args.index('-j') + 1].split(','):
                value = self.jobs[job]; start = 'None' if value['state'] == 'CANCELLED' else 'Unknown'
                rows.append(f"{job}|{value['state']}|0:0|0|{start}|None|0|iscrc_miosr|lrd_all_serial|cv3auto:{value['card']['sha256']}|")
            return '\n'.join(rows)
        if args[0] == 'scancel':
            job = args[1]; self.jobs[job]['state'] = 'CANCELLED'; self.operations.append(('cancel', job))
            if self.crash == 'after-first-cancel' and not self.crashed:
                self.crashed = True; raise RuntimeError('after first cancel')
            return ''
        raise AssertionError('unexpected fake command: ' + str(args))

    def release(self, card):
        result_path = self.slot / 'RELEASE.json'
        if result_path.exists():
            return load(result_path)
        if self.crash == 'after-health-save' and not self.crashed:
            self.crashed = True; raise RuntimeError('after health save')
        save(self.slot / 'RELEASE_INTENT.json', {'schema': 'fixture', 'card_sha256': card['sha256']})
        if self.crash == 'after-release-intent' and not self.crashed:
            self.crashed = True; raise RuntimeError('after release intent')
        if self.jobs['991']['held']:
            self.jobs['991']['held'] = False; self.jobs['991']['state'] = 'RUNNING'
            self.operations.append(('release', '991'))
        if self.crash == 'after-scheduler-release' and not self.crashed:
            self.crashed = True; raise RuntimeError('after scheduler release')
        return save(result_path, {'schema': 'fixture', 'job_id': '991',
            'controller_card_sha256': card['sha256'], 'submission_sha256': self.sub['sha256']})

    def test_exact_transfer_and_repeat_preserve_one_release(self):
        first = transfer.transfer(); operations = list(self.operations)
        second = transfer.transfer()
        self.assertEqual(first, second); self.assertEqual(self.operations, operations)
        self.assertEqual(operations, [('cancel', '59662142'), ('cancel', '59662139'),
                                      ('cancel', '59662113'), ('release', '991')])
        self.assertEqual(first['experimental_cancellations'], [])

    def test_dead_second_rescue_blocks_before_any_cancellation(self):
        self.jobs['993']['state'] = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'rescue must be live'):
            transfer.transfer()
        self.assertEqual(self.operations, [])
        self.assertTrue(self.jobs['991']['held'])

    def test_wrong_second_rescue_dependency_blocks(self):
        self.jobs['993']['dependency'] = 'afternotok:991'
        with self.assertRaisesRegex(ValueError, 'ancestors differ'):
            transfer.transfer()
        self.assertEqual(self.operations, [])

    def test_partial_cancellation_resumes_without_duplicate_cancel(self):
        self.crash = 'after-first-cancel'
        with self.assertRaisesRegex(RuntimeError, 'first cancel'):
            transfer.transfer()
        result = transfer.transfer()
        self.assertEqual(sum(x == ('cancel', '59662142') for x in self.operations), 1)
        self.assertEqual(sum(x[0] == 'cancel' for x in self.operations), 3)
        self.assertEqual(result['replacement_job_id'], '991')

    def test_committed_health_proof_resumes_before_release(self):
        self.crash = 'after-health-save'
        with self.assertRaisesRegex(RuntimeError, 'health save'):
            transfer.transfer()
        proof_path = self.cp.parent / 'HEALTH_RECONCILIATION-semantic-r5.json'
        before = file_sha(proof_path); result = transfer.transfer()
        self.assertEqual(before, file_sha(proof_path))
        self.assertEqual(sum(x[0] == 'cancel' for x in self.operations), 3)
        self.assertEqual(result['additional_reader_draws'], 0)

    def test_scheduler_release_without_receipt_resumes_release_intent(self):
        self.crash = 'after-scheduler-release'
        with self.assertRaisesRegex(RuntimeError, 'scheduler release'):
            transfer.transfer()
        result = transfer.transfer()
        self.assertEqual(sum(x == ('release', '991') for x in self.operations), 1)
        self.assertEqual(result['replacement_job_id'], '991')

    def test_held_release_intent_rechecks_second_rescue(self):
        self.crash = 'after-release-intent'
        with self.assertRaisesRegex(RuntimeError, 'release intent'):
            transfer.transfer()
        self.jobs['993']['state'] = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'rescue must be live'):
            transfer.transfer()
        self.assertTrue(self.jobs['991']['held'])
        self.assertFalse(any(x[0] == 'release' for x in self.operations))

    def test_held_release_intent_with_live_rescues_resumes(self):
        self.crash = 'after-release-intent'
        with self.assertRaisesRegex(RuntimeError, 'release intent'):
            transfer.transfer()
        result = transfer.transfer()
        self.assertEqual(sum(x == ('release', '991') for x in self.operations), 1)
        self.assertEqual(result['replacement_job_id'], '991')

    def test_release_receipt_before_closeout_resumes(self):
        original = transfer.save
        def interrupted(path, value, **kwargs):
            if Path(path).name == 'SEMANTIC_FORMAT_TRANSFER-r1.json':
                raise RuntimeError('before closeout')
            return original(path, value, **kwargs)
        with patch.object(transfer, 'save', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'closeout'):
            transfer.transfer()
        result = transfer.transfer()
        self.assertEqual(sum(x == ('release', '991') for x in self.operations), 1)
        self.assertEqual(result['replacement_job_id'], '991')

    def test_malformed_accounting_rejects(self):
        with patch.object(transfer.driver, 'command', return_value='123|CANCELLED|'), self.assertRaisesRegex(ValueError, 'ambiguous'):
            transfer.cancellation_accounting(('123',))


if __name__ == '__main__':
    unittest.main()
