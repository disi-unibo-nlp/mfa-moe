"""Installed CPU boundaries, report recovery and protected operational ownership."""
import ast
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r6.contracts import STAGING, file_sha, load, publish
from moe_exp.research_campaign_v3_r6_ops_r1 import policy, milestone


class Operations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r6-ops-contract-', dir=STAGING)
        self.root = Path(self.temp.name)
        self.campaign = {'root': str(self.root), 'sha256': 'scientific-source-seal',
                         'code_files': {}, 'source_files': {}, 'results_first_phases': True}
        self.binding = publish(self.root / policy.BINDING, {'campaign_sha256': self.campaign['sha256'],
            'files': {}, 'runtime_files': {}, 'remote_writes_allowed': False, 'manuscript_writes_allowed': False})

    def tearDown(self):
        self.temp.cleanup()

    def report(self, *, phase='final', cells=243747):
        directory = self.root / ('output' if phase == 'final' else 'deliveries/' + phase)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'report.md').write_text('Explicit incomplete science\n')
        manifest = publish(directory / 'EXPORT_FILES.json', {'files': {
            str(directory / 'report.md'): file_sha(directory / 'report.md')}})
        report = publish(directory / 'REPORT.json', {'SCIENCE_COMPLETE': False, 'cells': cells})
        return directory, report, manifest

    def proofs(self):
        reader = publish(self.root / 'READER_INPUT_VALIDATION.json', {
            'status': 'complete', 'original_failures_reproduced': 64, 'scientific_draws': 0})
        export = publish(self.root / 'FULL_EXPORT_VALIDATION.json', {'status': 'complete', 'cells': 243747})
        graph = publish(self.root / 'IMPORT_GRAPH_VALIDATION.json', {'status': 'complete',
            'reader_validation_sha256': reader['sha256'], 'full_export_validation_sha256': export['sha256'],
            'non_reap_generations': 69459, 'compact_shapes': 346})
        ready = publish(self.root / 'READINESS.json', {'passed': True, 'files': {},
            'source_unchanged_during_tests': True, 'actual_import_graph_validated': True,
            'actual_validation_sha256': graph['sha256']})
        publish(self.root / 'OPS_R1_VALIDATION_COMPLETED.json', {'readiness_sha256': ready['sha256'],
            'binding_sha256': self.binding['sha256'], 'manuscript_unchanged': True})

    def hooks(self, *, original_ready=None, original_control=None, original_launch=None):
        from moe_exp.research_campaign_v3_r6 import annotation, controller, export, publication, scheduler
        stack = ExitStack()
        for module, names in ((annotation, ('annotate',)), (controller, ('launch', 'control', 'ready', 'card')),
                              (export, ('deliver',)), (publication, ('deliver', 'build_local', 'push', 'push_snapshot')),
                              (scheduler, ('card',)), (scheduler.Slurm, ('command', 'verify'))):
            for name in names:
                stack.enter_context(patch.object(module, name, getattr(module, name)))
        stack.enter_context(patch.object(scheduler.Slurm, '_ops_r1_installed', False, create=True))
        stack.enter_context(patch.object(scheduler.Slurm, '_ops_r1_binding', None, create=True))
        if original_ready:
            controller.ready = original_ready
        if original_control:
            controller.control = original_control
        if original_launch:
            controller.launch = original_launch
        return stack

    def test_missing_real_readiness_blocks_launch(self):
        from moe_exp.research_campaign_v3_r6 import controller
        with self.hooks(original_launch=lambda *a: self.fail('Submitted without validation')):
            policy.install(self.campaign)
            with self.assertRaises(FileNotFoundError):
                controller.launch(self.campaign)

    def test_complete_proof_chain_is_required_for_revision_six(self):
        from moe_exp.research_campaign_v3_r6 import controller
        self.campaign['continuation_revision'] = 6
        self.proofs()
        with self.hooks(original_launch=lambda *a: {'validated': True}):
            policy.install(self.campaign)
            self.assertTrue(controller.launch(self.campaign)['validated'])
            (self.root / 'READER_INPUT_VALIDATION.json').unlink()
            with self.assertRaises(FileNotFoundError):
                controller.launch(self.campaign)

    def test_operations_completion_cannot_be_substituted_by_old_readiness(self):
        self.proofs()
        (self.root / 'OPS_R1_VALIDATION_COMPLETED.json').unlink()
        with self.assertRaises(FileNotFoundError):
            policy.require_readiness(self.campaign)

    def test_changed_operation_file_is_rejected(self):
        source = self.root / 'adapter.py'; source.write_text('bound\n')
        (self.root / policy.BINDING).unlink()
        publish(self.root / policy.BINDING, {**self.binding, 'files': {str(source): file_sha(source)}})
        source.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'bound source changed'):
            policy.check_binding(self.campaign)

    def test_final_report_is_idempotent_without_paper_completion(self):
        from moe_exp.research_campaign_v3_r6 import decisions, publication
        directory, report, _ = self.report()
        with patch.object(decisions, 'paper_exhibits', side_effect=AssertionError('Manuscript write')), \
             patch.object(publication, 'build_local', side_effect=AssertionError('Manuscript build')):
            result = policy.deliver(self.campaign, directory, report, final=True)
            self.assertEqual(policy.deliver(self.campaign, directory, report, final=True), result)
        self.assertTrue(policy.verified_local_report(self.campaign)['complete'])
        self.assertFalse(load(self.root / 'SCIENCE_COMPLETE.json')['complete'])
        self.assertFalse((self.root / 'LOCAL_PAPER_COMPLETE.json').exists())
        self.assertFalse((directory / 'PAPER_DELIVERY.json').exists())

    def test_interrupted_final_delivery_recovers_from_durable_report_marker(self):
        directory, report, _ = self.report()
        real_publish = policy.publish
        def interrupted(path, value):
            if Path(path).name == 'LOCAL_REPORT_COMPLETE.json':
                raise RuntimeError('Interrupted local report commit')
            return real_publish(path, value)
        with patch.object(policy, 'publish', interrupted):
            with self.assertRaises(RuntimeError):
                policy.deliver(self.campaign, directory, report, final=True)
        self.assertTrue((self.root / 'REPORT_COMPLETE.json').exists())
        self.assertFalse((self.root / 'LOCAL_REPORT_COMPLETE.json').exists())
        policy.deliver(self.campaign, directory, report, final=True)
        self.assertTrue(policy.verified_local_report(self.campaign))

    def test_phase_completion_binds_local_report_and_keeps_science_explicit(self):
        directory, report, _ = self.report(phase='memory')
        result = policy.deliver(self.campaign, directory, report, phase='memory')
        phase = load(self.root / 'PHASE_COMPLETE-memory.json')
        self.assertEqual(phase['local_report_sha256'], load(directory / 'LOCAL_REPORT.json')['sha256'])
        self.assertFalse(result['REPORT_COMPLETE'])
        self.assertFalse(phase['scientific_cells_complete'])
        self.assertFalse((self.root / 'REPORT_COMPLETE.json').exists())

    def test_changed_export_is_rejected_on_reentry(self):
        directory, report, _ = self.report()
        policy.deliver(self.campaign, directory, report, final=True)
        (directory / 'report.md').write_text('changed')
        with self.assertRaises(ValueError):
            policy.verified_local_report(self.campaign)

    def test_controller_finishes_on_real_report_without_manuscript_livelock(self):
        from moe_exp.research_campaign_v3_r6 import controller
        self.proofs(); directory, report, _ = self.report()
        policy.deliver(self.campaign, directory, report, final=True)
        seen = []
        with self.hooks(original_control=lambda campaign, card, scheduler: seen.append(campaign) or {'status': 'complete'}):
            policy.install(self.campaign)
            self.assertEqual(controller.control(self.campaign, {})['status'], 'complete')
            self.assertEqual(controller.ready(self.campaign, None, []), [])
        self.assertFalse(seen[0]['results_first_phases'])
        self.assertFalse(seen[0]['independent_publication'])
        self.assertTrue(self.campaign['results_first_phases'])

    def test_unfinished_phases_keep_order_and_do_not_dispatch_publication(self):
        from moe_exp.research_campaign_v3_r6 import controller, phases
        phase_function = phases.ready
        seen = []
        with self.hooks(original_ready=lambda campaign, scheduler, observed: seen.append(campaign) or []):
            policy.install(self.campaign)
            controller.ready(self.campaign, None, [])
        self.assertTrue(seen[0]['results_first_phases'])
        self.assertFalse(seen[0]['independent_publication'])
        self.assertIs(phases.ready, phase_function)

    def test_remote_publication_and_manuscript_build_are_denied(self):
        from moe_exp.research_campaign_v3_r6 import publication
        with self.hooks():
            policy.install(self.campaign)
            for name in ('deliver', 'build_local', 'push', 'push_snapshot'):
                with self.assertRaisesRegex(RuntimeError, 'deferred'):
                    getattr(publication, name)(self.campaign, {})

    def test_future_cards_and_worker_readback_bind_adapter(self):
        from moe_exp.research_campaign_v3_r6 import scheduler
        self.campaign['models'] = {}
        captured = []
        with self.hooks():
            scheduler.Slurm.command = lambda self, args: captured.append(args) or 'simulated'
            scheduler.Slurm.verify = lambda self, jid, card: {'fields': {'Command': str(policy.WRAPPER)}}
            policy.install(self.campaign)
            value = scheduler.card(self.campaign, 'controller')
            self.assertEqual(value['ops_binding_sha256'], self.binding['sha256'])
            old = str(policy.REPO / 'scripts/research_campaign_v3_r6/job.sbatch')
            scheduler.Slurm().command(['sbatch', old, 'card.json'])
            self.assertEqual(captured[-1][1], str(policy.WRAPPER))
            scheduler.Slurm().verify('1', value)

    def test_snapshot_is_reused_by_later_scientific_launch(self):
        from moe_exp.research_campaign_v3_r6 import scheduler
        with self.hooks():
            policy.install(self.campaign)
            first = milestone.marked(scheduler.card(self.campaign, 'snapshot', minutes=180, hold=True, begin=milestone.BEGIN))
            publish(self.root / 'cards' / (first['sha256'] + '.json'), first)
            publish(self.root / 'ops_r1/milestone/ACCEPTED.json', {'snapshot': {'card_sha256': first['sha256'], 'job_id': '1'}})
            repeated = scheduler.card(self.campaign, 'snapshot', minutes=180, hold=True, begin=milestone.BEGIN)
            self.assertEqual(first, repeated)

    def test_installed_external_namespace_cleanup_removes_pending_request(self):
        from moe_exp.research_campaign_v3_r6_ops_r1.preflight import installed_abort_check, installed_abort_method
        from moe_exp.research_campaign_v3_r6_ops_r1.reader import abort_external
        self.assertTrue(installed_abort_check()['external_namespace_removed_request'])
        method, _, _ = installed_abort_method()
        class Engine:
            def __init__(self):
                self.request_states = {'original-randomid': SimpleNamespace(parent_req=None, queue=None, lora_name=None)}
                self.external_req_ids = {'original': ['original-randomid']}
                self.parent_requests = {}
                self.lora_states = SimpleNamespace(request_finished=lambda *args: None)
            def _get_internal_request_ids(self, ids, internal):
                return ids if internal else [uid for i in ids for uid in self.external_req_ids.get(i, [])]
            def abort_request(self, ids, internal):
                return method(self, ids, internal=internal)
        engine = Engine()
        self.assertEqual(abort_external(engine, ['original']), ['original-randomid'])
        self.assertFalse(engine.request_states)

    def test_fallback_preserves_completed_report_references_without_science_claim(self):
        parent = self.root / 'parent'; self.campaign['resume_from'] = str(parent)
        publish(parent / 'output/REPORT.json', {'SCIENCE_COMPLETE': False, 'cells': 243747})
        publish(self.root / 'PARTIAL_VALIDATION_EVIDENCE.json', {'actual_seven_model_import': True})
        report = milestone.export_fallback(self.campaign)
        self.assertEqual(milestone.export_fallback(self.campaign), report)
        self.assertFalse(report['SCIENCE_COMPLETE'])
        self.assertFalse(report['manuscript_updated'])
        self.assertTrue(report['scientific_completion_not_inferred'])

    def test_protected_milestone_accepts_all_successors_before_zero_runtime_retirement(self):
        class Scheduler:
            def __init__(self):
                self.events, self.accepted = [], []
                self.states = {jid: 'PENDING' for jid in milestone.OLD_OWNERS}
            def snapshot(self):
                return {'jobs': []}
            def submit(self, value):
                jid = str(1000 + len(self.accepted))
                self.accepted.append(value); self.states[jid] = 'PENDING'
                self.events.append(('accept', jid))
                return {'job_id': jid}
            def verify(self, jid, value):
                return {'fields': {'Command': str(policy.WRAPPER), 'Reason': 'BeginTime'}}
            def observe(self, jid):
                state = self.states[jid]
                return {'job_id': jid, 'state': state, 'terminal': state == 'CANCELLED',
                        'elapsed_seconds': 0, 'start': 'Unknown', 'billing_hours': 0}
            def command(self, args):
                self.events.append(tuple(args))
                if args[0] == 'scancel':
                    if args[1] in milestone.OLD_OWNERS:
                        assert len(self.accepted) == 4
                    self.states[args[1]] = 'CANCELLED'
                return 'Reason=JobHeldUser'
            def release(self, jid):
                self.events.append(('release', jid))
        scheduler = Scheduler()
        publish(self.root / 'OPS_R1_PREFLIGHT.json', {'passed': True, 'binding_sha256': self.binding['sha256']})
        with self.hooks(), patch('moe_exp.research_campaign_v3_r6.controller._check', return_value={'admitted': True}):
            policy.install(self.campaign)
            result = milestone.protect(self.campaign, scheduler)
            self.assertEqual(milestone.protect(self.campaign, scheduler), result)
        self.assertEqual(len(scheduler.accepted), 4)
        self.assertTrue(all(value['ops_report_only'] for value in scheduler.accepted))
        self.assertEqual(sum(value['reservation_hours'] for value in scheduler.accepted), 28)
        self.assertFalse(result['scientific_work_released'])
        self.assertEqual(len(result['retirements']), 4)
        self.assertTrue(all(row['after']['elapsed_seconds'] == 0 for row in result['retirements']))


if __name__ == '__main__':
    unittest.main()
