import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import unittest
from moe_exp.study.common import WORK, Units, digest, publish, verified
from moe_exp.study.config import read_config
from moe_exp.study.journal import run_stage
from moe_exp.study.runtime import recover_outputs
from moe_exp.study.scheduler import Slurm, card
from moe_exp.study.workflow import initial_graph, pair, queue_finalization, parallel_lanes
from moe_exp.study.reservations import charged_hours, reserve_many, ledger

SNAPSHOT = {'consumed_hours': 25950., 'remaining_hours': 42050., 'outstanding_hours': 0.,
            'free_bytes': {'work': 100 * 1024**3, 'fast': 100 * 1024**3}}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=WORK / 'tools/tmp')
        self.root = Path(self.temp.name)
        self.cfg = read_config('configs/final_routing.yaml')
        self.cfg.update(root=str(self.root), control_root=str(self.root / 'study/test'), source_commit='test')
        self.binding = {'root': str(self.root), 'control_root': self.cfg['control_root'],
                        'source_commit': 'test', 'config': self.cfg}

    def tearDown(self):
        self.temp.cleanup()

    def test_output_without_capture_is_unknown_and_never_redrawn(self):
        units = Units(self.root / 'units'); row = {'uid': digest('a'), 'seed': 12}
        units.intent(row, '1'); units.admitted(row, '1', 'a-random')
        raw = units.output(row, {'token_ids': [1, 2]})
        recover_outputs(units, [row])
        self.assertEqual(units.state(digest('a'))['status'], 'unknown')
        self.assertFalse(units.intent(row, '2'))
        self.assertEqual(verified(units.folder(digest('a')) / 'OUTPUT.json'), raw)

    def test_committed_raw_and_capture_recover_without_engine(self):
        units = Units(self.root / 'units'); row = {'uid': digest('a')}
        units.intent(row, '1'); raw = units.output(row, {'token_ids': [1]})
        publish(units.folder(digest('a')) / 'CAPTURE.json', {'capture': {}, 'uid': row['uid'], 'raw_sha256': raw['sha256']})
        recover_outputs(units, [row]); result = units.state(digest('a'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['raw_sha256'], raw['sha256'])
        self.assertTrue(result['deterministic_output_recovery'])

    def test_admitted_without_output_remains_unknown(self):
        units = Units(self.root / 'units'); row = {'uid': digest('a')}
        units.intent(row, '1'); units.admitted(row, '1', 'a-random')
        recover_outputs(units, [row])
        self.assertEqual(units.state(digest('a'))['status'], 'unknown')
        self.assertFalse(units.intent(row, '2'))

    def test_failed_previous_receipt_blocks_model_loading(self):
        c = card(self.cfg, 'qualify', 'gpt', required_cards=['failed'])
        control = Path(self.cfg['control_root'])
        publish(control / 'stages/failed/RESULT.json', {'status': 'execution_failed'})
        handler = Mock(side_effect=AssertionError('model loaded'))
        result = run_stage(self.binding, c, handler)
        self.assertEqual(result['status'], 'upstream_unavailable'); handler.assert_not_called()

    def test_initial_gpu_job_waits_for_cpu_and_has_failure_owners(self):
        plan = initial_graph(self.cfg)
        cpu, gpu = plan['cards'][:2]
        self.assertEqual(cpu['minutes'], 30); self.assertEqual(gpu['minutes'], 120)
        self.assertEqual(gpu['required_cards'], [digest(cpu)])
        self.assertEqual(len(plan['guards']), 2)
        self.assertEqual(gpu['budget_phase'], 'preparation_qualification')

    def test_unfinished_cpu_prices_do_not_delay_qualification_but_gate_admission(self):
        from moe_exp.study.workflow import reader_plan
        source = self.root / 'parent'; self.cfg['source_campaign'] = str(source)
        for model in ('gpt','nemotron','qwen35','qwen36','gemma'):
            publish(source/model/'ANALYSIS_PRICE.json', {'status':'complete'})
        plan = initial_graph(self.cfg)
        pricing = [c for c in plan['cards'] if c['stage'] == 'legacy_price']
        self.assertEqual([c['model'] for c in pricing], ['glm', 'qwen330b'])
        self.assertTrue(all(c['minutes'] == 80 and c['gpus'] == 0 for c in pricing))
        self.assertEqual(plan['terminal_cards'], [digest(plan['cards'][1])])
        publish(Path(self.cfg['control_root'])/'waves/preflight/PLAN.json', plan)
        with patch('moe_exp.study.reservations.remaining_reader_minutes', return_value=236):
            reader = reader_plan(self.binding, next(c for c in plan['cards'] if c['stage']=='advance'))
        admit = next(c for c in reader['cards'] if c['stage'] == 'admit')
        self.assertTrue(set(map(digest,pricing)).issubset(admit['required_cards']))

    def test_lost_launch_receipt_reuses_frozen_plan_after_prices_land(self):
        from moe_exp.study.cli import submit
        control = Path(self.cfg['control_root']); graph = initial_graph(self.cfg)
        publish(control/'waves/preflight/PLAN.json', graph)
        publish(control/'SCHEDULER_VALIDATION.json', {'status':'complete'})
        with patch('moe_exp.study.cli.identity'), patch('moe_exp.study.cli.initialize', return_value=self.binding), patch('moe_exp.study.workflow.initial_graph', side_effect=AssertionError('redrew wave')), patch('moe_exp.study.workflow.submit_wave', return_value={'jobs':{}}) as launch, patch('moe_exp.study.cli.transfer'), patch('moe_exp.study.workflow.release_wave'):
            receipt = submit(self.cfg)
        self.assertEqual(launch.call_args.args[1], graph)
        self.assertEqual(receipt['status'], 'released')

    def test_requeued_stage_preserves_first_start_and_finishes(self):
        candidate = card(self.cfg, 'validate', minutes=30)
        folder = Path(self.cfg['control_root']) / 'stages' / digest(candidate)
        first = publish(folder / 'START.json', {'card_sha256': digest(candidate),
                                               'job_id': '123', 'started_epoch': 1})
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_RESTART_COUNT': '1'}):
            result = run_stage(self.binding, candidate, lambda *args: {'status': 'complete'})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(verified(folder / 'START.json'), first)
        self.assertTrue((folder / 'attempts/123-1/START.json').exists())

    def test_failure_committed_before_crash_recovers_without_handler(self):
        candidate = card(self.cfg, 'validate')
        folder = Path(self.cfg['control_root']) / 'stages' / digest(candidate)
        publish(folder / 'FAILURE.json', {'exception': 'RuntimeError', 'message': 'original failure'})
        handler = Mock(side_effect=AssertionError('retried failed work'))
        result = run_stage(self.binding, candidate, handler)
        self.assertEqual(result['error'], 'original failure')
        handler.assert_not_called()

    def test_shared_recovery_intent_reuses_exact_card_after_submission_failure(self):
        from moe_exp.study.workflow import qualification_recovery
        candidate = card(self.cfg, 'qualify', 'gpt', budget_phase='preparation_qualification')
        identity = digest(candidate); control = Path(self.cfg['control_root'])
        publish(control / 'cards' / (identity + '.json'), candidate)
        publish(control / 'submissions' / identity / 'ACCEPTED.json', {'job_id': '123'})
        failed = [{'card': identity}]
        with patch('moe_exp.study.resources.accounting', return_value={'123': {'state': 'NODE_FAIL'}}):
            first = qualification_recovery(self.binding, failed)
            self.assertEqual(first, qualification_recovery(self.binding, failed))
        with patch('moe_exp.study.resources.accounting', return_value={}):
            with self.assertRaisesRegex(ValueError, 'accounting is unavailable'):
                qualification_recovery(self.binding, failed)

    def test_resume_reconciles_stalled_controller_once_without_gpu_retry(self):
        from moe_exp.study.cli import resume
        from moe_exp.study.workflow import protect
        control = Path(self.cfg['control_root'])
        plan = protect(self.cfg, [], 'preflight', ['engineering'], 'finalization')
        publish(control / 'waves/preflight/PLAN.json', plan)
        publish(control / 'waves/preflight/RELEASED.json', {'status': 'released'})
        ids = {}
        for index, key in enumerate([plan['controller'], *plan['guards']]):
            jid = str(100 + index); ids[jid] = {'state': 'FAILED'}
            publish(control / 'submissions' / key / 'ACCEPTED.json', {'job_id': jid})
        with patch('moe_exp.study.cli.submit', return_value={}), patch('moe_exp.study.cli.active', return_value=self.binding), patch('moe_exp.study.resources.accounting', return_value=ids), patch('moe_exp.study.workflow.submit_wave', return_value={}) as launch:
            resume(self.cfg)
        graph = launch.call_args.args[1]
        self.assertEqual(graph['wave'], 'preflight-resume')
        self.assertTrue(all(c['stage'] in ('advance', 'guard') and c['gpus'] == 0 for c in graph['cards']))
        ids['100'] = {'state': 'CONFIGURING'}
        with patch('moe_exp.study.cli.submit', return_value={}), patch('moe_exp.study.cli.active', return_value=self.binding), patch('moe_exp.study.resources.accounting', return_value=ids), patch('moe_exp.study.workflow.submit_wave') as launch:
            resume(self.cfg)
        launch.assert_not_called()

    def test_terminal_usage_settles_failed_limits_and_running_keeps_full_tail(self):
        c = card(self.cfg, 'qualify', 'gpt', budget_phase='preparation_qualification')
        self.assertEqual(charged_hours(c, {'state': 'FAILED', 'billing_hours': 2.}, 1.25), (2.5, 'terminal_actual'))
        self.assertEqual(charged_hours(c, {'state': 'RUNNING', 'billing_hours': 2.}, 1.25), (80., 'full_time_limit'))

    def test_reservation_is_transactional_idempotent_and_counts_recovery(self):
        jobs = pair(self.cfg, 'reader_pilot', phase='preparation_reader')
        with patch('moe_exp.study.resources.snapshot', return_value=SNAPSHOT), patch('moe_exp.study.reservations.check_storage'):
            first = reserve_many(self.binding, jobs)
            self.assertEqual(first, reserve_many(self.binding, jobs))
            extra = card(self.cfg, 'reader_pilot', attempt=2, budget_phase='preparation_reader')
            before = list(Path(self.cfg['control_root']).glob('reservations/*.json'))
            with self.assertRaisesRegex(ValueError, 'exhausted'):reserve_many(self.binding, [extra])
            self.assertEqual(before, list(Path(self.cfg['control_root']).glob('reservations/*.json')))
        self.assertEqual(ledger(self.binding, usage={})['totals']['preparation_reader'], 80.)

    def test_lost_acceptance_readback_and_duplicate_submit_do_not_repeat_sbatch(self):
        c = card(self.cfg, 'validate', minutes=30, budget_phase='preparation_cpu'); publish(Path(self.cfg['control_root']) / 'reservations' / (digest(c) + '.json'), {'card_sha256': digest(c), 'phase': 'preparation_cpu'}); calls = []; fail = [True]
        def execute(args):
            calls.append(args)
            if args[0] == 'sbatch':return '123'
            if fail[0]:fail[0] = False; raise RuntimeError('lost readback')
            return 'JobId=123 UserId=lmolfett(133943) Account=iscrc_miosr JobName=' + c['job_name'] + ' Partition=boost_usr_prod NumCPUs=4 ReqTRES=cpu=4,billing=4 QOS=normal NumNodes=1 NumTasks=1 MinMemoryNode=24G TimeLimit=00:30:00'
        scheduler = Slurm(self.cfg, execute)
        with self.assertRaisesRegex(RuntimeError, 'lost'):scheduler.submit(c, ['99'])
        a = scheduler.submit(c, ['99']); self.assertEqual(a, scheduler.submit(c, ['99']))
        self.assertEqual(sum(call[0] == 'sbatch' for call in calls), 1)
        self.assertEqual(a['job_id'], '123')
        intent = verified(Path(self.cfg['control_root']) / 'submissions' / digest(c) / 'INTENT.json')
        self.assertIn('--dependency=afterany:99', intent['argv'])
        self.assertTrue(any(arg.endswith('/scripts/study/job.sh') for arg in intent['argv']))

    def test_purged_job_readback_uses_native_accounting(self):
        c = card(self.cfg, 'validate', minutes=30)
        def run(args):
            if args[0] == 'scontrol':raise subprocess.CalledProcessError(1, args, stderr='slurm_load_jobs error: Invalid job id specified')
            return '123|' + c['job_name'] + '|iscrc_miosr|boost_usr_prod|normal|4|1|1|billing=4,cpu=4,mem=24G,node=1|30|COMPLETED|100|2026-10-15T12:00:00|billing=4,cpu=4,mem=24G,node=1'
        result = Slurm(self.cfg, run).readback('123', c)
        self.assertEqual(result['readback_source'], 'sacct')
        self.assertEqual(result['fields']['JobState'], 'COMPLETED')

    def test_failure_finalization_keeps_one_recovery_and_stable_dependencies(self):
        scheduler = Mock(); scheduler.submit.side_effect = [{'job_id': '201'}, {'job_id': '202'}]
        with patch.dict(os.environ, {'SLURM_JOB_ID': '199'}), patch('moe_exp.study.workflow.Slurm', return_value=scheduler), patch('moe_exp.study.workflow.command', return_value=''), patch('moe_exp.study.reservations.reserve_many') as reserve:
            receipt = queue_finalization(self.binding, {'reason': 'failure'})
        self.assertEqual([r['job_id'] for r in receipt['jobs']], ['201', '202'])
        self.assertEqual(scheduler.submit.call_args_list[1].args[1], ['199', '201'])
        self.assertEqual(len(reserve.call_args.args[1]), 2)

    def test_lane_failure_does_not_become_scientific_prerequisite(self):
        publish(self.root / 'ADMISSION.json', {'status': 'admitted', 'gpu_concurrency': 1, 'cpu_concurrency': 1})
        a = pair(self.cfg, 'generate', 'gpt', shard=0)
        b = pair(self.cfg, 'generate', 'gpt', shard=1)
        result = parallel_lanes(self.binding, a + b)
        self.assertEqual(result[2]['depends_on'], [digest(result[1])])
        self.assertFalse(result[2]['required_cards'])
        self.assertEqual(result[3]['recovery_of'], digest(result[2]))

    def test_plan_is_read_only_and_retains_all_model_slots(self):
        from moe_exp.study.cli import plan
        publish(self.root / 'BINDING.json', self.binding)
        before = sorted(str(p) for p in self.root.rglob('*'))
        with patch('moe_exp.study.resources.snapshot', return_value=SNAPSHOT):
            value = plan(self.cfg)
        self.assertTrue(value['read_only'])
        self.assertEqual(len(value['assignments']), 8)
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_unreserved_submission_never_calls_sbatch(self):
        calls = []
        candidate = card(self.cfg, 'validate', minutes=30, budget_phase='preparation_cpu')
        with self.assertRaises(FileNotFoundError):
            Slurm(self.cfg, lambda argv: calls.append(argv)).submit(candidate)
        self.assertFalse(calls)

    def test_source_checkout_receipts_are_excluded_from_live_ledger(self):
        path = self.root / 'study/release/source/report/old/submissions/fake/ACCEPTED.json'
        publish(path, {'job_id': '999', 'card_sha256': 'fake'})
        self.assertEqual(ledger(self.binding, usage={})['rows'], [])

    def test_missing_capture_layer_keeps_raw_output_and_marks_unknown(self):
        from types import SimpleNamespace
        import time
        from test_session import FakeLLM
        from moe_exp.study.runtime import Backend
        backend = Backend.__new__(Backend)
        backend.binding = self.binding; backend.model = 'lfm25'; backend.capacity = 1
        backend.profile = {'layers': [2], 'card': {'max_model_len': 32, 'max_tokens': None}}
        backend.deadline = time.time() + 60; backend.closure = [[9]]; backend.load_seconds = 1.
        backend.tokenizer = SimpleNamespace(decode=lambda ids, **kw: 'one')
        backend.llm = FakeLLM()
        def step():
            values = list(backend.llm.llm_engine.live.values()); backend.llm.llm_engine.live.clear()
            return [SimpleNamespace(request_id=uid, finished=True, outputs=[SimpleNamespace(
                token_ids=[3], finish_reason='stop', stop_reason=None)]) for uid in values]
        backend.llm.llm_engine.step = step
        row = {'uid': digest('missing-layer'), 'action': None, 'capture_layers': [2], 'seed': 42}
        units = Units(self.root / 'routing-units')
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU-TEST'}), patch('moe_exp.study.runtime.prompt_prefix', return_value=([1], [2], {})), patch('moe_exp.study.runtime.sampling_parameters', return_value=SimpleNamespace(max_tokens=8)):
            result = backend.run([row], units, engineering_tokens=8)
        self.assertEqual(result[0]['token_ids'], [3])
        self.assertEqual(units.state(row['uid'])['status'], 'unknown')
        self.assertEqual(units.state(row['uid'])['reason'], 'required_routing_capture_missing')
        self.assertFalse((units.folder(row['uid']) / 'CAPTURE.json').exists())
        self.assertFalse(backend.llm.attached)

    def test_inherited_scientific_checks_do_not_patch_the_old_scheduler(self):
        from moe_exp.study.inherited import parent_campaign
        from moe_exp.research_campaign_v3_r6 import scheduler as historical
        original = (historical.card, historical.Slurm.verify)
        binding = {**self.binding, 'source_campaign': self.cfg['source_campaign']}
        with patch('moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1.check_binding', return_value={}):
            parent = parent_campaign(binding)
        self.assertEqual(original, (historical.card, historical.Slurm.verify))
        self.assertEqual(parent['root'], self.cfg['source_campaign'])
