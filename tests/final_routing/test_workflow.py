import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from moe_exp.final_routing.common import WORK,digest,publish,verified
from moe_exp.final_routing.config import read_config
from moe_exp.final_routing.workflow import initial_graph,pair,payload,parallel_lanes,queue_finalization
from moe_exp.final_routing.scheduler import Slurm,card
from moe_exp.final_routing.journal import run_stage,stage_directory
from moe_exp.final_routing.reservations import reserve,reserve_many
from moe_exp.final_routing.admission import pack


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='final-routing-contract-',dir=WORK/'tools/tmp')
        self.root=Path(self.temp.name)
        self.config=read_config('configs/final_routing.yaml');self.config['root']=str(self.root)
        self.binding={'root':str(self.root),'config':self.config}

    def tearDown(self):self.temp.cleanup()

    def test_exact_preparation_envelope_and_protected_failure_owners(self):
        graph=initial_graph(self.config,['123_4'])
        totals={}
        for c in graph['cards']:
            totals[c['budget_phase']]=totals.get(c['budget_phase'],0)+c['minutes']/60*c['billing_units']*1.25
        self.assertEqual(totals['preparation_cpu'],40)
        self.assertEqual(totals['preparation_reader'],80)
        self.assertEqual(totals['preparation_qualification'],1280)
        self.assertEqual(len(graph['guards']),2)
        self.assertEqual({c['model'] for c in graph['cards'] if c['stage']=='qualify'},set(self.config['models']))
        seen=set()
        for c in graph['cards']:
            self.assertTrue(set(c['depends_on'])<=seen)
            seen.add(digest(c))

    def test_failed_prerequisite_never_calls_gpu_handler(self):
        dependency=digest('failed')
        publish(self.root/'stages'/dependency/'RESULT.json',{'status':'execution_failed'})
        c=card(self.config,'qualify','gpt',required_cards=[dependency])
        handler=Mock(side_effect=AssertionError('model loaded'))
        result=run_stage(self.binding,c,handler)
        self.assertEqual(result['status'],'upstream_unavailable');handler.assert_not_called()
        self.assertFalse(result['model_loaded'])

    def test_successful_attempt_skips_reserved_recovery_without_loading(self):
        first,second=pair(self.config,'qualify','lfm25',phase='preparation_qualification')
        publish(stage_directory(self.root,first)/'RESULT.json',{'status':'complete'})
        handler=Mock(side_effect=AssertionError('redraw'))
        self.assertEqual(run_stage(self.binding,second,handler)['status'],'complete');handler.assert_not_called()

    def test_reservations_are_idempotent_and_recoveries_are_charged(self):
        first,second=pair(self.config,'reader_pilot',phase='preparation_reader')
        self.assertEqual(reserve(self.binding,first),reserve(self.binding,first))
        self.assertEqual(reserve(self.binding,second)['aggregate_reserved_hours'],80)
        other=card(self.config,'reader_pilot',attempt=2,budget_phase='preparation_reader')
        with self.assertRaisesRegex(ValueError,'exhausted'):reserve(self.binding,other)

    def test_over_budget_wave_reserves_nothing(self):
        cards=pair(self.config,'reader_pilot',phase='preparation_reader')
        cards.append(card(self.config,'reader_pilot',attempt=2,budget_phase='preparation_reader'))
        with self.assertRaisesRegex(ValueError,'exhausted'):reserve_many(self.binding,cards)
        self.assertEqual(list((self.root/'reservations').glob('*.json')),[])

    def test_missing_previous_wave_receipts_prevent_loading(self):
        c=card(self.config,'generate','lfm25',previous_wave=6)
        handler=Mock(side_effect=AssertionError('loaded before design receipts'))
        self.assertEqual(run_stage(self.binding,c,handler)['status'],'upstream_unavailable')
        handler.assert_not_called()

    def test_primary_design_is_immutable(self):
        from moe_exp.final_routing.admission import freeze_primary
        publish(self.root/'ADMISSION.json',{'status':'admitted','mode':'full'})
        original=publish(self.root/'DESIGN.json',{'status':'complete','mode':'fallback'})
        with patch('moe_exp.final_routing.admission.forecasts',side_effect=AssertionError('redesign')):
            self.assertEqual(freeze_primary(self.binding),original)

    def test_native_quota_wrapped_mount_and_unwrapped_mount(self):
        from moe_exp.final_routing.resources import quota_bytes
        values='336492028 1073741824 0 - 248556 0 0 -'
        for separator in (' ','\n                '):
            self.assertEqual(quota_bytes('/leonardo_scratch/fast'+separator+values,'/leonardo_scratch/fast'),
                             (1073741824-336492028)*1024)
        with self.assertRaisesRegex(ValueError,'missing exact quota'):
            quota_bytes('/leonardo_work '+values,'/leonardo_scratch/fast')

    def test_launch_checks_cluster_identity_before_any_mutation(self):
        from moe_exp.final_routing.cli import launch
        with patch('moe_exp.final_routing.cli.identity',side_effect=RuntimeError('identity probe')) as identity:
            with self.assertRaisesRegex(RuntimeError,'identity probe'):launch(self.config)
            identity.assert_called_once_with()
        self.assertFalse((self.root/'BINDING.json').exists())

    def test_packing_retains_every_identity_and_five_hour_limit(self):
        rows=[{'uid':digest(i)} for i in range(103)]
        groups=pack(rows,lambda r:600,600,capacity=4)
        self.assertEqual(sorted(r['uid'] for g in groups for r in g['records']),sorted(r['uid'] for r in rows))
        self.assertTrue(all(g['planned_seconds']<=18000 for g in groups))
        with self.assertRaisesRegex(ValueError,'cannot fit'):pack(rows,lambda r:18001,0)

    def test_lane_recovery_points_to_final_first_identity(self):
        publish(self.root/'ADMISSION.json',{'status':'admitted','gpu_concurrency':1,'cpu_concurrency':1})
        a=pair(self.config,'generate','gpt',shard=0)
        b=pair(self.config,'generate','gpt',shard=1)
        result=parallel_lanes(self.binding,a+b)
        self.assertEqual(result[3]['recovery_of'],digest(result[2]))
        self.assertEqual(result[2]['depends_on'],[digest(result[1])])
        self.assertFalse(result[2]['required_cards'])

    def test_lost_acceptance_readback_does_not_repeat_sbatch(self):
        c=card(self.config,'validate');calls=[];fails=[True]
        def execute(args):
            calls.append(args)
            if args[0]=='sbatch':return '123'
            if fails[0]:fails[0]=False;raise RuntimeError('lost readback')
            return 'JobId=123 UserId=lmolfett(133943) Account=iscrc_miosr JobName='+c['job_name']+' Partition=boost_usr_prod NumCPUs=4 ReqTRES=cpu=4,billing=4 QOS=normal NumNodes=1 NumTasks=1 MinMemoryNode=24G TimeLimit=02:00:00'
        scheduler=Slurm(self.config,execute)
        with self.assertRaisesRegex(RuntimeError,'lost'):scheduler.submit(c,['99'])
        self.assertEqual(scheduler.submit(c,['99'])['job_id'],'123')
        self.assertEqual(sum(a[0]=='sbatch' for a in calls),1)

    def test_finalization_has_one_accepted_recovery_and_stable_dependencies(self):
        scheduler=Mock();scheduler.submit.side_effect=[{'job_id':'201'},{'job_id':'202'}]
        with patch.dict(os.environ,{'SLURM_JOB_ID':'199'}),patch('moe_exp.final_routing.workflow.Slurm',return_value=scheduler),patch('moe_exp.final_routing.workflow.command',return_value=''):
            receipt=queue_finalization(self.binding,{'reason':'failure'})
        self.assertEqual([r['job_id'] for r in receipt['jobs']],['201','202'])
        self.assertEqual(scheduler.submit.call_args_list[1].args[1],['199','201'])
        self.assertEqual(len(list((self.root/'reservations').glob('*.json'))),2)


if __name__=='__main__':unittest.main()
