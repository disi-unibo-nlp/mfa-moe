import copy
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from moe_exp.final_routing.common import digest,file_sha,publish,verified
from moe_exp.final_routing.readouts import HISTORICAL
from moe_exp.final_routing_ops_r1 import adapter
from moe_exp.final_routing_ops_r1.launch import cpu_envelope,unused_cancelled


class RegistryTests(unittest.TestCase):
    def test_actual_historical_nulls_preserve_all_seven_streams_and_token_strata(self):
        sys.path.insert(0,str(HISTORICAL/'code'))
        from dp import s3_readout as sr
        original=sr.cm.MODELS;sr.cm.MODELS=adapter.ORIGINAL_MODELS
        source=Path(sr.__file__);before_sha=file_sha(source)
        md=SimpleNamespace(tokid=np.repeat(np.arange(8),16),tok_abs=np.tile(np.arange(4),32),
            top1=np.arange(3*128).reshape(3,128)%64,model='gpt')
        expected={}
        for model in adapter.ORIGINAL_MODELS:
            md.model=model;expected[model]=[sr.null_paths(md,draw) for draw in range(5)]
        try:
            adapter.register_lfm();self.assertEqual(sr.cm.MODELS,adapter.EXTENDED_MODELS)
            for model in adapter.ORIGINAL_MODELS:
                md.model=model
                for draw in range(5):np.testing.assert_array_equal(expected[model][draw],sr.null_paths(md,draw))
            for model,index in [('lfm25',7),('lfm25-engineering',8)]:
                self.assertEqual(sr.cm.MODELS.index(model),index);md.model=model
                for draw in range(5):
                    result=sr.null_paths(md,draw);np.testing.assert_array_equal(result,sr.null_paths(md,draw))
                    strata=md.tokid*8+md.tok_abs
                    for value in np.unique(strata):
                        mask=strata==value
                        self.assertEqual(sorted(map(tuple,result[:,mask].T)),sorted(map(tuple,md.top1[:,mask].T)))
            self.assertEqual(before_sha,file_sha(source))
        finally:sr.cm.MODELS=original

    def test_unknown_registry_rejected(self):
        from dp import s3_readout as sr
        with patch.object(sr.cm,'MODELS',('unexpected',)):
            with self.assertRaises(ValueError):adapter.register_lfm()


class BudgetTests(unittest.TestCase):
    def fixture(self):
        old=[{'stage':'cpu_price','minutes':120,'budget_phase':'preparation_cpu'},
             {'stage':'cpu_price','minutes':60,'budget_phase':'preparation_cpu'},
             {'stage':'prepare','minutes':285,'budget_phase':'preparation_cpu'}]
        accepted={digest(c):{'job_id':str(i)} for i,c in enumerate(old)}
        usage={'0':{'state':'FAILED','seconds':157},'1':{'state':'FAILED','seconds':28},
               '2':{'state':'RUNNING','seconds':5}}
        return old,accepted,usage

    def test_failures_charged_and_running_full_tail_reserved(self):
        old,a,u=self.fixture();new=[{'stage':'cpu_price','minutes':120,'budget_phase':'preparation_cpu','amended':True}]
        result=cpu_envelope(old,a,u,new)
        self.assertEqual(result['aggregate_seconds'],185+(285+120)*60)
        self.assertFalse(result['original_allowance_reset'])
        self.assertEqual([r['basis'] for r in result['rows']],['terminal_actual','terminal_actual','full_time_limit','full_time_limit'])

    def test_overspend_and_missing_accounting_rejected(self):
        old,a,u=self.fixture()
        with self.assertRaises(ValueError):cpu_envelope(old,a,u,[{'minutes':200,'budget_phase':'preparation_cpu'}])
        with self.assertRaises(ValueError):cpu_envelope(old,a,{},[])

    def test_retirement_requires_all_actual_unused_evidence(self):
        row={'state':'CANCELLED','seconds':0,'start':'None','allocated_tres':''}
        self.assertTrue(unused_cancelled(row))
        for key,value in [('state','FAILED'),('seconds',1),('start','2026-10-15T00:00:00'),('allocated_tres','cpu=4')]:
            self.assertFalse(unused_cancelled({**row,key:value}))


class JournalTests(unittest.TestCase):
    def test_actual_argv_and_duplicate_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proof=publish(root/'OPS_R1_BINDING.json',{'base':'test'})
            cfg={'root':str(root),'environment':'/private/env','slurm':{'account':'iscrc_miosr','partition':'boost_usr_prod','qos':'normal'}}
            c={'stage':'cpu_price','cpus':4,'gpus':0,'memory_gib':24,'minutes':120,'job_name':'test-r1',
               'ops_r1_sha256':proof['sha256']}
            calls=[]
            def run(args):calls.append(args);return '12345678'
            scheduler=adapter.Slurm(cfg,run=run)
            with patch.object(scheduler,'readback',return_value={'job_id':'12345678','fields':{}}):
                first=scheduler.submit(c,['11111111'],hold=True)
                second=scheduler.submit(c,['11111111'],hold=True)
            self.assertEqual(first,second);self.assertEqual(len(calls),1)
            intent=verified(root/'submissions'/digest(c)/'INTENT.json')
            self.assertEqual(intent['argv'],calls[0]);self.assertIn('--dependency=afterany:11111111',calls[0])
            self.assertTrue(any(v.endswith('/final_routing_ops_r1.sbatch') for v in calls[0]))
            with self.assertRaises(ValueError):scheduler.submit(c,['22222222'],hold=True)

    def test_missing_terminal_result_prevents_production_and_queues_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proof=publish(root/'OPS_R1_BINDING.json',{'test':True})
            c={'wave':0,'stage':'advance'};identity=digest(c)
            publish(root/'OPS_R1_INITIAL_GRAPH.json',{'ops_r1_sha256':proof['sha256'],'controller':identity,'terminal_cards':['missing']})
            with patch.object(adapter.base_workflow,'queue_finalization',return_value={'owned':True}) as close,patch.object(adapter.base_workflow,'production_plan') as production:
                value=adapter.advance({'root':str(root)},c,0)
            self.assertEqual(value['status'],'complete');close.assert_called_once();production.assert_not_called()
            self.assertEqual(close.call_args.args[1]['reason'],'stage_failure')


    def test_bound_wave_zero_gate_requires_replacement_admission(self):
        from moe_exp.final_routing import journal,execution,workflow
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);binding={'root':str(root),'sha256':'base'}
            proof=publish(root/'OPS_R1_BINDING.json',{'base_binding_sha256':'base','files':{}})
            publish(root/'OPS_R1_INITIAL_GRAPH.json',{'ops_r1_sha256':proof['sha256'],'terminal_cards':['replacement']})
            publish(root/'waves/00/PLAN.json',{'terminal_cards':['original']})
            publish(root/'stages/original/RESULT.json',{'status':'complete'})
            for module,name in [(workflow,'node'),(workflow,'Slurm'),(workflow,'advance'),(workflow,'queue_finalization'),
                                (journal,'dependency_results'),(execution,'handler'),(adapter,'_installed')]:
                stack.enter_context(patch.object(module,name,getattr(module,name)))
            adapter._installed=None;adapter.install(binding)
            c={'previous_wave':0,'ops_r1_sha256':proof['sha256']}
            self.assertEqual(journal.dependency_results(root,c)[0]['status'],'upstream_unavailable')
            publish(root/'stages/replacement/RESULT.json',{'status':'complete'})
            self.assertEqual(journal.dependency_results(root,c),[verified(root/'stages/replacement/RESULT.json')])
            with self.assertRaises(ValueError):journal.dependency_results(root,{'previous_wave':0})

    def test_failure_closeout_retires_only_verified_unused_held_worker(self):
        from moe_exp.final_routing import reservations
        from moe_exp.final_routing_ops_r1 import launch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c={'stage':'generate'};key=digest(c)
            publish(root/'OPS_R1_BINDING.json',{'test':True})
            publish(root/'cards'/(key+'.json'),c)
            publish(root/'submissions'/key/'ACCEPTED.json',{'card_sha256':key,'job_id':'42'})
            finals=[{'stage':'finalize','attempt':0},{'stage':'finalize','attempt':1}]
            def command(args):
                if args[0]=='squeue':return '42'
                if args[0]=='scontrol':return 'JobState=PENDING Reason=JobHeldUser RunTime=00:00:00 StartTime=Unknown'
                if args[0]=='scancel':self.assertEqual(args,['scancel','42']);return ''
                raise AssertionError(args)
            row={'state':'CANCELLED','seconds':0,'start':'None','allocated_tres':''}
            with patch.object(adapter.base_scheduler,'command',side_effect=command),patch.object(launch,'native_retirement',return_value={'42':row}),patch.object(adapter.base_workflow,'pair',return_value=finals),patch.object(reservations,'reserve'),patch.object(adapter,'Slurm') as scheduler,patch.dict('os.environ',{'SLURM_JOB_ID':'99'}):
                scheduler.return_value.submit.side_effect=[{'job_id':'101'},{'job_id':'102'}]
                value=adapter.queue_finalization({'root':str(root),'config':{}},{'reason':'fixture_failure'})
            intent=verified(root/'FINALIZATION_INTENT.json')
            self.assertEqual(intent['dependencies'],['99']);self.assertEqual(intent['unused_held_retirements'],{'42':row})
            self.assertEqual([r['job_id'] for r in value['jobs']],['101','102'])
            self.assertEqual(scheduler.return_value.submit.call_args_list[1].args[1],['99','101'])


if __name__=='__main__':unittest.main()
