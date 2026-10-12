"""Actual masking wrapper, failure preservation, saved joins and admission."""
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from moe_exp.research_campaign_v3_r4 import contracts as C
from moe_exp.research_campaign_v3_r4.binding import RouterBinding
from moe_exp.research_campaign_v3_r4.adapters import SignedCapture
from moe_exp.research_campaign_v3_r4.measurement import ForwardObserver
from moe_exp.research_campaign_v3_r4.lifecycle import cleanup, restore
from moe_exp.research_campaign_v3_r4.populations import token_ownership
from moe_exp.research_campaign_v3_r4.accuracy import joined_accuracy, attempt_key
from moe_exp.research_campaign_v3_r4.scheduler import card, serial_capacity, verified_resources
from moe_exp.research_campaign_v3_r4.statistics import bootstrap_panel
from moe_exp.expert_memory_study.loader import RouterMask
from moe_exp.research_campaign_v3_r4 import scheduler as S, controller as O
from moe_exp.research_campaign_v3_r4.resume import transfer
from test_campaign import FakeScheduler


def model_fixture():
    import torch
    class Router:
        top_k=2;global_num_experts=4;eplb_state=None
        e_score_correction_bias=torch.tensor([.04,.03,.02,.01])
        def _compute_routing(self, hidden_states, router_logits, indices_type=None):
            self.hidden=hidden_states
            ids=(router_logits+self.e_score_correction_bias).topk(2,-1).indices
            weights=torch.softmax(router_logits,-1).gather(1,ids)*1.7
            return weights,ids.to(indices_type or torch.int64)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.router=Router()
            class Experts:
                def forward_modular(self,x,w,ids):
                    return (x[:,None,:]*(ids[:,:,None]+1)*w[:,:,None]).sum(1)
            self.runner=SimpleNamespace(routed_experts=Experts(),_fused_output_is_reduced=True,moe_config=SimpleNamespace(tp_size=1))
        def named_modules(self,*args,**kwargs):
            yield 'layers.3.mlp.experts',SimpleNamespace(router=self.router,routed_experts=None)
        def get_submodule(self,name):return self.runner
        def forward(self,input_ids,positions,keyword=False):
            x=torch.ones(len(input_ids),3);logits=torch.tensor([[3.,2.,1.,0.]]*len(input_ids))
            if keyword:return self.router._compute_routing(hidden_states=x,router_logits=logits,indices_type=torch.int32)
            return self.router._compute_routing(x,logits,torch.int32)
    return Model()


class WrappedRouters(unittest.TestCase):
    def test_native_binding_masks_keyword_positional_and_identity(self):
        import torch
        model=model_fixture();mask=RouterMask(model.router,[1,2,3],4,compact=False)
        installed=model.router._compute_routing;prior=model.router.e_score_correction_bias
        binding=RouterBinding(model.router,('hidden_states','router_logits','indices_type'))
        x=torch.ones(2,3);logits=torch.tensor([[100.,1.,2.,3.]]*2)
        for bound in (binding.bind(x,logits,torch.int32),binding.bind(hidden_states=x,router_logits=logits,indices_type=torch.int32)):
            _,ids=binding.invoke(bound)
            self.assertIs(model.router.hidden,x);self.assertNotIn(0,ids.flatten().tolist())
            self.assertIs(model.router.e_score_correction_bias,prior)
        self.assertIs(model.router._compute_routing,installed)
        def unsupported(*args,**kwargs):pass
        model.router._compute_routing=unsupported
        with self.assertRaisesRegex(ValueError,'unsupported variadic'):RouterBinding(model.router)
    def test_both_signs_chunked_prefix_closure_repeated_restore(self):
        import torch
        model=model_fixture();mask=RouterMask(model.router,[0,1,2],4,compact=False)
        original=model.router._compute_routing
        contract={'expected_routed_layers':1,'native_k':2,'experts':4}
        for sign in (-1,1,-1):
            action={'layer':3,'experts':[1,2],'bias':10.*sign,'start':2,'pulse_tokens':3}
            capture=SignedCapture(model,contract,[9],[8,7],[[6]],action,[3],8,audit=True)
            model(torch.tensor([9,8]),torch.tensor([0,1]),keyword=True)
            model(torch.tensor([7]),torch.tensor([2]))
            for position,token in ((3,5),(4,5),(5,5),(6,6)):
                model(torch.tensor([token]),torch.tensor([position]),keyword=bool(position%2))
            result=capture.finish();rows=result['capture']['3']
            self.assertEqual(rows['active'],[True,True,True,False,False])
            self.assertTrue(result['closed']);self.assertNotIn(3,np.array(rows['executed_ids']).flatten())
            self.assertIs(model.router._compute_routing,original)
            self.assertEqual(len(model._forward_pre_hooks),0)
    def test_measurement_wrapped_complete_unique_positions(self):
        import torch
        model=model_fixture();RouterMask(model.router,[0,1,2,3],4,compact=False)
        original=model.router._compute_routing
        observer=ForwardObserver(model,{'layers':[3],'native_k':2,'experts':4},[9],[8,7],[[6]],[0,1],[0,1])
        model(torch.tensor([9,8]),torch.tensor([0,1]),keyword=True)
        model(torch.tensor([7]),torch.tensor([2]))
        data=observer.finish()
        self.assertEqual(data['layers']['3']['positions'],[0,1])
        self.assertIs(model.router._compute_routing,original)
        self.assertEqual(len(model._forward_pre_hooks),0)
    def test_incomplete_and_duplicate_calibration_fail_clean_execution(self):
        for duplicate in (False,True):
            model=model_fixture();RouterMask(model.router,[0,1,2,3],4,compact=False)
            original=model.router._compute_routing
            observer=ForwardObserver(model,{'layers':[3],'native_k':2,'experts':4},[9],[8],[[6]],[0],[0])
            if duplicate:observer.data['3']['positions']=[0,0]
            with self.assertRaisesRegex(ValueError,'duplicate' if duplicate else 'missing'):observer.finish()
            self.assertIs(model.router._compute_routing,original)
    def test_original_failure_secondary_cleanup_and_hook_restoration(self):
        model=model_fixture();RouterMask(model.router,[0,1,2,3],4,compact=False)
        original=model.router._compute_routing
        observer=ForwardObserver(model,{'layers':[3],'native_k':2,'experts':4},[9],[8],[[6]],[0],[0])
        removed=[]
        class Handle:
            def remove(self):removed.append(True);raise RuntimeError('secondary cleanup')
        observer.handle.remove();observer.handle=Handle()
        primary=KeyError('original worker failure')
        with self.assertRaises(KeyError) as raised:
            try:raise primary
            finally:observer.restore()
        self.assertIs(raised.exception,primary)
        self.assertEqual(primary.campaign_cleanup_errors[0]['phase'],'remove_forward_hook')
        self.assertIs(model.router._compute_routing,original);self.assertTrue(removed)


class SavedPopulations(unittest.TestCase):
    def layout(self):
        return {'token_count':5,'units':[{'start':0,'end':2},{'start':2,'end':4},{'start':4,'end':6}],
                'unit_tokens':[[0,1],[2],[3,4]],'reasoning_tokens':[0,1,2,3,4]}
    def ownership(self,labels,counts):
        with patch('moe_exp.schemas.TraceRecord',side_effect=lambda **value:SimpleNamespace(**value)),patch(
            'moe_exp.correlation_pipeline.dynamics.classes.saved_layout',return_value=self.layout()),patch(
            'moe_exp.correlation_pipeline.spans.trace_digest',return_value='sha'):
            return token_ownership({},labels,'sha',counts)
    def test_missing_sentence_cache_recovered_shuffled_original_labels(self):
        labels=[(2,4,6,'Verify'),(0,0,2,'Read')]
        counts=[2,0,0,0,0,2,0,5,4]
        a,_,_=self.ownership(labels,counts);b,_,_=self.ownership(list(reversed(labels)),counts)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[5].tolist(),[3,4]);self.assertNotIn(2,np.concatenate(a))
    def test_conflicts_and_annotation_gaps_are_not_relabelled(self):
        labels=[(0,0,2,'Read'),(0,0,2,'Verify'),(2,4,6,'Plan')]
        classes,_,audit=self.ownership(labels,[0,0,2,0,0,0,0,5,2])
        self.assertEqual(audit['conflicting'],1);self.assertEqual(classes[2].tolist(),[3,4])
        with self.assertRaisesRegex(ValueError,'nine class counts'):
            self.ownership(labels,[2,0,2,0,0,0,0,5,4])
    def test_offset_conflict_rejected_against_original_counts(self):
        with self.assertRaises(ValueError):self.ownership([(0,1,2,'Read')],[2,0,0,0,0,0,0,5,2])
    def test_accuracy_uses_exact_draw_not_question(self):
        common={'dataset':'math500','source_problem_id':'q','trace_sha256':'sha','completion_tokens':5,
                'strict_correct':False,'adjudicated_correct':None}
        a={**common,'sample_id':0,'problem_id':'q0','termination':'stop'}
        b={**common,'sample_id':1,'problem_id':'q1','termination':'length','trace_sha256':'other'}
        metadata={attempt_key(a):{'completion_tokens':5,'finish_reason':'stop','answer_present':True,'location':'a'},
                  attempt_key(b):{'completion_tokens':5,'finish_reason':'length','answer_present':True,'location':'b'}}
        rows=joined_accuracy([a,b],metadata,{'math500|q':'family'})
        self.assertEqual(rows[0]['unknown_upper'],1);self.assertEqual(rows[1]['unknown_upper'],0)
        self.assertNotEqual(rows[0]['attempt_id'],rows[1]['attempt_id'])
        with self.assertRaises(ValueError):joined_accuracy([a,b],{attempt_key(a):metadata[attempt_key(a)]},{'math500|q':'family'})


class UncertaintyAndResources(unittest.TestCase):
    def test_supported_linear_interval_beside_unestimable_odds_shared_draws(self):
        rows=[{'benchmark':b,'family':str(i),'a':i/40,'b':(i%4)/10,'zero':0.} for b in ('a','b') for i in range(40)]
        linear=bootstrap_panel(rows,['a','b','zero'],panel='test:linear',draws=1000,alpha=.025,resample_key='same',
            transform=(['difference'],lambda v:(v[:,0]-v[:,1])[:,None],[[0,1]]))
        odds=bootstrap_panel(rows,['a','b','zero'],panel='test:odds',draws=1000,alpha=.025,resample_key='same',
            transform=(['odds'],lambda v:np.full((len(v),1),np.nan),[[0,1,2]]))
        self.assertIsNotNone(linear['results']['difference']['interval'])
        self.assertIsNone(odds['results']['odds']['interval'])
        self.assertEqual(linear['resample_key'],odds['resample_key'])
        self.assertEqual(odds['undefined_draws_discarded'],0)
    def test_no_undefined_draws_dropped_to_create_finite_interval(self):
        rows=[{'benchmark':'a','family':str(i),'x':float(i==0)} for i in range(10)]
        def operation(v):
            with np.errstate(divide='ignore'):return np.log(v)
        panel=bootstrap_panel(rows,['x'],panel='test:undefined',draws=1000,transform=(['log'],operation,[[0]]))
        self.assertLess(panel['results']['log']['bootstrap_valid_resamples'],1000)
        self.assertIsNone(panel['results']['log']['interval'])
    def test_transformed_support_only_uses_its_declared_inputs(self):
        rows=[{'benchmark':'a','family':str(i),'x':float(i),'missing':None} for i in range(10)]
        panel=bootstrap_panel(rows,['x','missing'],panel='test:dependencies',draws=1000,
            transform=(['linear'],lambda v:v[:,:1],[[0]]))
        self.assertEqual(panel['results']['linear']['family_support'],10)
        self.assertIsNotNone(panel['results']['linear']['interval'])
    def test_joint_serial_memory_cpu_job_limits(self):
        campaign=C.seal({'root':'root','models':{}});worker=card(campaign,'derive','gpt');owner=card(campaign,'controller')
        snapshot={'jobs':[{'partition':'lrd_all_serial','state':'RUNNING','allocation_cpus':2,'memory_mib':4096}]}
        self.assertTrue(all(serial_capacity(snapshot,[worker]).values()))
        self.assertFalse(serial_capacity(snapshot,[worker,worker])['memory'])
        snapshot['jobs'].append({'partition':'lrd_all_serial','state':'RUNNING','allocation_cpus':4,'memory_mib':24576})
        self.assertFalse(serial_capacity(snapshot,[worker])['running'])
        self.assertFalse(serial_capacity(snapshot,[worker])['memory'])
        self.assertEqual((owner['cpus'],owner['memory_gib']),(2,4))
    def test_billing_rounding_and_resource_rejection(self):
        value=card(C.seal({'root':'root'}),'controller')
        proof=verified_resources({'AllocTRES':'(null)','ReqTRES':'cpu=2,mem=4G,billing=2'},value)
        self.assertEqual(proof['allocation_cpus'],2)
        self.assertEqual(proof['reservation_hours'],value['reservation_hours'])
        with self.assertRaises(ValueError):verified_resources({'ReqTRES':'cpu=4,mem=4G,billing=4'},value)
        with self.assertRaises(ValueError):verified_resources({'ReqTRES':'cpu=2,mem=4G'},value)


class OwnershipRecovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='repair-transfer-',dir=C.REPO/'report')
        self.root=Path(self.temp.name)
        self.old=C.publish(self.root/'old/CAMPAIGN.json',{'root':str(self.root/'old'),'code_files':{},'source_files':{}})
        self.new=C.publish(self.root/'new/CAMPAIGN.json',{'root':str(self.root/'new'),'resume_from':str(self.root/'old'),
            'code_files':{},'source_files':{}})
    def tearDown(self):self.temp.cleanup()
    def scheduler(self):
        class Scheduler(FakeScheduler):
            def command(inner,args):
                if args[:3]==['scontrol','show','job']:
                    value=inner.values[args[-1]]
                    return 'Comment='+value['card']['comment']+' Reason='+('JobHeldUser' if value.get('held') else 'Dependency')
                if args[:2]==['scontrol','hold']:
                    inner.values[args[-1]]['held']=True
                    return ''
                return super().command(args)
        return Scheduler()
    def test_prearmed_transfer_preserves_scientific_jobs_and_resumes_retirements(self):
        scheduler=self.scheduler()
        scientific=S.submit_card(self.old,S.card(self.old,'coordination','gpt',1),scheduler)
        old_handoff=S.protected_controller(self.old,scheduler,generation=0)
        prearmed=S.protected_controller(self.new,scheduler,generation=0,release=False)
        self.assertEqual(len(scheduler.releases),3)
        receipt=transfer(self.new,scheduler,prearmed)
        self.assertFalse(scheduler.values[scientific['job_id']]['terminal'])
        self.assertEqual(len(receipt['old_operational_retirements']),3)
        self.assertEqual(transfer(self.new,scheduler,prearmed),receipt)
        for owner in [prearmed['primary'],*prearmed['guards']]:
            self.assertNotIn(owner['job_id'],scheduler.releases)
    def test_controller_crash_after_accept_before_release_has_owners(self):
        scheduler=self.scheduler();scheduler.lost=True
        with self.assertRaises(TimeoutError):S.protected_controller(self.new,scheduler,generation=0,release=False)
        handoff=S.protected_controller(self.new,scheduler,generation=0,release=False)
        self.assertEqual(len(scheduler.values),3);self.assertEqual(scheduler.releases,[])
        S.protected_controller(self.new,scheduler,generation=0)
        self.assertEqual(scheduler.releases,[handoff['guards'][1]['job_id'],handoff['guards'][0]['job_id'],handoff['primary']['job_id']])
    def test_interrupted_retirement_is_reconciled_without_duplicate_dispatch(self):
        scheduler=self.scheduler()
        S.protected_controller(self.old,scheduler,generation=0)
        prearmed=S.protected_controller(self.new,scheduler,generation=0,release=False)
        original=scheduler.command;lost=[True]
        def command(args):
            result=original(args)
            if args[0]=='scancel' and lost[0]:
                lost[0]=False
                raise TimeoutError('cancel accepted, response lost')
            return result
        scheduler.command=command
        with self.assertRaises(TimeoutError):transfer(self.new,scheduler,prearmed)
        receipt=transfer(self.new,scheduler,prearmed)
        self.assertEqual(len(receipt['old_operational_retirements']),3)
        self.assertEqual(len(scheduler.values),6)
    def test_duplicate_array_submission_after_lost_receipt_reconciles_once(self):
        scheduler=self.scheduler();scheduler.lost=True
        parent=S.array_card([S.card(self.new,'coordination','gpt',i) for i in (1,2)])
        with self.assertRaises(TimeoutError):S.submit_card(self.new,parent,scheduler)
        first=S.submit_card(self.new,parent,scheduler)
        self.assertEqual(first,S.submit_card(self.new,parent,scheduler))
        self.assertEqual(len(scheduler.values),3)


class StagedQualification(unittest.TestCase):
    def test_mapping_storage_is_reserved_only_until_committed(self):
        with tempfile.TemporaryDirectory(prefix='repair-mapping-budget-',dir=C.REPO/'report') as temp:
            root=Path(temp)
            campaign={'root':temp,'resume_from':str(root/'old'),'models':{m:{'experts':8} for m in C.MODELS}}
            for model in C.MODELS:
                C.publish(root/model/'SOURCES.json',{'records':[{'completion_tokens':100}]})
                C.publish(root/model/'TASKS.json',{'records':[]})
                C.publish(root/model/'SUPPLEMENTARY.json',{'status':'complete'})
            C.publish(root/'REPORT_COMPLETE.json',{'complete':True})
            with patch('moe_exp.research_campaign_v3_r4.observations.pair_tasks',return_value=[]):
                self.assertEqual(O._saved_output_residual(campaign),7*(16*100+8192))
                for model in C.MODELS:C.publish(root/model/'TOKEN_OWNERSHIP.json',{'records':[]})
                self.assertEqual(O._saved_output_residual(campaign),0)

    def test_qualification_preserves_primary_secondary_cleanup_and_identity(self):
        from moe_exp.research_campaign_v3_r4.workers import qualify
        with tempfile.TemporaryDirectory(prefix='repair-qualifier-',dir=C.REPO/'report') as temp:
            campaign={'root':temp,'models':{'gpt':{'layers':[0,1]}}}
            def primary(*args,**kwargs):raise KeyError('original qualifier failure')
            def secondary():raise RuntimeError('secondary shutdown failure')
            backend=SimpleNamespace(llm=SimpleNamespace(collective_rpc=primary),close=secondary)
            shape={'key':'native','mask':None,'shard_index':1}
            with patch('moe_exp.research_campaign_v3_r4.workers.Backend',return_value=backend),patch(
                    'moe_exp.research_campaign_v3_r4.workers.qualification_rows',return_value={'records':[shape]}):
                result=qualify(campaign,'gpt',10**12,1)
            failure=result['failures'][0]
            self.assertEqual(failure['error_type'],'KeyError')
            self.assertEqual(failure['phase'],'native_context_and_compact_qualification')
            self.assertEqual(failure['assignment']['shape'],shape)
            self.assertEqual(failure['cleanup_errors'][0]['error'],'secondary shutdown failure')
            self.assertEqual(result['failure_scope'],'shared')

    def test_missing_first_model_inputs_do_not_stall_independent_models(self):
        with tempfile.TemporaryDirectory(prefix='repair-staged-missing-',dir=C.REPO/'report') as temp:
            root=Path(temp)
            campaign=C.seal({'root':str(root),'resume_from':str(root/'old'),'models':{m:{'layers':[0,1,2],'experts':8} for m in C.MODELS}})
            for model in C.MODELS:
                for stage in ('derive','native','mapping','supplementary'):
                    C.publish(root/'stages'/O.stage_key(stage,model)/'RESULT.json',{'status':'execution_failed' if model=='qwen330b' and stage=='derive' else 'complete','failure_scope':'model' if model=='qwen330b' and stage=='derive' else None})
            C.publish(root/'stages'/O.stage_key('resolve')/'RESULT.json',{'status':'complete'})
            C.publish(root/'SAVED_COORDINATION_ADMISSION.json',{'admitted':True})
            with patch('moe_exp.research_campaign_v3_r4.observations.pair_tasks',return_value=[]):
                work=O.ready(campaign,FakeScheduler(),[])
            self.assertEqual({c['model'] for c in work if c['gpus']},set(C.MODELS)-{'qwen330b'})
            self.assertFalse((root/'TERMINAL.json').exists())

    def test_first_wave_contains_only_qwen330b_and_model_failure_keeps_others(self):
        with tempfile.TemporaryDirectory(prefix='repair-staged-',dir=C.REPO/'report') as temp:
            root=Path(temp)
            campaign=C.seal({'root':str(root),'resume_from':str(root/'old'),'models':{m:{'layers':[0,1,2],'experts':8} for m in C.MODELS}})
            for model in C.MODELS:
                for stage in ('derive','native','mapping'):
                    C.publish(root/'stages'/O.stage_key(stage,model)/'RESULT.json',{'status':'complete'})
                C.publish(root/'stages'/O.stage_key('supplementary',model)/'RESULT.json',{'status':'complete'})
            C.publish(root/'stages'/O.stage_key('resolve')/'RESULT.json',{'status':'complete'})
            C.publish(root/'SAVED_COORDINATION_ADMISSION.json',{'admitted':True})
            with patch('moe_exp.research_campaign_v3_r4.observations.pair_tasks',return_value=[]):
                work=O.ready(campaign,FakeScheduler(),[])
                self.assertEqual([(c['stage'],c['model']) for c in work if c['gpus']],[('measure','qwen330b')])
                C.publish(root/'stages'/O.stage_key('measure','qwen330b')/'RESULT.json',{
                    'status':'execution_failed','failure_scope':'model','error':'CUDA out of memory'})
                work=O.ready(campaign,FakeScheduler(),[])
                self.assertEqual({c['model'] for c in work if c['gpus']},set(C.MODELS)-{'qwen330b'})
                self.assertFalse((root/'TERMINAL.json').exists())

    def test_mask_reference_preserves_execution_identity_and_shape(self):
        from moe_exp.research_campaign_v3_r4.masks import reference,resolve
        from moe_exp.research_campaign_v3_r4.workers import mask_shape
        with tempfile.TemporaryDirectory(prefix='repair-mask-',dir=C.REPO/'report') as temp:
            campaign={'root':temp,'models':{'gpt':{'layers':[0,1]}}}
            mask={'0':[0,1,3],'1':[1,2,4,5]}
            ref=reference(campaign,'gpt',mask)
            self.assertEqual(resolve(ref),mask)
            self.assertEqual(mask_shape(campaign,'gpt',mask),mask_shape(campaign,'gpt',ref))
            row=dict(zip(C.EXECUTION_FIELDS,['gpt','profile',{'prefix_tokens':10},1,'run-000',mask,None,90,'history']))
            self.assertEqual(C.execution_id(row),C.execution_id({**row,'mask':ref}))


if __name__=='__main__':unittest.main()
