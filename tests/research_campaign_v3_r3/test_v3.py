"""Regressions for actual v2 defects, native arithmetic and transitive evidence."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from moe_exp.research_campaign_v3_r3 import contracts as C,controller as O,scheduler as S
from moe_exp.research_campaign_v3_r3.numeric import read_parquet,normalize
from moe_exp.research_campaign_v3_r3.statistics import bootstrap_panel
from moe_exp.research_campaign_v3_r3.resume import ancestry,inherited_state
from moe_exp.research_campaign_v3_r3.export import _resident,_panels,contrast_bounds
from moe_exp.research_campaign_v3_r3.masks import reference,resolve
from test_campaign import FakeScheduler


class V3(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v3-regression-',dir=C.REPO/'report');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_actual_all_seven_inherited_parquet_roundtrip(self):
        import pandas as pd
        parent=C.REPO/'report/research-campaign-v2-20261009-r1'
        for model in C.MODELS:
            path=parent/model/'native_dynamics_estimands.parquet'
            self.assertTrue(path.exists(),model)
            # Bounded real fixture: retains NaN in actual serialized schema.
            import pyarrow.parquet as pq
            data=next(pq.ParquetFile(path).iter_batches(batch_size=12)).to_pandas()
            fixture=self.root/(model+'.parquet');data.to_parquet(fixture,index=False)
            rows=read_parquet(fixture)
            self.assertTrue(any(r.get('unavailable_reasons') for r in rows),model)
            columns=[k for k in data if k not in ('benchmark','family','attempt_id')]
            panel=bootstrap_panel(rows,columns,panel='test:actual:'+model,draws=32)
            json.dumps(panel,allow_nan=False)
            C.publish(self.root/(model+'.json'),{'rows':rows,'panel':panel})
    def test_strict_numeric_hashing_and_chunk_draw_identity(self):
        import pandas as pd
        values={'x':np.float64(np.nan),'y':float('inf'),'z':pd.NA}
        self.assertEqual(normalize(values),dict.fromkeys(values))
        self.assertEqual(C.digest(values),C.digest(dict.fromkeys(values)))
        rows=[{'benchmark':b,'family':str(i),'x':float(i),'y':float(i%3)} for b in ('a','b') for i in range(25)]
        a=bootstrap_panel(rows,['x','y'],panel='test:chunks',draws=1000,chunk_size=17)
        b=bootstrap_panel(rows,['x','y'],panel='test:chunks',draws=1000,chunk_size=256)
        self.assertEqual(a,b)
        zero=bootstrap_panel([{'benchmark':'a','family':'f','x':float('nan')}],['x'],panel='test:zero',draws=10)
        self.assertEqual(zero['results']['x']['family_support'],0)
    def test_ancestry_cycles_and_grandparent_started_assignments(self):
        v1=C.publish(self.root/'v1/CAMPAIGN.json',{'root':str(self.root/'v1'),'code_files':{},'source_files':{}})
        v2=C.publish(self.root/'v2/CAMPAIGN.json',{'root':str(self.root/'v2'),'resume_from':v1['root'],'code_files':{},'source_files':{}})
        v3={'root':str(self.root/'v3'),'resume_from':v2['root']}
        self.assertEqual([c['sha256'] for c in ancestry(v3)],[v2['sha256'],v1['sha256']])
        uid=C.digest('original');C.publish(self.root/'v1/gpt/executions'/f'{uid}.start-0.json',{'uid':uid,'status':'started'})
        for a,b in [('v3','v2'),('v2','v1')]:
            C.publish(self.root/a/'gpt/INHERITED_UNITS.json',{'directories':{'executions':str(self.root/b/'gpt/executions')},'compatible_statuses':['complete']})
        self.assertEqual(inherited_state(self.root/'v3/gpt/executions',uid)['status'],'started')
        (self.root/'v1/CAMPAIGN.json').write_text(json.dumps(C.seal({**v1,'resume_from':v2['root']})))
        with self.assertRaisesRegex(ValueError,'cycle'):ancestry(v3)
    def test_recursive_accounting_deduplicates_array_task_identities(self):
        scheduler=FakeScheduler()
        first=C.seal({'root':str(self.root/'v1'),'code_files':{},'source_files':{}})
        C.publish(self.root/'v1/CAMPAIGN.json',first)
        second=C.publish(self.root/'v2/CAMPAIGN.json',{'root':str(self.root/'v2'),'resume_from':first['root'],'code_files':{},'source_files':{}})
        third={'root':str(self.root/'v3'),'resume_from':second['root']}
        observed=[{'accepted':{'job_id':'123_0'},'accounting':{'billing_hours':2.}},
                  {'accepted':{'job_id':'123_1'},'accounting':{'billing_hours':3.}}]
        def jobs(campaign,scheduler):return [] if campaign is third else observed
        with patch.object(O,'jobs',jobs):self.assertEqual(O._inherited_actual(third,scheduler),5.)
    def test_referenced_mask_memory_fixture_and_mismatches(self):
        native={'layers':[1,3],'experts':4,'profile_sha256':'profile','card':{'max_model_len':64},'kv_cache_memory_bytes':1024}
        campaign=C.seal({'root':str(self.root),'models':{'gpt':native},'runtime_files':{}})
        mask={'1':[0,2],'3':[1,2,3]};ref=reference(campaign,'gpt',mask)
        self.assertEqual(resolve(ref),mask)
        with self.assertRaisesRegex(ValueError,'count'):resolve({**ref,'counts':[2,2]})
        result={'assignment':{'mask':ref},'memory_after':[]}
        self.assertIsNone(_resident(result,self.root,'gpt',campaign=campaign))
        reports=[{'cache_sha256':'fixed','memory':{'nvml_process_residence':[{'bytes':100}]}}]*2
        C.publish(self.root/'gpt/qualification'/(C.digest([2,3])+'.json'),{'status':'complete','physical_layers':[1,3],
            'physical_counts':[2,3],'profile_sha256':'profile','runtime_binding':C.digest({}),'full_context':64,'fixed_kv_bytes':1024,
            'fixture_cache_sha256':['fixed','fixed'],'fixture':{'memory_after':reports}})
        self.assertEqual(_resident(result,self.root,'gpt',campaign=campaign),200.)
        self.assertEqual(_resident({'assignment':{'mask':mask}},self.root,'gpt',campaign=campaign),200.)
        self.assertIsNone(_resident(result,self.root,'gpt',campaign={**campaign,'runtime_files':{'different':'binding'}}))
    def test_paired_memory_contrasts_and_asymmetric_bounds(self):
        self.assertAlmostEqual(contrast_bounds(.2,.6,.1,.9)[0],-.7)
        self.assertAlmostEqual(contrast_bounds(.2,.6,.1,.9)[1],.5)
        rows=[]
        for i in range(20):
            for method,removal,accuracy,lo,hi in [('native',0.,.5,.1,.9),('random',.25,i/30,.2,.6)]:
                rows.append({'study':'memory','model':'gpt','benchmark':'math500','family':str(i),'question':str(i),
                    'method':method,'allocation':'shared' if method=='native' else 'equal','removal':removal,
                    'strict_correct':accuracy,'unknown_lower':lo,'unknown_upper':hi,'new_tokens':i,'cumulative_tokens':i})
        panels=_panels(rows+[{**r,'model':'nemotron'} for r in rows])
        self.assertEqual(panels[0]['resampling_identity'],panels[1]['resampling_identity'])
        panel=panels[0]['results'];key='random:equal:0.25-minus-native|'
        self.assertAlmostEqual(panel[key+'unknown_lower']['estimate'],-.7)
        self.assertAlmostEqual(panel[key+'unknown_upper']['estimate'],.5)
        self.assertEqual(panel[key+'strict_correct']['family_support'],20)
    def test_exact_shared_derivation_failure_stops_qwen_and_other_gpus(self):
        campaign=C.seal({'root':str(self.root),'models':{m:{} for m in C.MODELS},'resume_from':str(self.root/'parent')})
        C.publish(self.root/'stages'/O.stage_key('resolve')/'RESULT.json',{'status':'complete'})
        for model in C.MODELS:
            C.publish(self.root/model/'SOURCES.json',{'records':[{'attempt_id':'original'}]})
            C.publish(self.root/'stages'/O.stage_key('derive',model)/'RESULT.json',{'status':'execution_failed','failure_scope':'shared','error':'Out of range float values are not JSON compliant'})
            C.publish(self.root/'stages'/O.stage_key('mapping',model)/'RESULT.json',{'status':'complete'})
        with patch('moe_exp.research_campaign_v3_r3.observations.pair_tasks',return_value=[]):
            work=O.ready(campaign,FakeScheduler(),[])
        self.assertFalse(any(c['gpus'] for c in work))
        self.assertFalse((self.root/'FIRST_MODEL_GATE.json').exists())
        self.assertEqual(C.load(self.root/'TERMINAL.json')['status'],'upstream_unavailable')
    def test_native_reduction_contributions_and_adapter_restoration(self):
        import torch
        from moe_exp.research_campaign_v3_r3.reap import measure
        class Experts:
            def moe_sum(self,x,out):out.copy_(x.sum(1))
        experts=Experts();kernel=SimpleNamespace(fused_experts=experts)
        method=SimpleNamespace(moe_kernel=kernel)
        class Routed:
            quant_method=method
            def forward_modular(self,x,w,ids):
                # Input-side weighting plus nonlinear activation disproves the
                # old external one-hot weighting reconstruction.
                outputs=(x[:,None,:]*w[:,:,None]*(ids[:,:,None]+1)).square()
                out=torch.empty_like(x);experts.moe_sum(outputs,out);return out
        runner=SimpleNamespace(routed_experts=Routed(),moe_config=SimpleNamespace(tp_size=1),_fused_output_is_reduced=True)
        norms,supported,error=measure(runner,torch.ones(2,3),torch.tensor([[.3,.7]]*2),torch.tensor([[0,1]]*2))
        self.assertTrue(supported);self.assertEqual(error,0.)
        self.assertAlmostEqual(float(norms[0,1]),4*np.sqrt(3),places=5)
        self.assertNotIn('moe_sum',vars(experts));self.assertIs(method.moe_kernel,kernel)
        with patch.object(Routed,'forward_modular',side_effect=KeyError('primary')):
            with self.assertRaises(KeyError):measure(runner,torch.ones(2,3),torch.ones(2,2),torch.tensor([[0,1]]*2))
        self.assertNotIn('moe_sum',vars(experts));self.assertIs(method.moe_kernel,kernel)

    def test_compact_extraction_parity_restart_and_cache_lifetime(self):
        import torch
        from moe_exp.research_campaign_v3_r3 import extraction as X
        from moe_exp.research_campaign_v3_r3.observations import _joint_counts
        source={'attempt_id':'a','source_location':{},'benchmark':'math500','family':'f','tensor_path':str(self.root/'raw')}
        (self.root/'raw').write_bytes(b'original')
        sources=C.publish(self.root/'gpt/SOURCES.json',{'records':[source]})
        C.publish(self.root/'gpt/TOKEN_OWNERSHIP.json',{'records':[]})
        campaign=C.seal({'root':str(self.root),'models':{'gpt':{'layers':[1,3,5],'experts':4,'native_k':2}}})
        ids=torch.tensor([[[i%4,(i+1+l)%4] for i in range(6)] for l in (1,2,1)])
        raw={'ids':ids,'weights':torch.ones_like(ids,dtype=torch.float32)/2,'reasoning_tokens':torch.arange(6)}
        classes=[np.array([0,1]),np.array([3,4])]+[np.array([],int)]*5
        nine=np.array([[2,2,0,0,0,0,0,6,4]])
        layout={'unit_tokens':[[0,1],[2],[3,4],[5]]}
        with patch.object(X,'validate_mapping'),patch.object(X,'counts_for',return_value=(None,nine)),patch.object(X,'archive',return_value=raw) as read,patch.object(X,'class_tokens',return_value=classes),patch.object(X,'read_source',return_value={}),patch('moe_exp.schemas.TraceRecord',return_value=None),patch('moe_exp.correlation_pipeline.dynamics.classes.saved_layout',return_value=layout):
            first=X.extract(campaign,'gpt');self.assertEqual(X.extract(campaign,'gpt'),first)
            self.assertEqual(read.call_count,1)
        compact,owned=X.selected(campaign,'gpt',source,[1,5])
        for c in (0,1):
            np.testing.assert_array_equal(_joint_counts(compact[1][owned[c]],compact[5][owned[c]],4),
                _joint_counts(np.asarray(ids)[0,classes[c]],np.asarray(ids)[2,classes[c]],4))
        with self.assertRaisesRegex(ValueError,'unfinished'):X.retire(campaign,'gpt')
        for i in range(1,3):C.publish(self.root/'gpt/coordination'/f'pair-{i:03d}'/'RESULT.json',{'status':'complete'})
        C.publish(self.root/'gpt/SUPPLEMENTARY.json',{'status':'complete'})
        with patch('moe_exp.research_campaign_v3_r3.observations.pair_tasks',return_value=[{'shard_index':1},{'shard_index':2}]):X.retire(campaign,'gpt')
        self.assertFalse((self.root/'gpt/compact').exists());self.assertTrue((self.root/'raw').exists())
        with self.assertRaisesRegex(ValueError,'retired'):X.extract(campaign,'gpt')

    def test_transformed_statistics_use_paired_family_inputs(self):
        rows=[{'benchmark':'a','family':str(i),'x':float(i),'y':float(i+1) if i<5 else None} for i in range(10)]
        panel=bootstrap_panel(rows,['x','y'],panel='test:paired-support',draws=1000,
            transform=(['difference'],lambda x:(x[:,0]-x[:,1])[:,None],[[0,1]]))
        self.assertEqual(panel['results']['difference']['family_support'],5)
        self.assertAlmostEqual(panel['results']['difference']['estimate'],-1.)
        panel=bootstrap_panel(rows,['x'],panel='test:benchmark-support',draws=100,benchmarks=['a','b'])
        self.assertIsNone(panel['results']['x']['estimate'])
        self.assertEqual(panel['results']['x']['benchmark_family_support']['b'],0)

    def test_recursive_import_keeps_valid_bank_beside_invalid_reap(self):
        from moe_exp.research_campaign_v3_r3.resume import import_evidence
        from moe_exp.research_campaign_v3_r3.workers import measurement_rows
        v1root=self.root/'v1';v2root=self.root/'v2';v3root=self.root/'v3'
        model={'profile_sha256':'profile','layers':[1],'experts':4,'native_k':2}
        v1=C.publish(v1root/'CAMPAIGN.json',{'root':str(v1root),'models':{'gpt':model},'code_files':{},'source_files':{}})
        v2=C.publish(v2root/'CAMPAIGN.json',{'root':str(v2root),'resume_from':str(v1root),'models':{'gpt':model},
            'code_files':{},'source_files':{},'instrumentation_binding':'router-v2'})
        v3=C.seal({'root':str(v3root),'resume_from':str(v2root),'models':{'gpt':model},'instrumentation_binding':'router-v2','reap_binding':'native-boundary-v3'})
        for name in ('FOLDS','COHORTS','REUSE','REUSED_CATALOGUE'):C.publish(v1root/(name+'.json'),{'records':[]})
        source={'attempt_id':'saved','completion_tokens':4,'family':'f','benchmark':'math500'}
        sources=C.publish(v1root/'gpt/SOURCES.json',{'records':[source]});prefix=C.publish(v1root/'gpt/PREFIXES.json',{'records':[]})
        uid=C.digest(['measurement','gpt','saved','profile','router-v2']);row={'uid':uid,'model':'gpt','source':source,'task_index':0}
        bank=v2root/'gpt/banks/bank.npz';bank.parent.mkdir(parents=True);np.savez(bank,scores=np.ones((2,4)))
        units=C.Units(v2root/'gpt/measurements');units.begin(row,'old');units.commit(row,{'status':'complete',
            'artifact':{'path':str(bank),'sha256':C.file_sha(bank),'bytes':bank.stat().st_size},'reap_supported':False})
        C.publish(v1root/'stages'/O.stage_key('derive','gpt')/'RESULT.json',{'status':'execution_failed','error':'retained'})
        imported=import_evidence(v3)
        components=C.load(v3root/'MEASUREMENT_COMPONENTS.json')['records']
        self.assertTrue(components[0]['router_bank_qualified']);self.assertFalse(components[0]['reap_qualified'])
        replacement=measurement_rows(v3,'gpt')['records'][0]
        self.assertNotEqual(replacement['uid'],uid);self.assertEqual(replacement['original_uid'],uid)
        self.assertEqual(replacement['supersession']['superseded_component'],'REAP norms')
        self.assertTrue(imported['failed_receipts'])
        self.assertEqual(C.Units(v3root/'gpt/measurements').state(uid)['status'],'complete')

    def test_failed_cache_has_durable_unavailable_dependencies(self):
        from moe_exp.research_campaign_v3_r3.extraction import retire_failed
        campaign=C.seal({'root':str(self.root)})
        C.publish(self.root/'gpt/compact/OWNER.json',{'campaign_sha256':campaign['sha256'],'model':'gpt'})
        (self.root/'gpt/compact/partial.npy').write_bytes(b'temporary')
        (self.root/'original.capture').write_bytes(b'preserved')
        failure={'status':'execution_failed','failure_scope':'model','original_exception':'model-specific data limitation'}
        with patch('moe_exp.research_campaign_v3_r3.observations.pair_tasks',return_value=[{'shard_index':1}]):
            retire_failed(campaign,'gpt',failure)
        self.assertFalse((self.root/'gpt/compact').exists())
        self.assertTrue((self.root/'original.capture').exists())
        self.assertEqual(C.load(self.root/'gpt/coordination/pair-001/RESULT.json')['status'],'upstream_unavailable')

    def test_sealed_graph_has_exact_original_pair_geometry(self):
        from moe_exp.research_campaign_v3_r3.inputs import dependency_graph
        from moe_exp.research_campaign_v3_r3.observations import pair_tasks
        parent=C.load(C.REPO/'report/research-campaign-v2-20261009-r1/CAMPAIGN.json')
        graph=dependency_graph(parent)
        for model in C.MODELS:
            self.assertEqual(graph['registered_rows'][model]['coordination_pairs'],pair_tasks(parent,model))
        self.assertEqual(sum(len(r['coordination_pairs']) for r in graph['registered_rows'].values()),899)
        self.assertIn('reader_price',C.STAGES['admit'])

    def test_referenced_mask_through_terminal_export(self):
        import pandas as pd
        from moe_exp.research_campaign_v3_r3 import export as E
        native={'layers':[1,3],'experts':4,'profile_sha256':'profile','card':{'max_model_len':64},'kv_cache_memory_bytes':1024}
        campaign=C.seal({'root':str(self.root),'campaign_id':'synthetic-reference','models':{m:native for m in C.MODELS},
            'runtime_files':{},'annotator':{},'preserved_scientific_files':{},'preserved_operational_files':{}})
        ref=reference(campaign,'gpt',{'1':[0,2],'3':[1,2,3]});uid=C.digest('original-pruned')
        task={'uid':uid,'mask':ref};units=C.Units(self.root/'gpt/executions');units.begin(task,'original')
        units.commit(task,{'status':'complete','generated_tokens':3})
        scores=C.Units(self.root/'gpt/scores');scores.begin(task,'score');scores.commit(task,{
            'status':'complete','strict_correct':1.,'unknown_lower':1.,'unknown_upper':1.,'new_tokens':3,'cumulative_tokens':3})
        reports=[{'cache_sha256':'fixed','memory':{'nvml_process_residence':[{'bytes':100}]}}]*2
        C.publish(self.root/'gpt/qualification'/(C.digest([2,3])+'.json'),{'status':'complete','physical_layers':[1,3],
            'physical_counts':[2,3],'profile_sha256':'profile','runtime_binding':C.digest({}),'full_context':64,
            'fixed_kv_bytes':1024,'fixture_cache_sha256':['fixed','fixed'],'fixture':{'memory_after':reports}})
        cell={'uid':uid,'study':'memory','model':'gpt','status':'pending','benchmark':'math500','family':'f','question':'q',
            'method':'random','allocation':'equal','removal':.25}
        def planned(c,m):return {'records':[task] if m=='gpt' else [],'cells':[cell] if m=='gpt' else []}
        with patch.object(E,'_planned',planned),patch.object(E,'_figures',return_value=[]),patch.object(E,'_observational_figures',return_value=[]),patch.object(O,'status',return_value={'actual_local_hours':0.,'terminal':None}):
            E.closeout(campaign)
        data=pd.read_parquet(self.root/'output/cells.parquet')
        self.assertEqual(float(data.resident_bytes.iloc[0]),200.)
        self.assertTrue(C.load(self.root/'REPORT_COMPLETE.json')['complete'])

    def test_generated_paper_exhibits_bind_nested_pdf_figures(self):
        from moe_exp.research_campaign_v3_r3 import decisions as D
        root=self.root/'report/research-campaign-v3-fixture';directory=root/'output'
        figure=directory/'figures/effect.pdf';figure.parent.mkdir(parents=True);figure.write_bytes(b'vector fixture')
        campaign=C.seal({'root':str(root)});report=C.seal({'SCIENCE_COMPLETE':False,'snapshot':False,'cells':1})
        with patch.object(D,'REPO',self.root):D.paper_exhibits(campaign,directory,report)
        exhibit=self.root/'paper/6a510c6db0d9b08aa438c142/generated/research_campaign_v3/output/figures/effect.pdf'
        self.assertEqual(C.file_sha(exhibit),C.file_sha(figure))
        binding=C.load(exhibit.parents[1]/'EXHIBITS_BINDING.json')
        self.assertIn(str(exhibit),binding['files'])

    def test_actual_pending_slurm_one_node_range_readback(self):
        campaign=C.seal({'root':str(self.root),'models':{}})
        card=S.card(campaign,'resolve',hold=True,minutes=180)
        # Native readback of held 59900049 used NumNodes=1-1. It had
        # ReqTRES node=1, four billed CPUs and no allocated resources.
        raw=('JobId=59900049 UserId=lmolfett(133943) Account=iscrc_miosr '
             'Partition=lrd_all_serial Comment='+card['comment']+
             ' Reason=JobHeldUser NumNodes={nodes} NumCPUs=4 '
             'ReqTRES=cpu=4,mem=24G,node=1,billing=4,gres/tmpfs=10737418240 AllocTRES=(null)')
        scheduler=S.Slurm()
        for nodes in ('1','1-1'):
            with patch.object(scheduler,'command',return_value=raw.format(nodes=nodes)):
                self.assertEqual(scheduler.verify('59900049',card)['resources']['allocation_cpus'],4)
        for nodes in ('2','1-2','2-2'):
            with patch.object(scheduler,'command',return_value=raw.format(nodes=nodes)):
                with self.assertRaisesRegex(ValueError,'node count'):scheduler.verify('59900049',card)

if __name__=='__main__':unittest.main()
