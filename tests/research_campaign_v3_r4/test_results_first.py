"""Actual-roster and adversarial delivery regressions for the r4 continuation."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r4 import contracts as C,controller as O,export as E
from moe_exp.research_campaign_v3_r4.assignments import balanced
from moe_exp.research_campaign_v3_r4.phases import assign,ready
from moe_exp.research_campaign_v3_r4.responses import semantic_response


class ResultsFirst(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.REPO/'report',prefix='r4-results-test-')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_actual_family_overlay_is_balanced_and_model_independent(self):
        old=C.REPO/'report/research-campaign-v3-20261011-r3'
        cohort=C.load(old/'COHORTS.json')['causal']
        rows=balanced(cohort);lookup={(r['benchmark'],r['family']):r['configuration_indices'] for r in rows}
        self.assertEqual(len(lookup),129)
        overall=Counter(v[1] for v in lookup.values())
        self.assertEqual(sorted(overall.values()),[16]*7+[17])
        for b in C.BENCHMARKS:
            count=Counter(v[1] for (benchmark,_),v in lookup.items() if benchmark==b)
            self.assertLessEqual(max(count.values())-min(count.get(i,0) for i in range(1,9)),1)
        shuffled=balanced(list(reversed(cohort)))
        self.assertEqual(lookup,{(r['benchmark'],r['family']):r['configuration_indices'] for r in shuffled})
        for model in C.MODELS:
            prefixes=C.load(old/model/'PREFIXES.json')['records']
            self.assertEqual(set(lookup),{(r['benchmark'],r['family']) for r in prefixes})

    def test_phase_aliases_preserve_execution_identities(self):
        rows=[{'uid':'same'},{'uid':'diagnostic'}]
        cells=[{'study':'causal','uid':'same','threshold_index':0},
               {'study':'causal','uid':'same','threshold_index':1},
               {'study':'causal','uid':'diagnostic','threshold_index':2}]
        value=assign({'records':rows,'cells':cells})
        self.assertEqual(value['records'][0]['phase'],'central')
        self.assertEqual(value['records'][0]['analysis_phases'],['central','sensitivity'])
        self.assertEqual([r['uid'] for r in value['records']],['same','diagnostic'])

    def test_memory_export_precedes_causal_completion(self):
        campaign={'root':str(self.root),'models':{'gpt':{}}}
        C.publish(self.root/'gpt/EXECUTION_SHARDS.json',{'shards':[[],[]],'phases':['memory','central']})
        C.publish(self.root/'gpt/TASKS.json',{'records':[],'cells':[]})
        C.publish(self.root/'stages'/O.stage_key('execute','gpt',0)/'RESULT.json',{'status':'complete'})
        calls=[]
        self.assertFalse(ready(campaign,[],lambda stage,model='all',shard=0,**kw:calls.append((stage,model,shard))))
        self.assertIn(('phase_export','all',1),calls)
        self.assertNotIn(('execute','gpt',1),calls)

    def test_failed_closeout_has_one_recovery_and_stops(self):
        campaign=C.seal({'root':str(self.root)})
        C.publish(self.root/'stages'/O.stage_key('closeout')/'RESULT.json',{'status':'execution_failed'})
        first=O.ready(campaign,None,[])
        self.assertEqual([(c['stage'],c['shard_index']) for c in first],[('closeout',1)])
        C.publish(self.root/'stages'/O.stage_key('closeout','all',1)/'RESULT.json',{'status':'execution_failed'})
        self.assertEqual(O.ready(campaign,None,[]),[])
        self.assertEqual(C.load(self.root/'OPERATIONAL_FAILURE.json')['export_only_recoveries'],1)
        self.assertEqual(O.ready(campaign,None,[]),[])

    def test_durable_report_survives_paper_delivery_failure(self):
        campaign={'root':str(self.root)};report={'sha256':'frozen','SCIENCE_COMPLETE':False}
        out=self.root/'output';out.mkdir()
        with patch('moe_exp.research_campaign_v3_r4.decisions.paper_exhibits',side_effect=RuntimeError('publication interruption')):
            with self.assertRaises(RuntimeError):E.deliver(campaign,out,report,final=True)
        self.assertTrue((self.root/'REPORT_COMPLETE.json').exists())
        self.assertFalse((self.root/'PAPER_DELIVERY.json').exists())
        with patch('moe_exp.research_campaign_v3_r4.decisions.paper_exhibits',return_value=None):
            self.assertTrue(E.deliver(campaign,out,report,final=True)['PAPER_DELIVERED'])

    def test_window_clips_tokens_gaps_break_runs_and_native_uses_same_pulse(self):
        labels={'a':'Verify','b':None,'c':'Verify'}
        annotations=type('Labels',(),{'state':lambda _,uid:{'label':labels[uid]}})()
        context={'sentences':[{'annotation_uid':uid,'token_start':a,'token_end':b,
            'sentence_index':i,'segment':0,'first_complete':i==0,'n_tokens':b-a}
            for i,(uid,a,b) in enumerate([('a',0,6),('b',6,9),('c',9,16)])]}
        result={'status':'complete','generated_tokens':16,'finish_reason':'length'}
        cell={'target':'Verify','windows':[4,8,12],'analysis_pulse_tokens':4,'prefix':{'source_class':'Plan'}}
        a,_=semantic_response({'window':16,'action':{'pulse_tokens':4}},result,context,annotations,cell)
        n,_=semantic_response({'window':16,'action':None},result,context,annotations,cell,
            analysis_pulse_tokens=a['analysis_pulse_tokens'],analysis_windows=a['analysis_windows'])
        self.assertEqual(a['recovery_lower:w2:Verify'],5/8)
        self.assertEqual(a['recovery_upper:w2:Verify'],1)
        self.assertEqual(n['recovery_lower:w2:Verify'],a['recovery_lower:w2:Verify'])
        self.assertEqual(a['restricted_target_dwell_tokens'],3)
        self.assertEqual(a['target_reentry_count'],0)
        self.assertGreater(a['postpulse_trajectory_gaps'],0)
        self.assertEqual(a['first_postpulse_matches_source'],0)

    def test_stopping_truncation_and_missing_execution_are_distinct(self):
        labels=type('Labels',(),{'state':lambda _,uid:None})()
        cell={'target':'Verify','windows':[4,8,16],'analysis_pulse_tokens':4}
        values=[]
        for result in ({'status':'complete','generated_tokens':2,'finish_reason':'stop'},
                       {'status':'complete','generated_tokens':2,'finish_reason':'length'},None):
            value,_=semantic_response({'window':16},result,None,labels,cell);values.append(value)
        self.assertEqual(values[0]['recovery_upper:w2:Verify'],0)
        self.assertEqual(values[1]['recovery_upper:w2:Verify'],1)
        self.assertEqual(values[2]['recovery_upper:w2:Verify'],1)
        self.assertEqual(len({r['recovery_status:w2'] for r in values}),3)

    def test_primary_declares_complete_roster_with_all_benchmarks(self):
        from moe_exp.research_campaign_v3_r4.primary import central_panels
        cohort=[{'benchmark':'math500','family':'f','question':'q'}]
        C.publish(self.root/'COHORTS.json',{'causal':cohort})
        def panel(rows,columns,**kw):
            self.assertEqual(tuple(kw['benchmarks']),C.BENCHMARKS)
            return {'results':{c:{'estimate':None,'interval':None} for c in columns}}
        rows=[{'study':'causal','benchmark':'math500','family':'f','question':'q','model':'gpt',
               'target':'Verify','depth_third':0,'sign':1,'arm':'target','threshold_index':0}]
        with patch('moe_exp.research_campaign_v3_r4.primary.bootstrap_panel',side_effect=panel):
            p=central_panels(rows,{'root':str(self.root)})[0]
        self.assertEqual(p['declared_contrasts'],1176)
        self.assertEqual(p['bound_components'],2352)
        self.assertEqual(len(p['identified_effects']),1176)

    def test_separate_quotas_cannot_substitute_for_each_other(self):
        from moe_exp.research_campaign_v3_r4.scheduler import admission
        snapshot={'remaining_local_hours':42050,'consumed_local_hours':25950,
            'storage_headroom_bytes':C.STORAGE_RESERVE+99,'fast_storage_headroom_bytes':10**12}
        result=admission(snapshot,baseline_consumed=25950,campaign_actual=0,outstanding=0,
            residual=1,output_bytes=100,fast_output_bytes=100)
        self.assertFalse(result['admitted'])
        self.assertFalse(result['storage_quotas_pooled'])

    def test_compact_price_can_use_its_fixture_before_aggregate(self):
        from moe_exp.research_campaign_v3_r4.workers import execute_rows,mask_shape
        campaign={'root':str(self.root),'models':{'gpt':{'layers':[0]}}}
        row={'uid':C.digest('original'),'mask':{'0':[0,1]},'action':None}
        shape=mask_shape(campaign,'gpt',row['mask'])
        C.publish(self.root/'gpt/qualification'/(shape+'.json'),{'status':'complete'})
        with patch('moe_exp.research_campaign_v3_r4.workers.Backend',side_effect=RuntimeError('initialization observed')) as backend,patch.dict('os.environ',{'SLURM_JOB_ID':'test'}):
            with self.assertRaisesRegex(RuntimeError,'initialization observed'):
                execute_rows(campaign,'gpt',[row],0)
        backend.assert_called_once()
        self.assertEqual(C.Units(self.root/'gpt/executions').state(row['uid'])['status'],'upstream_unavailable')

    def test_successive_exhibits_keep_memory_and_central_verified(self):
        from moe_exp.research_campaign_v3_r4 import decisions as D
        fake_repo=self.root/'repo';root=fake_repo/'report/research-campaign-v3-test'
        campaign=C.seal({'root':str(root),'campaign_id':root.name})
        base=fake_repo/'paper/6a510c6db0d9b08aa438c142/generated/research_campaign_v3'
        bindings=[]
        for phase,figure in [('memory','gpt-memory.pdf'),('central','gpt-central-primary.pdf'),('sensitivity',None)]:
            out=root/'deliveries'/phase;out.mkdir(parents=True)
            if figure:
                (out/'figures').mkdir();(out/'figures'/figure).write_bytes(phase.encode())
            report=C.publish(out/'REPORT.json',{'phase':phase,'snapshot':True,'SCIENCE_COMPLETE':False,'cells':1})
            with patch.object(D,'REPO',fake_repo):
                binding=D.paper_exhibits(campaign,out,report)
                self.assertEqual(binding,D.paper_exhibits(campaign,out,report))
            bindings.append(binding)
        text=(base/'main_results.tex').read_text()
        self.assertIn('gpt-memory.pdf',text);self.assertIn('gpt-central-primary.pdf',text)
        self.assertEqual(text.count('gpt-memory.pdf'),1)
        for binding in bindings:C.check_files(binding['files'])

    def test_post_final_snapshot_does_not_import_old_delivery_receipt(self):
        campaign={'root':str(self.root)}
        C.publish(self.root/'REPORT_COMPLETE.json',{'complete':True})
        C.publish(self.root/'output/REPORT.json',{'SCIENCE_COMPLETE':False})
        C.publish(self.root/'output/PAPER_DELIVERY.json',{'old_final_receipt':True})
        def deliver(campaign,directory,report,**kwargs):
            self.assertFalse((directory/'PAPER_DELIVERY.json').exists())
            return {'status':'complete'}
        with patch.object(E,'deliver',side_effect=deliver),patch.object(O,'status',return_value={}):
            self.assertEqual(E.closeout(campaign,final=False)['status'],'complete')

    def test_sensitivity_does_not_replace_central_panels(self):
        from moe_exp.research_campaign_v3_r4.primary import central_panels,supporting_panels
        C.publish(self.root/'COHORTS.json',{'causal':[{'benchmark':'math500','family':'f','question':'q'}]})
        rows=[{'study':'causal','threshold_index':1,'arm':'target'}]
        self.assertEqual(central_panels(rows,{'root':str(self.root)}),[])
        self.assertEqual(supporting_panels(rows,{'root':str(self.root)}),[])

    def test_native_cells_keep_both_questions_in_one_family(self):
        campaign={'root':str(self.root)}
        cohort=[{'benchmark':'math500','family':'f','question':q} for q in ('q1','q2')]
        C.publish(self.root/'COHORTS.json',{'causal':cohort,'pruning':[]})
        C.publish(self.root/'gpt/PREFIXES.json',{'records':[]})
        cells=[{'study':'causal','model':'gpt',**row,'arm':'native','uid':C.digest(row['question'])} for row in cohort]
        C.publish(self.root/'gpt/NATIVE_TASKS.json',{'records':[],'cells':cells})
        with patch('moe_exp.research_campaign_v3_r4.assignments.roster',return_value=[{**r,'configuration_indices':[0,1]} for r in cohort]):
            table=E._planned(campaign,'gpt')
        native=[r for r in table['cells'] if r.get('arm')=='native']
        self.assertEqual({r['question']:r['uid'] for r in native},{r['question']:C.digest(r['question']) for r in cohort})

    def test_storage_prices_scientific_receipts_separately_from_handoffs(self):
        from moe_exp.research_campaign_v3_r4 import storage
        parent=self.root/'parent'
        (parent/'cards').mkdir(parents=True);(parent/'cards/one.json').write_bytes(b'x'*100)
        (parent/'stages/coordination').mkdir(parents=True)
        (parent/'stages/coordination/RESULT.json').write_bytes(b'x'*2000000)
        root=self.root/'research-campaign-v3-storage-test'
        with patch.object(storage,'FAST_PRIVATE',self.root/'fast'):
            policy=storage.initialize(root,{'root':str(parent)})
        self.assertEqual(policy['metadata_bytes_per_handoff_upper'],400)
        self.assertEqual(policy['historical_max_scientific_stage_receipt_bytes'],2000000)
        self.assertLess(policy['work_metadata_bound_bytes'],10*1024**3)
