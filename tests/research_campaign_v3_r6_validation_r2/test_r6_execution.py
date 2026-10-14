"""Behavioral boundaries exposed by the actual r5 reader and export failures."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from moe_exp.research_campaign_v3_r6 import contracts as C
from moe_exp.research_campaign_v3_r6.reader_contract import token_ids, ReaderUnits


class Tokenizer:
    def apply_chat_template(self,conversation,**kw):
        return [1,2,3] if kw.get('return_dict') is False else {'input_ids':[1,2,3]}


CONFIG={'enable_thinking':True,'preserve_thinking':False,'reasoning_effort':'low',
        'max_tokens':16,'batch_size':64,'judge_model':'original','program_sha256':'unchanged'}


class R6(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.REPO/'report',prefix='r6-test-')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_token_helper_requires_integer_ids_and_explicit_list(self):
        self.assertEqual(token_ids(Tokenizer(),[],CONFIG),[1,2,3])
        class Bad:
            def apply_chat_template(self,*a,**k):return {'input_ids':[1,2,3]}
        with self.assertRaisesRegex(ValueError,'integer token IDs'):token_ids(Bad(),[],CONFIG)
        class Strings:
            def apply_chat_template(self,*a,**k):return ['1','2']
        with self.assertRaisesRegex(ValueError,'integer token IDs'):token_ids(Strings(),[],CONFIG)

    def test_proof_does_not_override_a_new_started_or_committed_draw(self):
        row={'uid':C.digest('original'),'model':'nemotron','inputs':{}}
        original=C.publish(self.root/'old-result.json',{'status':'execution_failed'})
        C.publish(self.root/'READER_RECOVERY_ADJUDICATION.json',{'records':[{
            'uid':row['uid'],'model':'nemotron','recovery_allowed':True,
            'original_result':str(self.root/'old-result.json'),
            'original_result_sha256':C.file_sha(self.root/'old-result.json')}]})
        units=ReaderUnits(self.root/'nemotron/annotations')
        self.assertTrue(units.begin(row,'original-correction'))
        self.assertFalse(units.begin(row,'second-correction'))
        units.commit(row,{'status':'execution_failed','unknown':True})
        self.assertFalse(units.begin(row,'redraw'))
        self.assertEqual(C.load(self.root/'old-result.json'),original)

    def test_reader_records_admission_and_output_and_never_redraws_failed_admission(self):
        from moe_exp.research_campaign_v3_r6.reader_engine import annotate
        rows=[{'uid':C.digest(i),'model':'nemotron','inputs':{}} for i in range(2)]
        campaign={'root':str(self.root),'code_files':{},'annotator':{
            'path':'bound','config':CONFIG,'native_context':64,'decoding':{},
            'revision':'same','engine_changes':[]}}
        class Renderer:
            def __init__(self,*a):pass
            def request(self,*a,**k):return {'conversation':[]}
            def close(self):pass
        class InprocClient:
            def shutdown(self):pass
        class Engine:
            def __init__(self):self.engine_core=InprocClient();self.pending={};self.adds=0
            def has_unfinished_requests(self):return bool(self.pending)
            def add_request(self,uid,prompt,params):
                self.adds+=1
                if self.adds==2:raise RuntimeError('engine admission lost')
                self.pending[uid]=prompt['prompt_token_ids']
            def abort_request(self,*a,**k):self.pending.clear()
        engine=Engine()
        llm=SimpleNamespace(llm_engine=engine,get_tokenizer=lambda:Tokenizer())
        with patch('moe_exp.research_campaign_v3_r6.reader_engine.allocated'), \
             patch('moe_exp.research_campaign_v3_r6.annotation.Renderer',Renderer), \
             patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
             patch('vllm.LLM',return_value=llm), \
             patch('vllm.SamplingParams',side_effect=lambda **k:SimpleNamespace(**k)), \
             patch.dict(os.environ,{'SLURM_JOB_ID':'original'}):
            with self.assertRaisesRegex(RuntimeError,'admission lost'):annotate(campaign,'nemotron',rows,10**12)
            self.assertEqual(annotate(campaign,'nemotron',rows,10**12)['status'],'reused')
        self.assertEqual(engine.adds,2)
        self.assertTrue((self.root/'nemotron/annotations/lifecycle'/rows[0]['uid']/'ADMITTED.json').exists())
        self.assertFalse((self.root/'nemotron/annotations/lifecycle'/rows[1]['uid']/'ADMITTED.json').exists())
        self.assertTrue(all(ReaderUnits(self.root/'nemotron/annotations').state(r['uid'])['unknown'] for r in rows))

    def test_reader_failure_blocks_only_reader_capabilities(self):
        from moe_exp.research_campaign_v3_r6.failure_domains import record,blocked
        campaign={'root':str(self.root)}
        record(campaign,{'stage':'reader_price','model':'all','status':'execution_failed','failure_scope':'shared'})
        self.assertIsNotNone(blocked(campaign,'annotate','gpt'))
        self.assertIsNone(blocked(campaign,'coordination','qwen35'))
        self.assertIsNone(blocked(campaign,'memory_qualify','gpt'))

    def test_completed_raw_outputs_recover_after_parser_interruption_without_model_reload(self):
        from moe_exp.research_campaign_v3_r6.reader_engine import annotate
        rows=[{'uid':C.digest(['raw',i]),'model':'nemotron','inputs':{}} for i in range(2)]
        campaign={'root':str(self.root),'code_files':{},'annotator':{
            'path':'bound','config':CONFIG,'native_context':64,'decoding':{},'revision':'same','engine_changes':[]}}
        class Renderer:
            def __init__(self,*a):pass
            def request(self,*a,**k):return {'conversation':[]}
            def close(self):pass
        class InprocClient:
            def shutdown(self):pass
        class Engine:
            def __init__(self):self.engine_core=InprocClient();self.pending={};self.adds=0
            def has_unfinished_requests(self):return bool(self.pending)
            def add_request(self,uid,prompt,params):self.adds+=1;self.pending[uid]=prompt['prompt_token_ids']
            def abort_request(self,*a,**k):self.pending.clear()
            def step(self):
                out=[SimpleNamespace(request_id=k,finished=True,prompt_token_ids=v,
                    outputs=[SimpleNamespace(token_ids=[4],finish_reason='stop')]) for k,v in self.pending.items()]
                self.pending.clear();return out
        engine=Engine();llm=SimpleNamespace(llm_engine=engine,get_tokenizer=lambda:Tokenizer())
        parsed={'status':'complete','label':'Read','unknown':False,'seconds':1.,'generated_tokens':1}
        with patch('moe_exp.research_campaign_v3_r6.reader_engine.allocated'), \
             patch('moe_exp.research_campaign_v3_r6.annotation.Renderer',Renderer), \
             patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
             patch('vllm.LLM',return_value=llm) as loads, \
             patch('vllm.SamplingParams',side_effect=lambda **k:SimpleNamespace(**k)), \
             patch.dict(os.environ,{'SLURM_JOB_ID':'original'}):
            with patch('moe_exp.research_campaign_v3_r6.reader_engine.parse_output',side_effect=ValueError('parser interrupted')):
                with self.assertRaisesRegex(ValueError,'parser interrupted'):annotate(campaign,'nemotron',rows,10**12)
            self.assertTrue(all((self.root/'nemotron/annotations/lifecycle'/r['uid']/'OUTPUT.json').exists() for r in rows))
            with patch('moe_exp.research_campaign_v3_r6.reader_engine.parse_output',return_value=parsed):
                self.assertEqual(annotate(campaign,'nemotron',rows,10**12)['status'],'reused')
            self.assertEqual(loads.call_count,1)
        self.assertEqual(engine.adds,2)
        self.assertTrue(all(ReaderUnits(self.root/'nemotron/annotations').state(r['uid'])['status']=='complete' for r in rows))

    def test_index_preserves_native_and_unknown_state_semantics(self):
        from moe_exp.research_campaign_v3_r6.receipt_index import IndexedUnits
        row={'uid':C.digest('native')};u=C.Units(self.root/'gpt/executions')
        u.begin(row,'one');u.commit(row,{'status':'complete','generated_tokens':0})
        other={'uid':C.digest('interrupted')};u.begin(other,'two')
        index=IndexedUnits(u.root)
        for uid in [row['uid'],other['uid'],C.digest('absent')]:self.assertEqual(u.state(uid),index.state(uid))

    def test_index_preserves_legacy_string_ancestry_and_first_interrupted_owner(self):
        from moe_exp.research_campaign_v3_r6.receipt_index import IndexedUnits
        row={'uid':C.digest('inherited')};old=C.Units(self.root/'old/executions')
        old.begin(row,'first');old.commit(row,{'status':'complete','generated_tokens':1})
        C.publish(self.root/'new/INHERITED_UNITS.json',{'directories':{'executions':str(old.root)},
            'compatible_statuses':['complete','reused','invalid','truncated']})
        current=C.Units(self.root/'new/executions')
        self.assertEqual(current.state(row['uid']),IndexedUnits(current.root).state(row['uid']))

    def test_measurement_forecast_never_commits_shards_even_after_admission(self):
        from moe_exp.research_campaign_v3_r6 import controller as O
        rows=[{'uid':C.digest(i),'task_index':i} for i in range(3)]
        C.publish(self.root/'gpt/MEASURE_TASKS.json',{'records':rows})
        C.publish(self.root/'PREPARATION_ADMISSION.json',{'admitted':True})
        fake=SimpleNamespace(state=lambda uid: {'status':'complete','uid':uid,'seconds':2.,
            'load_seconds':1.,'artifact':{'bytes':10}} if uid==rows[0]['uid'] else None)
        with patch.object(O,'MODELS',('gpt',)),patch.object(O,'Units',return_value=fake):
            O._measurement_forecast({'root':str(self.root)})
        self.assertFalse((self.root/'gpt/MEASURE_SHARDS.json').exists())

    def test_isolated_index_publishes_ignored_tex_without_touching_user_index(self):
        from moe_exp.research_campaign_v3_r6 import publication as P
        repo=self.root/'checkout';remote=self.root/'remote';stage=self.root/'staging'
        repo.mkdir();stage.mkdir()
        def git(*args,cwd=repo):
            return subprocess.run(['git',*args],cwd=cwd,capture_output=True,text=True,check=True).stdout.strip()
        git('init');git('checkout','-b','analysis/routing-dynamics')
        git('config','user.name','Validation');git('config','user.email','validation@example.invalid')
        (repo/'.gitignore').write_text('generated/\n');(repo/'unchanged.txt').write_text('original')
        git('add','.');git('commit','-m','original');git('init','--bare',str(remote));git('remote','add','origin',str(remote))
        git('push','origin','HEAD:analysis/routing-dynamics')
        (repo/'unchanged.txt').write_text('local edit');before=git('status','--porcelain')
        snapshot=self.root/'snapshot';(snapshot/'generated').mkdir(parents=True)
        f=snapshot/'generated/required.tex';f.write_text('bound source')
        local={'report_sha256':C.digest('report'),'phase':'memory','snapshot':str(snapshot),
               'files':{'generated/required.tex':C.file_sha(f)}}
        real=P.command
        def command(args,cwd,env=None):
            if args[:4]==['git','remote','get-url','origin']:return 'https://github.com/disi-unibo-nlp/mfa-moe.git'
            return real(args,cwd,env)
        with patch.object(P,'REPO',repo),patch.object(P,'STAGING',stage),patch.object(P,'command',side_effect=command):
            first=P.push_snapshot({'root':str(self.root)},local,'github')
            second=P.push_snapshot({'root':str(self.root)},local,'github')
        self.assertEqual(first['commit'],second['commit']);self.assertTrue(second['reconciled_existing_delivery'])
        self.assertEqual(git('status','--porcelain'),before)
        self.assertFalse((repo/'generated').exists())
