"""Bounded J1 pricing, credential isolation and frozen-vote validation tests."""
from pathlib import Path
from types import SimpleNamespace
from collections import UserDict
import copy
import contextlib
import io
import os
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/experimental_resume')]
import utility_j1_chain_v1 as j


def settings():
    return {'history':{'median_prompt_tokens_per_second':2174.25,'minimum_positive_prompt_tokens_per_second':60.6,
        'median_output_tokens_per_second':114.1,'minimum_positive_output_tokens_per_second':13.9,
        'observed_active_plus_shutdown_seconds':816,'observed_items':228},
        'work_allowance':1.25,'votes':3,'transport_attempts_per_vote':3,'max_tokens_per_vote':8192,
        'load_seconds_projection':736,'shutdown_seconds_projection':240,'per_shard_reserve_seconds':900,
        'transport_timeout_seconds':1800,'transport_sleep_seconds_per_vote':30,'cold_readiness_limit_seconds':2400,
        'max_complete_items_per_shard':48,'max_shard_wall_seconds':21600}


class PriceTests(unittest.TestCase):
    def test_mapping_ids_exact_and_malformed_rejected(self):
        self.assertEqual(j.exact_ids(UserDict(input_ids=[[1,2,3]])),[1,2,3])
        for value in ([1,True],[1,-1],[[1],[2]],[],{},[1.0]):
            with self.assertRaises(ValueError):j.exact_ids(value)

    def test_zero_items_is_zero_GPU_and_no_shards(self):
        price=j.price_rows([],settings())
        self.assertEqual(price['shards'],[])
        for key in ('requested_allocation_GPU_hour_ceiling','median_full_cap_all_attempts_projected_GPU_h','client_timeout_envelope_GPU_h'):
            self.assertEqual(price[key],0)

    def test_all_votes_attempts_loads_overhead_and_exact_prompts_counted(self):
        price=j.price_rows([{'item_id':'x','prompt_tokens':1000}],settings())
        shard=price['shards'][0]
        self.assertEqual(price['votes'],3);self.assertEqual(price['max_transport_attempts'],9)
        self.assertEqual(shard['exact_prompt_tokens_all_attempts'],9000)
        self.assertEqual(shard['max_decode_tokens_all_attempts'],9*8192)
        self.assertGreater(price['median_full_cap_all_attempts_projected_GPU_h'],2*(736+240+900)/3600)
        self.assertGreater(price['pessimistic_minimum_positive_rate_projected_GPU_h'],price['median_full_cap_all_attempts_projected_GPU_h'])
        self.assertEqual(price['cold_loads'],1);self.assertEqual(price['shutdowns'],1)

    def test_generic384_items_pack_completely_and_reserve216_GPU_hours(self):
        rows=[{'item_id':str(i),'prompt_tokens':16384} for i in range(384)]
        price=j.price_rows(rows,settings())
        self.assertEqual(len(price['shards']),18)
        self.assertEqual(price['requested_allocation_GPU_hour_ceiling'],216.)
        self.assertEqual([i for s in price['shards'] for i in s['item_ids']],[r['item_id'] for r in rows])
        self.assertTrue(all(len(s['item_ids'])<=48 and s['median_full_cap_wall_seconds']<=21600 and
                            s['client_timeout_envelope_wall_seconds']<=21600 for s in price['shards']))
        self.assertAlmostEqual(price['median_full_cap_all_attempts_projected_GPU_h'],209.15713981580888)

    def test_context_overflow_and_unfit_complete_item_rejected(self):
        with self.assertRaisesRegex(ValueError,'context'):
            j.price_rows([{'item_id':'x','prompt_tokens':16385}],settings())
        bad=settings();bad['max_shard_wall_seconds']=100
        with self.assertRaisesRegex(ValueError,'cannot fit'):
            j.price_rows([{'item_id':'x','prompt_tokens':1000}],bad)

    def test_messages_match_frozen_offline_problem_reference_candidate_only(self):
        item={'problem':'p','gold':'g','candidate':'c','arm':'forbidden','future':'ignored'}
        self.assertEqual(j.messages(item,SimpleNamespace(SYSTEM='system')),
            [{'role':'system','content':'system'},{'role':'user','content':'Problem:\np\n\nReference answer:\ng\n\nCandidate:\nc'}])


class EnvironmentTests(unittest.TestCase):
    def test_all_credentials_provider_and_affinity_variables_removed(self):
        env={'PATH':'/bin','HOME':'/owned','MODULEPATH':'/modules','BASH_FUNC_module%%':'module body',
            'HF_TOKEN':'private','HF_ANYTHING_SECRET':'private','OPENAI_API_KEY':'private','AWS_ACCESS_KEY_ID':'private',
            'ANTHROPIC_API_KEY':'private','STEER_FAKE_TOKEN':'private','CUDA_VISIBLE_DEVICES':'0,1',
            'SLURM_CPU_BIND_LIST':'0xff','SRUN_CPU_BIND':'cores','SBATCH_EXPORT':'ALL','OTHER_PASSWORD':'private'}
        saved=copy.deepcopy(env)
        out=j.safe_environment(env,{'STEER_CODE':'/frozen','STEER_EXPECT_TREE':'sha','STEER_PROVENANCE_DIR':'/owned/provenance'})
        self.assertEqual(env,saved)
        self.assertEqual(set(out),{'PATH','HOME','MODULEPATH','BASH_FUNC_module%%','STEER_CODE','STEER_EXPECT_TREE','STEER_PROVENANCE_DIR'})
        self.assertNotIn('private',out.values())
        with self.assertRaisesRegex(ValueError,'unapproved'):
            j.safe_environment(env,{'HF_TOKEN':'private'})


SAMPLER={'temperature':1.,'top_p':.95,'top_k':20,'min_p':0.,'presence_penalty':0.,'repetition_penalty':1.}
JUDGE=SimpleNamespace(PARSER_VERSION=2,PROMPT_VERSION=1,JUDGE_SAMPLER=SAMPLER)

def row(uid='x'):
    return {'item_id':uid,'votes':['EQUIVALENT','EQUIVALENT','NOT_EQUIVALENT'],'verdict':'EQUIVALENT',
            'unanimous':False,'truncated_votes':0,'parser_version':2,'prompt_version':1,'sampler':dict(SAMPLER)}


class VerdictTests(unittest.TestCase):
    def check_rows(self,rows,ids):
        with tempfile.TemporaryDirectory(dir=REPO/'tests') as directory:
            path=Path(directory)/'verdicts.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            outcomes=SimpleNamespace(scoring_modules=lambda:(SimpleNamespace(answer_equivalence=JUDGE),None))
            with mock.patch.object(j,'outcome_module',return_value=outcomes):return j.valid_verdicts(path,ids)

    def test_exact_three_vote_majority_accepted(self):
        self.assertEqual(self.check_rows([row()],['x'])[0]['verdict'],'EQUIVALENT')

    def test_uncertain_transport_errors_preserved_without_semantic_retry(self):
        r=row();r.update(votes=['ERROR:TimeoutError','TRUNCATED','UNPARSED'],verdict='UNCERTAIN',truncated_votes=1)
        self.assertEqual(self.check_rows([r],['x'])[0]['verdict'],'UNCERTAIN')

    def test_duplicate_missing_foreign_items_and_short_vote_lists_rejected(self):
        for rows,ids in (([row(),row()],['x']),([],['x']),([row('y')],['x'])):
            with self.assertRaisesRegex(ValueError,'item IDs'):self.check_rows(rows,ids)
        r=row();r['votes']=r['votes'][:2]
        with self.assertRaisesRegex(ValueError,'contract'):self.check_rows([r],['x'])

    def test_wrong_sampler_parser_or_majority_is_rejected(self):
        for key,value in [('parser_version',1),('prompt_version',2),('sampler',{}),('verdict','NOT_EQUIVALENT')]:
            r=row();r[key]=value
            with self.assertRaises(ValueError):self.check_rows([r],['x'])
        r=row();r['votes']=[{},'EQUIVALENT','EQUIVALENT']
        with self.assertRaises(ValueError):self.check_rows([r],['x'])

    def test_parent_only_GPU_accounting_is_exact(self):
        result=subprocess.CompletedProcess([],0,'123|COMPLETED|0:0|1552|cpu=16,gres/gpu=2,mem=120G|\n','')
        with mock.patch.object(j.subprocess,'run',return_value=result):
            cost=j.allocation('123')
        self.assertAlmostEqual(cost['allocated_GPU_hours'],2*1552/3600)
        failed=subprocess.CompletedProcess([],0,'123|FAILED|1:0|1552|cpu=16,gres/gpu=2|\n','')
        with mock.patch.object(j.subprocess,'run',return_value=failed):
            with self.assertRaisesRegex(ValueError,'successfully'):j.allocation('123')


class DispatchTests(unittest.TestCase):
    def run_dispatch(self,shards):
        with tempfile.TemporaryDirectory(dir=REPO/'tests') as directory:
            root=Path(directory)
            calls=[]
            def submit(directory,name,args,env,binding):
                calls.append((name,args,env,binding));return str(100+len(calls))
            with mock.patch.object(j,'priced',return_value=({'sha256':'plan'},{'sha256':'prep'},{'sha256':'price','shards':shards},[])), \
                 mock.patch.object(j.shared,'submit',side_effect=submit), \
                 mock.patch.object(j.verify,'live_account_snapshot'),mock.patch.object(j.verify,'ensure_verified'), \
                 mock.patch.object(j.dependencies,'dependency',return_value=(['--dependency=afterok:101'],{'mode':'LIVE_AFTEROK'})), \
                 mock.patch.dict(os.environ,{'SLURM_JOB_ID':'99','HF_TOKEN':'private-test-value'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                j.dispatch(root)
            return calls

    def test_zero_items_submits_CPU_finalizer_only(self):
        calls=self.run_dispatch([])
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'utility-j1-finalize')
        self.assertIn(str(j.WRAPPERS['finalize']),calls[0][1])
        self.assertFalse(any(a.startswith('--dependency') for a in calls[0][1]))

    def test_nonempty_uses_exact_legacy_wrapper_and_sanitized_environment(self):
        shard={'index':0,'items_path':'/owned/items.jsonl','verdicts_path':'/owned/verdicts.jsonl','items_file_sha256':'items','item_ids':['x']}
        calls=self.run_dispatch([shard])
        self.assertEqual(len(calls),2)
        self.assertIn(str(j.LEGACY),calls[0][1])
        self.assertEqual(calls[0][1][-2:],['/owned/items.jsonl','/owned/verdicts.jsonl'])
        self.assertNotIn('HF_TOKEN',calls[0][2])
        self.assertNotIn('private-test-value',calls[0][2].values())
        self.assertIn('--dependency=afterok:101',calls[1][1])


if __name__=='__main__':unittest.main()
