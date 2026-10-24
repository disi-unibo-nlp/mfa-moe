import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from measurement_study.bundle import freeze_bundle, verify_bundle, verify_imports, source_tree_hash

STAGE = Path(__file__).resolve().parents[2] / 'results/measurement-study/campaign-v3/tests'
STAGE.mkdir(parents=True, exist_ok=True)


class BundleTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source = root / 'source'
        src = root / 'repo-src'
        for sub in ('measurement_study', 'scripts'):
            (source / sub).mkdir(parents=True)
        (source/'measurement_study/__init__.py').write_text('')
        original = Path(__file__).resolve().parents[2]/'measurement_study/bundle.py'
        (source/'measurement_study/bundle.py').write_bytes(original.read_bytes())
        (src/'moe_exp').mkdir(parents=True)
        (src/'moe_exp/__init__.py').write_text('')
        (src/'moe_exp/fixture.py').write_text('VALUE = 7\n')
        (source/'scripts/worker_fixture.py').write_text('from moe_exp.fixture import VALUE\ndef work(x):\n return VALUE+x\n')
        for name in ('run_routing_study.py','verify_routing_study.py','routing_study_env.sh'):
            (source/'scripts'/name).write_text('# fixture closure file\n')
        # Runtime closure required by src/moe_exp/routing_study/common.py:code_hashes().
        for package, files in (('routing_study', ('__init__.py','common.py','reducers.py')),
                               ('moe_identity_guiding', ('execution.py','calibration.py','routing.py')),
                               ('moe_margin_guiding', ('calibration.py','routing.py'))):
            (source/'src/moe_exp'/package).mkdir(parents=True)
            for name in files:
                (source/'src/moe_exp'/package/name).write_text('# fixture\n')
        (source/'src/moe_exp/routing_study/reducer_provenance.json').write_text('{}')
        (source/'src/moe_exp/routing_study/checkpoint_files.json').write_text('{}')
        (src/'moe_exp/correlation_pipeline').mkdir(parents=True)
        (src/'moe_exp/correlation_pipeline/scoring.py').write_text('CONTRACT=3\n')
        (src/'moe_exp/utils.py').write_text('VALUE=1\n')
        (source/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (source/'sbatch').mkdir()
        (source/'sbatch/routing_study_smoke.sbatch').write_text('#!/bin/bash\n')
        receipt = root/'t1.json'
        receipt.write_text(json.dumps({'status':'approved','reviewer':'claude','repo_src_sha256':source_tree_hash(src)}))
        return source, src, root/'bundle', receipt

    def test_freeze_corruption_immutable_and_traversal(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt = self.fixture(temp)
            result=freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            self.assertIn('src/moe_exp/fixture.py',result['files'])
            with self.assertRaises(FileExistsError):
                freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            (bundle/'src/moe_exp/fixture.py').write_text('VALUE=9\n')
            with self.assertRaisesRegex(ValueError,'Frozen code changed'):
                verify_bundle(bundle)
            hashes=json.loads((bundle/'SHA256.json').read_text());hashes['../../outside']='bad'
            (bundle/'SHA256.json').write_text(json.dumps(hashes))
            with self.assertRaises(ValueError):verify_bundle(bundle)

    def test_source_final_receipt_required(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            receipt.write_text('{"status":"pending","reviewer":"claude"}')
            with self.assertRaisesRegex(ValueError,'T1 receipt'):
                freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            self.assertFalse(bundle.exists())

    def test_actual_spawn_worker_imports_frozen_source(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            script=Path(temp)/'run.py'
            script.write_text('''import sys, multiprocessing\nfrom concurrent.futures import ProcessPoolExecutor\nfrom pathlib import Path\nbundle=Path(sys.argv[1])\nsys.path[:0]=[str(bundle),str(bundle/'src'),str(bundle/'scripts')]\nfrom measurement_study.bundle import worker_init,verify_imports\nfrom worker_fixture import work\nif __name__=='__main__':\n worker_init(bundle)\n verify_imports(bundle,['worker_fixture','moe_exp.fixture'])\n with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(bundle,)) as pool:\n  assert pool.submit(work,5).result()==12\n print('worker-complete')\n''')
            result=subprocess.run([sys.executable,'-B',str(script),str(bundle)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('worker-complete',result.stdout)

    def test_t1_receipt_rejects_unreviewed_source_edit(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            (src/'moe_exp/fixture.py').write_text('VALUE=8\n')
            with self.assertRaisesRegex(ValueError,'reviewed repo/src'):
                freeze_bundle(source,src,bundle,{'schema_version':3},receipt)

    def test_worktree_routing_overlay_preserves_repo_scorer(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            overlay=source/'src/moe_exp/routing_study'
            (overlay/'common.py').write_text('ORIGIN="worktree"\n')
            (src/'moe_exp/routing_study').mkdir()
            (src/'moe_exp/routing_study/common.py').write_text('ORIGIN="repo"\n')
            (src/'moe_exp/correlation_pipeline/scoring.py').write_text('CONTRACT=3\n')
            receipt.write_text(json.dumps({'status':'approved','reviewer':'claude','repo_src_sha256':source_tree_hash(src)}))
            result=freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            self.assertEqual((bundle/'src/moe_exp/routing_study/common.py').read_text(),'ORIGIN="worktree"\n')
            self.assertEqual((bundle/'src/moe_exp/correlation_pipeline/scoring.py').read_text(),'CONTRACT=3\n')
            self.assertEqual((bundle/'src/moe_exp/routing_study/checkpoint_files.json').read_text(),'{}')
            for name in ('moe_identity_guiding/execution.py','moe_margin_guiding/calibration.py'):
                self.assertIn('src/moe_exp/'+name,result['files'])
            self.assertEqual(result['campaign']['guiding_overlay_sha256']['moe_identity_guiding'],
                             source_tree_hash(source/'src/moe_exp/moe_identity_guiding'))
            self.assertEqual(result['campaign']['routing_study_overlay_sha256'],source_tree_hash(overlay))
            self.assertEqual(result['campaign']['source_origins']['src/moe_exp/routing_study/common.py'],str(overlay/'common.py'))

    def test_real_frozen_launcher_forwards_bundle_and_spawns_worker(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            wrapper=Path(__file__).resolve().parents[2]/'scripts/run_campaign_v3_module.py'
            (source/'scripts/run_campaign_v3_module.py').write_bytes(wrapper.read_bytes())
            (source/'measurement_study/production_plan.py').write_text("""import argparse,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from measurement_study.bundle import worker_init
from worker_fixture import work
def main():
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--bundle',required=True);a=p.parse_args()
 assert a.source=='fixture'
 with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(a.bundle,)) as pool:
  assert pool.submit(work,5).result()==12
 print('frozen-launcher-worker-complete')
""")
            freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            result=subprocess.run([sys.executable,'-B',str(bundle/'scripts/run_campaign_v3_module.py'),
                                   '--bundle',str(bundle),'--module','measurement_study.production_plan','fixture'],
                                  capture_output=True,text=True,timeout=30,
                                  env={**os.environ,'SLURM_JOB_ID':'cpu-fixture'})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('frozen-launcher-worker-complete',result.stdout)

    def test_live_import_rejected(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            source,src,bundle,receipt=self.fixture(temp)
            freeze_bundle(source,src,bundle,{'schema_version':3},receipt)
            with self.assertRaisesRegex(ValueError,'escaped'):
                verify_imports(bundle,['measurement_study.bundle'])

    def test_real_worktree_bundle_satisfies_runtime_code_hashes_closure(self):
        """m3: the frozen closure must satisfy src/moe_exp/routing_study/common.py."""
        worktree=Path(__file__).resolve().parents[2]
        repo_src=worktree.parent/'repo/src'
        runner='''import json, sys
sys.path[:0]=[sys.argv[1], sys.argv[1]+"/src", sys.argv[1]+"/scripts"]
from moe_exp.routing_study.common import code_hashes
print(json.dumps(code_hashes(), sort_keys=True))
'''
        for attempt in range(3):
            with tempfile.TemporaryDirectory(dir=STAGE) as temp:
                root=Path(temp)
                receipt=root/'t1.json'
                receipt.write_text(json.dumps({'status':'approved','reviewer':'claude',
                    'repo_src_sha256':source_tree_hash(repo_src)}))
                try:
                    result=freeze_bundle(worktree,repo_src,root/'bundle',{'schema_version':3},receipt)
                except ValueError as error:
                    if 'changed during freeze' in str(error) and attempt < 2:
                        continue
                    raise
                frozen=subprocess.run([sys.executable,'-B','-c',runner,str(root/'bundle')],
                                      capture_output=True,text=True,timeout=120)
                self.assertEqual(frozen.returncode,0,frozen.stderr)
                bundle_hashes=json.loads(frozen.stdout)
                live=subprocess.run([sys.executable,'-B','-c',runner,str(worktree)],
                                    capture_output=True,text=True,timeout=120)
                self.assertEqual(live.returncode,0,live.stderr)
                live_hashes=json.loads(live.stdout)
                self.assertEqual(set(bundle_hashes),set(live_hashes))
                self.assertIn('moe_exp/routing_study/checkpoint_files.json',bundle_hashes)
                for name,digest in live_hashes.items():
                    if name in ('moe_exp/correlation_pipeline/scoring.py','moe_exp/utils.py'):
                        self.assertEqual(bundle_hashes[name],result['files']['src/'+name])
                    else:
                        self.assertEqual(bundle_hashes[name],digest,name)
                return
        self.fail('real worktree changed during freeze retries')

if __name__=='__main__':unittest.main()
