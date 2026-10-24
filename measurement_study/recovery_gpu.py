"""GPU acceptance and unchanged-resume checks for one frozen Gemma validation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from measurement_study.production_plan import ROOT,storage_root,sha,atomic,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    for rel,expected in json.loads((a.bundle/'SHA256.json').read_text()).items():
        if sha(a.bundle/rel)!=expected:raise ValueError('Frozen validator changed')
    manifest_root=storage_root('gemma')/'manifests/gemma'
    ready=json.loads((manifest_root/'ready.json').read_text())
    assert ready['code_binding']==sha(a.bundle/'SHA256.json')
    assert ready['status']=='ready' and ready['tokenizer_saved_ids_verified']
    shard=ready['verification_shard'];out=storage_root('gemma')/'capture/gemma'/str(shard)
    cmd=[sys.executable,'-B','-m','measurement_study.production_capture','gemma','--shard',str(shard),'--verify-only','--bundle',str(a.bundle)]
    subprocess.run(cmd,check=True)
    def snapshot():
        result={}
        for marker in out.glob('*.complete.json'):
            record=json.loads(marker.read_text());result[str(marker)]=(sha(marker),marker.stat().st_mtime_ns)
            for f in record['files']:
                path=Path(f['path']);assert sha(path)==f['sha256'];result[str(path)]=(f['sha256'],path.stat().st_mtime_ns)
        return result
    before=snapshot();assert len(before)>=6
    subprocess.run(cmd,check=True)
    assert snapshot()==before,'Resume changed accepted captures'
    verification=json.loads((out/'verification.json').read_text())
    assert verification['maximum_input_tokens']==ready['maximum_input_tokens']
    atomic(out/'acceptance.json',dict(status='passed',verification=verification,unchanged_resume=True,bundle_sha256=sha(a.bundle/'SHA256.json'),checkpoint_revision=ready['spec']['revision'],historical_generation_revision_status='unverified'))
if __name__=='__main__':main()
