"""Read-only saved-label validation before the sole GPU retry."""
import hashlib,json,os
from pathlib import Path
from moe_exp.correlation_pipeline.annotation_partition import enumerate_items,DATASETS
from moe_exp.correlation_pipeline.annotation_batch import run_part,digest
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    out=ROOT/'results/measurement-study/label-resume-v1/qwen330b/part-01'
    old=json.loads((out/'preflight.json').read_text());assert old['max_model_len']==57344
    first=json.loads((out/'checkpoints/batch-000000.json').read_text());config=first['binding']['config']
    items=enumerate_items(Path(old['source']),DATASETS,limit=25000)
    plan=dict(schema_version=1,trace_root=old['source'],datasets=list(DATASETS),include_traces_sha256=None,source_identities=len(items),selected_identities=len(items),part_size=25000,parts=1,part_counts=[len(items)],selected_sha256=hashlib.sha256('\n'.join(digest(x[0]) for x in items).encode()).hexdigest(),first_identity=items[0][0],last_identity=items[-1][0])
    assert digest(plan)==config['source_plan_sha256']
    program=ROOT.parent/'repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json'
    assert hashlib.sha256(program.read_bytes()).hexdigest()==config['program_sha256']
    result=run_part(items,config,output_dir=out,classify=None,expected_count=25000,dry_run=True)
    assert result['completed']==15616 and result['expected']==25000
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'checkpoints').glob('*.json'))}
    result.update(status='validated',remaining=9384,max_model_len=57344,checkpoint_hashes=hashes,judge_config=config)
    target=ROOT/'results/measurement-study/label-recovery-preflight.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoint_hashes','judge_config')}),flush=True)
if __name__=='__main__':main()
