"""Seal a separate qualification overlay; existing sampler trees remain untouched."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[2]
S=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def freeze():
    sources={p.name:p for p in (REPO/'src/moe_exp/routing_control').glob('*.py')}
    sources['qualify_ordered.py']=REPO/'scripts/experimental_resume/qualify_ordered.py'
    for p in sources.values():ast.parse(p.read_text())
    table={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sources.items()}
    key=hashlib.sha256(json.dumps(table,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    directory=S/'addenda/ordered'/key[:16];overlay=directory/'moe_exp_src'
    if not overlay.exists():shutil.copytree(S/'code/s1-9a61e32f48c04c24/moe_exp_src',overlay)
    overlay.chmod(0o700);(overlay/'moe_exp').chmod(0o700)
    package=overlay/'moe_exp/routing_control';package.mkdir(exist_ok=True)
    for name,p in sources.items():
        target=directory/name if name=='qualify_ordered.py' else package/name
        if target.exists() and target.read_bytes()!=p.read_bytes():raise ValueError('immutable overlay changed')
        if not target.exists():target.write_bytes(p.read_bytes());target.chmod(0o400)
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(overlay.rglob('*')) if p.is_file()}
    files[str(directory/'qualify_ordered.py')]=table['qualify_ordered.py']
    body={'schema':'ordered-qualification-code-v1','files':files,'overlay':str(overlay),
        'driver':str(directory/'qualify_ordered.py'),'worker_sources_sha256':key,
        'base_tree_sha256':'9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430',
        'qualification_complete_price':{'requests':17,'maximum_decode_tokens':16512,'GPUs':2,'wall_minutes':19,
            'observed_shutdown_reserve_seconds':196,'maximum_including_shutdown_GPU_hours':(1140+196)*2/3600,
            'previous_failed_initialization_GPU_hours':13*2/3600,'ceiling':.75,
            'all_prefix_preparation_loads_prefill_capture_and_recovery_included':True,
            'historical_H14_full_allocation_seconds':838,'fixed_fixture_execution_estimate_seconds':120,
            'prefix_preparation_and_prefill_allowance_seconds':60,'grading_GPU_hours':0,'retries_after_this':0}}
    value={**body,'sha256':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    out=S/'runs/ordered-qualification-v1';out.mkdir(exist_ok=True)
    path=out/'PREPARED.retry.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('prepared retry already differs')
    path.write_text(json.dumps(value,indent=1)+'\n')
    package.chmod(0o500);(overlay/'moe_exp').chmod(0o500);overlay.chmod(0o500)
    print(json.dumps({'binding':value['sha256'],'driver':body['driver'],'files':len(files)}))


if __name__=='__main__':freeze()
