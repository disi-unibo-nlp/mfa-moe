"""Allocated, source-bound retirement of disposable terminal-job caches."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = REPO/'report/storage-recovery-20261011-r1'
sys.path[:0] = [str(REPO/'src'),str(REPO/'scripts/experimental_resume')]
from moe_exp.research_campaign_v3.contracts import allocated, check_files, load, publish, require

allocated()
require(os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_serial', 'wrong cleanup allocation')
card = load(ROOT/'CARD.json')
check_files(card['source_files'])
subprocess.run([sys.executable,'-B',str(REPO/'scripts/research_campaign_v3_r3/reclaim_storage.py'),
                'apply','--plan',str(ROOT/'PLAN.json')],check=True)
spec = importlib.util.spec_from_file_location('cleanup',REPO/'scripts/research_campaign_v3_r3/reclaim_storage.py')
cleanup_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(cleanup_module)
protected = cleanup_module.protected_paths()
download_root = Path('/leonardo_work/IscrC_MIOSR/lmolfett/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct')
known_gpu_jobs = {'59954129','59954133','59954135'}
for jid in cleanup_module.live_ids():
    readback = cleanup_module.command(['scontrol','show','job','-o',jid])
    require('Partition=boost_usr_prod' not in readback or jid in known_gpu_jobs,
            'new GPU allocation requires another download-use inspection')
for log,model in card['active_model_logs'].items():
    with Path(log).open() as handle:
        first = handle.readline()
    require(model in first and 'Qwen2.5-VL' not in first,'active checkpoint changed')
download_bytes = 0
for item in card['unused_download_weights']:
    link,blob = Path(item['link']),Path(item['blob'])
    require(link.parent == download_root/'snapshots'/card['unused_download_revision'] and
            link.name.endswith('.safetensors') and blob.parent == download_root/'blobs',
            'weight retirement is outside exact obsolete download')
    require(str(blob) not in protected and str(link) not in protected,'weight is source-bound')
    if link.is_symlink():
        require(link.resolve() == blob,'weight download link changed')
        link.unlink()
    require(not link.exists(),'weight link is no longer a download symlink')
    if blob.exists():
        require(not blob.is_symlink() and cleanup_module.identity(blob) == item['identity'] and
                blob.stat().st_size == item['bytes'] and blob.stat().st_nlink == 1,
                'weight blob changed, is shared, or is not owned')
        blob.unlink()
    require(not blob.exists(),'weight retirement not verified')
    download_bytes += item['bytes']
    publish(ROOT/'download_removed'/(blob.name+'.json'),{**item,'status':'retired',
        'repository':'Qwen/Qwen2.5-VL-7B-Instruct','revision':card['unused_download_revision'],
        'configuration_tokenizer_and_download_metadata_preserved':True})
campaign = load(REPO/'report/research-campaign-v3-20261009-r2/CAMPAIGN.json')
check_files(campaign['code_files'])
parent = Path(campaign['root'])
compact = parent/'gpt/compact'
before = sum(p.stat().st_size for p in compact.rglob('*') if p.is_file()) if compact.is_dir() else 0
if not (parent/'gpt/CACHE_RETIRED.json').exists():
    from moe_exp.research_campaign_v3.extraction import retire
    retire(campaign,'gpt')
require(not compact.exists(), 'completed GPT temporary cache still present')
cleanup = json.loads((ROOT/'COMPLETE.json').read_text())
result = publish(ROOT/'RESULT.json', {'status':'complete','job_id':os.environ['SLURM_JOB_ID'],
    'compiler_cache_allocated_bytes':sum(r.get('estimated_allocated_bytes',0) for r in cleanup['removed']),
    'completed_GPT_compact_logical_bytes':before,'compiler_cache_count':len(cleanup['removed']),
    'unused_public_download_bytes':download_bytes,
    'quota_after':cleanup_module.command(['lfs','quota','-p','20149688','/leonardo_work']),
    'skipped':cleanup['skipped'],'GPT_cache_retirement':str(parent/'gpt/CACHE_RETIRED.json'),
    'original_captures_and_evidence_preserved':True,'required_checkpoints_and_active_jobs_preserved':True})
print(json.dumps(result,sort_keys=True,allow_nan=False))
