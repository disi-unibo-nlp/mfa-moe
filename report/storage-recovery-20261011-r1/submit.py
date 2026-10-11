"""One journaled, bounded storage-recovery allocation; never redraw science."""
from pathlib import Path
import sys
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT=REPO/'report/storage-recovery-20261011-r1'
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r3.contracts import file_sha,load,publish,locked,require
from moe_exp.research_campaign_v3_r3.scheduler import Slurm,serial_capacity
from importlib.util import module_from_spec,spec_from_file_location
spec=spec_from_file_location('cleanup',REPO/'scripts/research_campaign_v3_r3/reclaim_storage.py')
cleanup=module_from_spec(spec);spec.loader.exec_module(cleanup)

with locked(ROOT,'cleanup-submission'):
    if (ROOT/'ACCEPTED.json').exists():
        print(load(ROOT/'ACCEPTED.json')['job_id']);sys.exit(0)
    require(not (ROOT/'INTENT.json').exists(),'reconcile existing intent before another submit')
    plan=cleanup.plan if False else __import__('json').loads((ROOT/'PLAN.json').read_text())
    require(plan['sha256']==cleanup.digest({k:v for k,v in plan.items() if k!='sha256'}),'cleanup plan changed')
    download=Path('/leonardo_work/IscrC_MIOSR/lmolfett/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct')
    revision='cc594898137f460bfe9f0759e9844b3ce807cfb5'
    weights=[]
    for link in sorted((download/'snapshots'/revision).glob('*.safetensors')):
        blob=link.resolve()
        require(link.is_symlink() and blob.parent==download/'blobs' and blob.stat().st_uid==__import__('os').getuid(),
                'obsolete model is not an owned public cache')
        weights.append({'link':str(link),'blob':str(blob),'bytes':blob.stat().st_size,'identity':cleanup.identity(blob)})
    require(len(weights)==5,'expected exact original obsolete download')
    files=[ROOT/'worker.py',ROOT/'job.sbatch',Path(__file__).resolve(),ROOT/'PLAN.json',
           REPO/'scripts/research_campaign_v3_r3/reclaim_storage.py']
    card=publish(ROOT/'CARD.json',{'root':str(ROOT),'stage':'storage-recovery','partition':'lrd_all_serial',
        'comment':'mfa-v3-storage-recovery-20261011-r1','cpus':2,'allocation_cpus':2,'memory_gib':2,
        'memory_mib':2048,'billing_units_upper':2,'minutes':30,'gpus':0,'hold':False,
        'reservation_hours':1.,'margin_hours':1.25,'source_files':{str(p):file_sha(p) for p in files},
        'authorization':'User: always authorized; inspect and delete obsolete memory before protected relaunch',
        'catalogue':['X06','X11','X12'],'paper_artifact':'complete seven-model native atlas, paired utility and feasibility report',
        'missing_cell':'six saved-data analyses blocked by WORK quota; no new scientific draw',
        'controls':'terminal accounting, live-array exclusion, immutable-source exclusion, exact path and inode checks',
        'independent_family_support':'original frozen families; cleanup creates no measurement or enrollment',
        'planned_compiler_cache_bytes':plan['estimated_allocated_bytes'],
        'unused_download_weights':weights,'unused_download_revision':revision,
        'active_model_logs':{
          '/leonardo_work/IscrC_MIOSR/lmolfett/runs/medexplain-benchmark/20261010-v2/logs/followup-qwen-59954133_4294967294.out':'Qwen3.6-35B-A3B-FP8',
          '/leonardo_work/IscrC_MIOSR/lmolfett/runs/medexplain-benchmark/20261010-v2/logs/followup-medgemma-59954129_4294967294.out':'medgemma-27b-it'}})
    scheduler=Slurm();snapshot=scheduler.snapshot()
    require(all(serial_capacity(snapshot,[card],handoff_slots=0).values()),'serial cleanup capacity unavailable')
    require(snapshot['remaining_local_hours']>=1.25,'cleanup exceeds live entitlement')
    publish(ROOT/'PREFLIGHT.json',snapshot)
    (ROOT/'logs').mkdir(exist_ok=True,mode=0o700)
    args=['sbatch','--account=iscrc_miosr','--partition=lrd_all_serial','--qos=normal',
          '--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=2G','--time=00:30:00','--no-requeue',
          '--job-name=rc3-storage-recovery','--comment='+card['comment'],'--chdir='+str(REPO),
          '--output='+str(ROOT/'logs/%x-%j.out'),'--error='+str(ROOT/'logs/%x-%j.err')]
    validation=scheduler.command([*args,'--test-only',str(ROOT/'job.sbatch')])
    publish(ROOT/'SCHEDULER_VALIDATION.json',{'response':validation,'test_only':True})
    publish(ROOT/'INTENT.json',{'card_sha256':card['sha256'],'comment':card['comment'],'argv':args})
    response=scheduler.command([*args,'--parsable',str(ROOT/'job.sbatch')])
    jid=response.split(';')[0];require(jid.isdigit(),'ambiguous cleanup submission')
    proof=scheduler.verify(jid,card)
    publish(ROOT/'ACCEPTED.json',{'job_id':jid,'response':response,'readback':proof,'card_sha256':card['sha256']})
    print(jid)
