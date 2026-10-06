"""Seal two-stage R3-D checkpoint merge recovery after fulfilled afterany guard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT=REPO/'report/experimental-resume-v1'
SOURCE=REPO/'scripts/experimental_resume/merge_r3d_sidecar_v2.py'
LAUNCHER=REPO/'scripts/experimental_resume/merge_r3d_sidecar_v2.sbatch'
OUT=REPORT/'R3D_MERGE_RECOVERY_PRICE_v2.json'


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sealed(path):
    x=json.loads(path.read_text())
    if digest({k:v for k,v in x.items() if k!='sha256'})!=x.get('sha256'):
        raise ValueError(f'changed R3-D receipt {path}')
    return x


def state(job):
    rows=subprocess.run(['sacct','-j',str(job),'-X','-P','-n','-o','State,ExitCode'],
                        check=True,capture_output=True,text=True).stdout.strip().splitlines()
    if len(rows)!=1:raise ValueError(f'ambiguous Slurm job {job}: {rows}')
    return rows[0].split('|')[:2]


def main():
    q=sealed(REPORT/'R3D_SIDECAR_VERIFY_QWEN_v2.json')
    g=sealed(REPORT/'R3D_SIDECAR_VERIFY_GPT_DEST_v2.json')
    if q['total_fit_checkpoints']!=750 or g['total_fit_checkpoints']!=300:
        raise ValueError('complete independent checkpoint counts differ')
    if state(59111360)[0] in ('RUNNING','PENDING','CONFIGURING','COMPLETING'):
        raise ValueError('canonical writer still active')
    for job in (59173234,59187091,59180550,59201047):
        if state(job)!=['COMPLETED','0:0']:
            raise ValueError(f'unqualified sidecar/verifier {job}')
    cont=subprocess.run(['scontrol','show','job','59115320'],check=True,
                        capture_output=True,text=True).stdout
    if 'JobState=PENDING Reason=JobHeldUser Dependency=(null)' not in cont:
        raise ValueError('exact continuation no longer held after fulfilled dependency')
    body={'schema':'r3d-merge-recovery-price-v2','status':'READY_TWO_SEQUENTIAL_CPU_MERGES',
          'reason':'v1 Qwen merge 59188564 failed before copying because Slurm cleared fulfilled afterany to Dependency=(null); dependent GPT merge 59201091 canceled',
          'failed_prior_jobs':['59188564','59201091'],
          'canonical_stopped_job':'59111360','held_continuation_job':'59115320',
          'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'launcher_sha256':hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(),
          'qwen_verifier_sha256':q['sha256'],'gpt_verifier_sha256':g['sha256'],
          'stages':[{'kind':'qwen','fit_files':750,'cpus':2,'memory':'16G','wall_seconds':1800,
                     'max_core_hours':1,'receipt':str(REPORT/'R3D_SIDECAR_MERGE_QWEN_v2.json')},
                    {'kind':'gpt-dest','fit_files':300,'cpus':2,'memory':'16G','wall_seconds':1800,
                     'max_core_hours':1,'dependency':'afterok Qwen v2 merge',
                     'receipt':str(REPORT/'R3D_SIDECAR_MERGE_GPT_DEST_v2.json')}],
          'total_max_core_hours':2,'gpus':0,
          'slurm':{'account':'iscrc_miosr','partition':'boost_usr_prod','qos':'normal'},
          'release_gate':'Both v2 merge Slurm states COMPLETED0 and sealed receipts valid; exact 59115320 still user-held; then release it only'}
    value={**body,'sha256':digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text())!=value:
            raise ValueError('existing v2 recovery price differs')
    else:
        OUT.write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps({'price':str(OUT),'seal':value['sha256']}))


if __name__=='__main__':main()
