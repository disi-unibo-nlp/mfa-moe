"""One-shot CPU sensitivity attachments behind fresh reader and analysis receipts."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess

import analyze_fresh_veto_sensitivity_v1 as analysis
import dispatch_generated_dense_v1 as verify
import dispatch_overnight_readers_v1 as shared

WRAPPER = Path(__file__).with_suffix('.sbatch')
ANALYSIS_WRAPPER = Path(analysis.__file__).with_suffix('.sbatch')
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD'}


def dependency(job, completed_artifacts, *, run=subprocess.run):
    """A purged successful parent is proved by accounting AND bound artifacts."""
    analysis.require(isinstance(job,str) and re.fullmatch(r'[0-9]+',job), 'invalid parent Slurm ID')
    live = run(['scontrol','show','job',job],capture_output=True,text=True,check=False)
    proof = {'job_id':job,'scontrol_returncode':live.returncode,'scontrol_stdout':live.stdout,
             'scontrol_stderr':live.stderr}
    if live.returncode == 0:
        analysis.require(re.search(r'\bJobId='+job+r'\b',live.stdout) and 'UserId=lmolfett(' in live.stdout,
                         'parent job ID or owner differs')
        match = re.search(r'\bJobState=([A-Z_]+)',live.stdout)
        analysis.require(match is not None,'parent state absent')
        state = match.group(1)
        if state in ACTIVE:
            proof['mode']='LIVE_AFTEROK'
            return [f'--dependency=afterok:{job}'],proof
        analysis.require(state=='COMPLETED' and 'ExitCode=0:0' in live.stdout,
                         'parent did not complete successfully')
        proof['mode']='COMPLETED_ARTIFACT_VERIFIED'
    else:
        history=run(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State,ExitCode,User'],
                    capture_output=True,text=True,check=True)
        rows=[line.strip().rstrip('|').split('|') for line in history.stdout.splitlines() if line.strip()]
        exact=[row for row in rows if row[0]==job]
        analysis.require(len(exact)==1 and exact[0][1:]==['COMPLETED','0:0','lmolfett'],
                         'purged parent lacks exact successful accounting')
        proof.update(mode='PURGED_COMPLETED_ARTIFACT_VERIFIED',sacct_stdout=history.stdout)
    proof['completed_artifacts']=completed_artifacts()
    analysis.require(bool(proof['completed_artifacts']),'successful parent lacks bound artifacts')
    return [],proof


def chain_path(manifest,name):
    return shared.DOC / ('overnight-submissions-v2-'+manifest['sha256'][:16]) / name


def measurement_receipt(manifest):
    path=chain_path(manifest,'MEASUREMENT_CHAIN.json')
    value=shared.base.sealed(path)
    analysis.require(value['schema']=='overnight-measurement-chain-v2' and
                     value['binding']['manifest_sha256']==manifest['sha256'] and
                     Path(value['analysis_output']).resolve()==analysis.PRIMARY.resolve(),
                     'fresh measurement chain differs')
    return path,value


def submit_once(directory,name,parent,artifact_check,wrapper,arguments,binding,*,submit=shared.submit):
    path=directory/(name+'.json')
    if path.exists():
        receipt=shared.base.sealed(path)
        analysis.require(all(receipt['binding'].get(k)==v for k,v in binding.items()) and
                         receipt['binding']['parent_job']==parent,
                         'existing sensitivity attachment differs')
        verify.ensure_verified(directory,name,receipt['job_id'])
        return receipt['job_id'],receipt['binding']['dependency_proof']
    dep,proof=dependency(parent,artifact_check)
    bound={**binding,'parent_job':parent,'dependency_proof':proof}
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    job=submit(directory,name,[*dep,f'--job-name={name}',str(wrapper),*arguments],os.environ.copy(),bound)
    verify.ensure_verified(directory,name,job)
    return job,proof


def attach(phase,*,submit=shared.submit):
    plan=analysis.validate_plan()
    manifest=shared.base.sealed(analysis.MANIFEST)
    analysis.validate_manifest(manifest,shared.base.sealed(analysis.veto.enrollment.OUT),shared.base.sealed(analysis.PROTOCOL))
    directory=shared.DOC/('fresh-veto-sensitivity-chain-'+manifest['sha256'][:16]+'-'+plan['sha256'][:12])
    directory.mkdir(exist_ok=True)
    common={'schema':'fresh-veto-sensitivity-chain-binding-v1','manifest_sha256':manifest['sha256'],
            'plan_sha256':plan['sha256'],'format_result_sha256':analysis.FORMAT_SHA,'phase':phase}
    with (directory/'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot='LIVE_ACCOUNT-'+os.environ.get('SLURM_JOB_ID',str(os.getpid()))+'.json'
        if not (directory/snapshot).exists():
            verify.live_account_snapshot(directory,snapshot)
        if phase=='generation':
            path=chain_path(manifest,'GENERATION_CHAIN.json')
            chain=shared.base.sealed(path)
            analysis.require(chain['schema']=='overnight-generation-chain-v2' and
                             chain['binding']['manifest_sha256']==manifest['sha256'],
                             'fresh generation chain differs')
            def done():
                p,v=measurement_receipt(manifest)
                return {'measurement_chain_path':str(p),'measurement_chain_sha256':v['sha256']}
            job,proof=submit_once(directory,'fresh-veto-attach-analysis',chain['reader_dispatch_job'],done,WRAPPER,
                ['--phase','measurement'],{**common,'source_receipt':str(path),'source_receipt_sha256':chain['sha256']},submit=submit)
            body={'schema':'fresh-veto-reader-attachment-v1',**common,'reader_dispatch_job':chain['reader_dispatch_job'],
                  'next_attachment_job':job,'dependency_proof':proof}
            result=shared.save(directory/'READER_ATTACHMENT.json',body)
        else:
            path,chain=measurement_receipt(manifest)
            def done():
                assigned,primary=analysis.primary_sources(manifest)
                return {'primary_analysis_sha256':primary['sha256'],'primary_assigned_results_sha256':assigned['sha256']}
            job,proof=submit_once(directory,'fresh-veto-outcome-sensitivity',chain['analysis_job'],done,ANALYSIS_WRAPPER,
                [],{**common,'source_receipt':str(path),'source_receipt_sha256':chain['sha256']},submit=submit)
            result=shared.save(directory/'ANALYSIS_ATTACHMENT.json',{'schema':'fresh-veto-analysis-attachment-v1',
                **common,'primary_analysis_job':chain['analysis_job'],'sensitivity_analysis_job':job,'dependency_proof':proof})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('generation','measurement'),default='generation')
    args=parser.parse_args()
    analysis.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
                     os.environ.get('SLURM_JOB_PARTITION')=='lrd_all_viz' and getpass.getuser()=='lmolfett' and
                     not socket.gethostname().startswith('login'), 'sensitivity attachment requires CPU Slurm step')
    result=attach(args.phase)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
