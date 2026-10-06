"""Recover pre-fix queued CPU jobs whose stored environment retained affinity."""
import os
from pathlib import Path
import subprocess
import dispatch_overnight_readers_v1 as d


def main():
    directory=d.DOC/'overnight-dense-affinity-recovery-submissions-v1';directory.mkdir(exist_ok=True)
    records=[]
    for stage, tag, source_job in [('A','9b714fe1091f1d16','59347821'),('B','9c3dc95fda9cf5e9','59345850')]:
        old=d.DOC/('generated-dense-chain-'+tag+'-8e0bff870bed')
        original=d.base.sealed(old/'CHAIN.json')
        prep=d.base.sealed(old/'dense-cpu-prepare.json')
        dispatch=d.base.sealed(old/'dense-gpu-dispatch.json')
        account=subprocess.run(['sacct','-X','-nP','-j',source_job,'--format=JobID,State,ExitCode'],check=True,capture_output=True,text=True).stdout
        if [source_job,'COMPLETED','0:0'] not in [line.strip().split('|') for line in account.splitlines()]:
            raise ValueError('completed prerequisite not verified')
        binding={'old_chain_sha256':original['sha256'],'completed_prerequisite_job':source_job,'prerequisite_sacct':account,
                 'shared_submit_sha256':d.base.file_sha(d.__file__),'recovery_source_sha256':d.base.file_sha(__file__),
                 'scope':'Only queued scheduler environment changes. Prepared B frame/price reused; A preparation did not execute. Scientific sources and outputs unchanged.'}
        preparation=None
        if stage=='A':
            arguments=[x for x in prep['arguments'] if not x.startswith(('--dependency=','--job-name='))]
            preparation=d.submit(directory,'a-prepare',['--job-name=dense-A-prepare-affinity-r1',*arguments],os.environ.copy(),binding)
        arguments=[x for x in dispatch['arguments'] if not x.startswith(('--dependency=','--job-name='))]
        prefix=['--dependency=afterok:'+preparation] if preparation else []
        launched=d.submit(directory,stage.lower()+'-dispatch',[*prefix,'--job-name=dense-'+stage+'-dispatch-affinity-r1',*arguments],os.environ.copy(),binding)
        records.append({'stage':stage,'preparation_job':preparation or source_job,'dispatch_job':launched})
    d.save(directory/'CHAIN.json',{'schema':'dense-parent-affinity-recovery-v1','records':records})


if __name__=='__main__':main()
