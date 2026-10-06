"""Finish recovered chains with already-completed prerequisites verified in sacct."""
import fcntl
import os
import subprocess
from pathlib import Path
import dispatch_overnight_readers_v1 as d
import submit_overnight_operational_recovery_v3 as prior

DIRECTORY = d.DOC / 'overnight-remaining-attachments-submissions-v3'
SCRIPTS = prior.SCRIPTS


def main():
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        enrollment = d.base.sealed(d.DOC / 'MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_v1.json')
        history = subprocess.run(['sacct', '-X', '-nP', '-j', '59344223',
            '--format=JobID,State,ExitCode'], check=True, capture_output=True, text=True).stdout
        if ['59344223', 'COMPLETED', '0:0'] not in [line.strip().split('|') for line in history.splitlines()] or enrollment['status'] != 'READY_EXACT_NATIVE_PREFIXES':
            raise ValueError('enrollment has not completed successfully with its expected artifact')
        qualification = d.base.sealed(prior.DIRECTORY / 'operator-qualification-adjudication.json')['job_id']
        c_dispatch = d.base.sealed(prior.DIRECTORY / 'c-generation-dispatch.json')['job_id']
        wrappers = ['prepare_overnight_operator_stage_v3.sbatch', 'dispatch_overnight_stage_v3.sbatch']
        binding = {'files': prior.bind_files(*wrappers), 'enrollment_sha256': enrollment['sha256'],
            'completed_enrollment_job': '59344223', 'enrollment_sacct': history,
            'scheduler_note': 'Completed enrollment aged out of scheduler dependency cache; successful accounting and sealed artifact verified before submit.',
            'qualification_adjudication_job': qualification}
        env = os.environ.copy(); env.update(OVERNIGHT_STAGE='FRESH', OVERNIGHT_DISPATCH_PHASE='generation')
        preparation = d.submit(DIRECTORY, 'fresh-prepare',
            ['--dependency=afterok:' + qualification, '--job-name=st-overnight-FRESH-prepare-v3', str(SCRIPTS / wrappers[0])], env, binding)
        dispatcher = d.submit(DIRECTORY, 'fresh-generation-dispatch',
            ['--dependency=afterok:' + preparation, '--job-name=st-overnight-FRESH-launch-v3', str(SCRIPTS / wrappers[1])],
            env, {**binding, 'preparation_job': preparation})
        for job in ('59345821', '59345670'):
            prior.rebind(job, '59345513', dispatcher)
        prior.cancel_superseded('59345513')
        prior.cancel_superseded('59345113')
        first_stage_wrapper = SCRIPTS / 'submit_routing_first_stage_v2.sbatch'
        plan = d.base.sealed(d.DOC / 'ROUTING_FIRST_STAGE_PLAN_v2.json')
        subprocess.run(['bash', '-n', str(first_stage_wrapper)], check=True)
        first_stage_binding = {'plan_sha256': plan['sha256'], 'files': prior.bind_files(first_stage_wrapper.name)}
        ab = d.base.sealed(prior.DIRECTORY / 'AB_CHAIN.json')
        rows = []
        for stage in ('A', 'B', 'C', 'FRESH'):
            arguments = ['--job-name=routing-engagement-' + stage + '-attach']
            if stage in ('A', 'B'):
                frame = next(row['frame_price_job'] for row in ab['records'] if row['stage'] == stage)
                arguments.extend([str(first_stage_wrapper), '--stage', stage, '--frame-price-job', frame])
            else:
                arguments.extend(['--dependency=afterok:' + (c_dispatch if stage == 'C' else dispatcher),
                                  str(first_stage_wrapper), '--stage', stage])
            job = d.submit(DIRECTORY, stage.lower() + '-routing-engagement-attach', arguments,
                           os.environ.copy(), {**first_stage_binding, 'stage': stage})
            rows.append({'stage': stage, 'attachment_job': job})
        d.save(DIRECTORY / 'CHAIN.json', {'schema': 'overnight-remaining-attachments-v3',
            'fresh_preparation_job': preparation, 'fresh_generation_dispatch_job': dispatcher,
            'c_generation_dispatch_job': c_dispatch, 'routing_first_stage_attachments': rows})


if __name__ == '__main__':
    main()
