"""Authorized one-shot recovery: preserve failed jobs and reattach exact stages."""
from __future__ import annotations
import argparse
import fcntl
import os
import subprocess
from pathlib import Path
import dispatch_overnight_readers_v1 as d

DIRECTORY = d.DOC / 'overnight-operational-recovery-submissions-v3'
SCRIPTS = d.REPO / 'scripts/experimental_resume'


def verify(job):
    result = subprocess.run(['scontrol', 'show', 'job', str(job)], check=True,
                            capture_output=True, text=True).stdout
    if f'JobId={job}' not in result or 'UserId=lmolfett(' not in result:
        raise ValueError('job identity/ownership differs')
    return result


def bind_files(*names):
    paths = [Path(__file__), Path(d.__file__), *[SCRIPTS / name for name in names]]
    for path in paths:
        if path.suffix == '.sbatch':
            subprocess.run(['bash', '-n', str(path)], check=True)
    return {str(path.resolve()): d.base.file_sha(path) for path in paths}


def rebind(job, old_parent, new_parent):
    name = DIRECTORY / f'rebound-{job}.json'
    if name.exists():
        record = d.base.sealed(name)
        if record['new_parent'] != new_parent or record['old_parent'] != old_parent:
            raise ValueError('rebound receipt differs')
        return
    before = verify(job)
    expected = 'afterok:' + str(new_parent)
    if expected not in before:
        if 'JobState=PENDING' not in before or 'afterok:' + str(old_parent) not in before:
            raise ValueError('dependent is no longer pending on the expected original parent: ' + job)
        d.save(DIRECTORY / f'rebind-{job}.attempt.json', {
            'job_id': job, 'old_parent': old_parent, 'new_parent': new_parent, 'before': before})
        subprocess.run(['scontrol', 'update', 'JobId=' + job, 'Dependency=' + expected], check=True)
    after = verify(job)
    if expected not in after:
        raise ValueError('new dependency did not verify')
    d.save(name, {'schema': 'overnight-dependency-amendment-v1', 'job_id': job,
                 'old_parent': old_parent, 'new_parent': new_parent, 'after': after})
    print('REBOUND', job, new_parent, flush=True)


def cancel_superseded(job):
    path = DIRECTORY / f'superseded-{job}.json'
    if path.exists():
        d.base.sealed(path)
        return
    before = verify(job)
    if 'JobState=PENDING' in before:
        subprocess.run(['scancel', job], check=True)
    elif 'JobState=CANCELLED' not in before:
        raise ValueError('superseded job changed state; inspect before cancelling: ' + job)
    after = verify(job)
    if 'JobState=CANCELLED' not in after:
        raise ValueError('cancellation did not verify')
    d.save(path, {'schema': 'overnight-superseded-pending-job-v1', 'job_id': job,
                 'reason': 'Verified replacement and dependent rewiring completed; no GPU result discarded.',
                 'before': before, 'after': after})


def ab():
    wrapper = SCRIPTS / 'build_price_overnight_semantics_adjudicated_v1.sbatch'
    plan = d.base.sealed(d.DOC / 'OVERNIGHT_AB_DOSE_ADJUDICATION_PLAN_v1.json')
    files = bind_files(wrapper.name, 'adjudicate_overnight_ab_seals_v1.py')
    records = []
    for lane, array, old_price, reader, dense in [
            ('A', '59344807', '59344822', '59344824', '59345851'),
            ('B', '59344808', '59344826', '59344828', '59345850')]:
        manifest_path, manifest, measurement, _ = d.paths(lane)
        envelope = d.base.sealed(d.DOC / f'OVERNIGHT_DISCOVERY_{lane}_ENVELOPE_v1.json')
        env = os.environ.copy()
        env.update(OVERNIGHT_LANE=lane, OVERNIGHT_MANIFEST=str(manifest_path),
            OVERNIGHT_GENERATION_PRICE=str(d.DOC / f'OVERNIGHT_DISCOVERY_{lane}_PRICE_v1.json'),
            OVERNIGHT_GENERATION_OUT=str(d.RUNS / ('overnight-routing-v1-' + manifest['sha256'][:16])),
            OVERNIGHT_MEASUREMENT_OUT=str(measurement),
            OVERNIGHT_READER_GPU_HOUR_CEILING=str(envelope['semantic_measurement']['measurement_gpu_hour_ceiling']))
        binding = {'manifest_sha256': manifest['sha256'], 'generation_job': array,
                   'adjudication_plan_sha256': plan['sha256'], 'files': files}
        price = d.submit(DIRECTORY, lane.lower() + '-frame-price-adjudicated',
            ['--dependency=afterany:' + array, '--job-name=st-overnight-' + lane + '-price-adjudicated', str(wrapper)], env, binding)
        rebind(reader, old_price, price)
        rebind(dense, old_price, price)
        cancel_superseded(old_price)
        records.append({'stage': lane, 'generation_job': array, 'frame_price_job': price,
                        'reader_dispatch_job': reader, 'dense_preparation_job': dense})
    d.save(DIRECTORY / 'AB_CHAIN.json', {'schema': 'overnight-ab-adjudicated-chain-v3', 'records': records})


def operators():
    amendment = d.base.sealed(d.DOC / 'OVERNIGHT_OPERATOR_CPU_ADJUDICATION_AMENDMENT_v3.json')
    for path, sha in amendment['source_files'].items():
        if d.base.file_sha(path) != sha:
            raise ValueError('operator amendment source changed: ' + path)
    wrappers = ['adjudicate_overnight_operator_qualification_v3.sbatch',
                'prepare_overnight_operator_stage_v3.sbatch', 'dispatch_overnight_stage_v3.sbatch']
    binding = {'amendment_sha256': amendment['sha256'], 'files': bind_files(*wrappers)}
    adjudication = d.submit(DIRECTORY, 'operator-qualification-adjudication',
        ['--dependency=afterany:59344841', str(SCRIPTS / wrappers[0])], os.environ.copy(), binding)
    records = []
    for stage, old_prep, old_dispatch, dense, paper in [
            ('C', '59345112', '59345508', '59345819', '59345668'),
            ('FRESH', '59345113', '59345513', '59345821', '59345670')]:
        env = os.environ.copy(); env.update(OVERNIGHT_STAGE=stage, OVERNIGHT_DISPATCH_PHASE='generation')
        dependency = 'afterok:' + adjudication + (':59344223' if stage == 'FRESH' else '')
        preparation = d.submit(DIRECTORY, stage.lower() + '-prepare',
            ['--dependency=' + dependency, '--job-name=st-overnight-' + stage + '-prepare-v3', str(SCRIPTS / wrappers[1])],
            env, {**binding, 'stage': stage})
        dispatcher = d.submit(DIRECTORY, stage.lower() + '-generation-dispatch',
            ['--dependency=afterok:' + preparation, '--job-name=st-overnight-' + stage + '-launch-v3', str(SCRIPTS / wrappers[2])],
            env, {**binding, 'stage': stage, 'preparation_job': preparation})
        rebind(dense, old_dispatch, dispatcher)
        rebind(paper, old_dispatch, dispatcher)
        cancel_superseded(old_dispatch)
        cancel_superseded(old_prep)
        records.append({'stage': stage, 'qualification_adjudication_job': adjudication,
                        'preparation_job': preparation, 'dispatcher_job': dispatcher,
                        'dense_attach_job': dense, 'paper_attach_job': paper})
    d.save(DIRECTORY / 'OPERATOR_CHAIN.json', {'schema': 'overnight-operator-chain-v3', 'records': records})


def cap():
    import mechanism_semantic_cap_recovery_v2 as recovery
    plan = d.base.sealed(recovery.PLAN); recovery.validate(plan)
    if plan['estimated_complete_gpu_hours'] > plan['complete_gpu_hour_ceiling'] or len(plan['generation_price']['shards']) != 5:
        raise ValueError('unchanged cap recovery resource envelope differs')
    wrappers = ['run_mechanism_semantic_cap_recovery_v2.sbatch', 'analyze_mechanism_semantic_cap_recovery_v2.sbatch']
    binding = {'manifest_sha256': plan['sha256'], 'files': bind_files(*wrappers),
               'previous_failed_qualification_job': '59346750',
               'preservation': 'Previous job failed at srun affinity before model launch; cancelled children had zero runtime. Same prompts, seeds, UIDs, caps and source bindings.'}
    env = os.environ.copy(); env.update(CAP_RECOVERY_MODE='qualification', CAP_RECOVERY_RESUME_UNCOMMITTED='0')
    qualification = d.submit(DIRECTORY, 'cap-qualification',
        ['--time=01:00:00', '--job-name=mechanism-cap-qual-affinity-r1', str(SCRIPTS / wrappers[0])], env, binding)
    env['CAP_RECOVERY_MODE'] = 'generation'
    generation = d.submit(DIRECTORY, 'cap-generation',
        ['--dependency=afterok:' + qualification, '--array=0-4', '--time=02:00:00',
         '--job-name=mechanism-cap-replay-affinity-r1', str(SCRIPTS / wrappers[0])], env, binding)
    analysis = d.submit(DIRECTORY, 'cap-analysis',
        ['--dependency=afterok:' + generation, '--job-name=mechanism-cap-analysis-affinity-r1', str(SCRIPTS / wrappers[1])], env, binding)
    d.save(DIRECTORY / 'CAP_CHAIN.json', {'schema': 'mechanism-cap-affinity-recovery-v1', 'binding': binding,
        'qualification_job': qualification, 'generation_job': generation, 'analysis_job': analysis})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('ab', 'operators', 'cap'), required=True)
    args = parser.parse_args()
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'ab': ab, 'operators': operators, 'cap': cap}[args.phase]()


if __name__ == '__main__':
    main()
