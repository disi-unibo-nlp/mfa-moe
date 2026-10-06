"""One-shot, budget-bounded legacy launch plans; no background orchestrator.

Implements the named legacy tasks with explicit caps and immutable receipts.
The separate 10.5 GPU-h study cannot be submitted through this program.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
SNAP = S / 'code/s1-f2ded3957eb54fd5'


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def wrapper(task):
    gpu = task.startswith('h14') or task.startswith('ordered-qualify') or (task == 'x2' or task.startswith('x2-nll'))
    source = SNAP / 'sbatch' / ('steer_gpu.sbatch' if gpu else 'steer_cpu.sbatch')
    text = source.read_text()
    # Keep temporary/cache/IPC staging at the location required by the user's AGENTS rules.
    if gpu:
        text = text.replace('RUN=$P/r/$SLURM_JOB_ID', 'RUN=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/g$SLURM_JOB_ID')
        # Historical broad HF_ capture could include a token; use public settings only.
        text = text.replace('"VLLM_", "STEER_", "HF_", "TRANSFORMERS_",',
                            '"VLLM_", "STEER_", "TRANSFORMERS_",')
        text = text.replace('names = {"PYTHONPATH",',
                            'names = {"HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_MODULES_CACHE", '
                            '"HF_HUB_DISABLE_TELEMETRY", "PYTHONPATH",')
        if task == 'x2':
            text = text.replace('CMD=("$VLLM_PY" -B -u -m "$MODULE" "$@" "${EXTRA[@]}")',
                                'CMD=("$VLLM_PY" -B -u "${STEER_GUARD:?frozen output guard required}" "$@" "${EXTRA[@]}")')
        if task.startswith('x2-nll') or task.startswith('ordered-qualify'):
            text = text.replace('run|qualify)', 'run|qualify|measure)')
            text = text.replace('    run) MODULE=moe_steer.runner;',
                '    measure) MODULE=native_nll; EXTRA=() ;;\n    run) MODULE=moe_steer.runner;')
            text = text.replace('CMD=("$VLLM_PY" -B -u -m "$MODULE" "$@" "${EXTRA[@]}")',
                'CMD=("$VLLM_PY" -B -u "${STEER_NLL_DRIVER:?frozen NLL driver required}" "$@")')
        if task.startswith('ordered-qualify'):
            text=text.replace('MOE_EXP_SRC=$STEER_CODE/moe_exp_src',
                'MOE_EXP_SRC=${STEER_ORDERED_SRC:?frozen ordered-worker overlay required}')
    else:
        text = text.replace('export STEER_TMP=$P/r/$SLURM_JOB_ID;',
                            'export STEER_TMP=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/c$SLURM_JOB_ID;')
    sha = hashlib.sha256(text.encode()).hexdigest()
    directory = S / 'addenda/launch' / sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / source.name
    if path.exists() and path.read_text() != text:
        raise ValueError('immutable launch wrapper changed')
    if not path.exists():
        path.write_text(text)
        path.chmod(0o400)
    command(['bash', '-n', str(path)])
    return path, sha


def plan(task):
    if pwd.getpwuid(os.getuid()).pw_name != 'lmolfett' or not os.uname().nodename.endswith('.leonardo.local'):
        raise ValueError('verified LEONARDO lmolfett identity required')
    if not (ROOT / 'swarm/DRAIN').exists() or 'DRAIN=yes' not in (ROOT / 'swarm/STATUS.md').read_text():
        raise ValueError('old keeper must acknowledge drain before a new launcher acts')
    frozen = json.loads((S / 'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']
    launch, launch_sha = wrapper(task)
    if task.startswith('h14'):
        snapshot = frozen['h14']['snapshot']
        env = f'ALL,STEER_CODE={snapshot},STEER_EXPECT_TREE={frozen["h14"]["tree_sha256"]},STEER_DEADLINE_MARGIN_S=30'
        recovery = task == 'h14-recovery'
        args = [f'--job-name=st-gpu-{10 if recovery else 9}', '--dependency=singleton',
                '--time=00:22:00' if recovery else '--time=00:14:30',
                '--comment=resume-v1:h14', f'--export={env}', str(launch),
                'qualify', '--check', 'h14', '--out', str(S / 'qualification/h14' /
                    ('resume-v1-recovery' if recovery else 'resume-v1'))]
        # Slurm rounds this seconds-bearing request up to fifteen whole minutes.
        ceiling, reserve = (.75, 2 * 1320 / 3600) if recovery else (.5, .5)
    elif (task == 'x2' or task.startswith('x2-nll')):
        production = json.loads((S / 'runs/x2-resume-v1/FROZEN.json').read_text())
        snapshot = frozen['s2']['snapshot']
        verdict = json.loads((S / 'qualification/h14/resume-v1-recovery/H14.json').read_text())
        if not verdict['pass']:
            raise ValueError('X2 engine qualification must pass')
        guard = Path(production['guard_path'])
        if hashlib.sha256(guard.read_bytes()).hexdigest() != production['guard_sha256']:
            raise ValueError('frozen output guard changed')
        env = (f'ALL,STEER_CODE={snapshot},STEER_EXPECT_TREE={frozen["s2"]["tree_sha256"]},'
               f'STEER_DEADLINE_MARGIN_S=300,STEER_GUARD={guard}')
        args = ['--job-name=st-gpu-11', '--dependency=singleton', '--time=03:00:00',
                '--comment=resume-v1:x2', f'--export={env}', str(launch), 'run',
                '--manifest', production['manifest'], '--output-root', str(S / 'runs/x2-resume-v1/outputs'),
                '--shard', '0', '--max-num-seqs', '48', '--return-routed-experts',
                '--margin-seconds', '600', '--abort-margin-seconds', '300']
        ceiling, reserve = 8., 6.
        if task.startswith('x2-nll'):
            nll=(json.loads((S/'runs/resume-v1/NLL_RECOVERY_AMENDMENT.json').read_text())['new_addendum']
                 if task == 'x2-nll-recovery' else
                 json.loads((S/'runs/resume-v1/NLL_UID_AMENDMENT.json').read_text())['new_addendum']
                 if task == 'x2-nll-v2' else frozen['nll'])
            if task == 'x2-nll-v2':
                rows=command(['sacct','-j','59095026','-n','-P','-o','JobID,State,ElapsedRaw,AllocTRES'])
                old=next(r for r in rows.splitlines() if r.startswith('59095026|')).split('|')
                if not old[1].startswith('CANCELLED') or int(old[2])!=0:
                    raise ValueError('old incompatible NLL job must be cancelled before any GPU allocation')
            directory=Path(nll['path'])
            inventory=json.loads((directory/'MANIFEST.json').read_text())
            # The NLL addendum was sealed separately from the frozen sampler.
            for rel,entry in inventory['files'].items():
                expected=entry['sha256'] if isinstance(entry,dict) else entry
                if hashlib.sha256((directory/rel).read_bytes()).hexdigest()!=expected:
                    raise ValueError('sealed native measurement code changed: '+rel)
            generation=json.loads((S/'runs/resume-v1/x2-submission.json').read_text())
            env=(f'ALL,STEER_CODE={snapshot},STEER_EXPECT_TREE={frozen["s2"]["tree_sha256"]},'
                 f'STEER_DEADLINE_MARGIN_S=30,STEER_NLL_DRIVER={directory}/native_nll.py')
            args=['--job-name=st-gpu-12',f'--dependency=afterok:{generation["job_id"]}',
                  '--kill-on-invalid-dep=yes','--time=00:44:00','--comment=resume-v1:x2-native-nll',
                  f'--export={env}',str(launch),'measure','--manifest',production['manifest'],
                  '--results',production['output'],'--parity',str(S/'qualification/parity'),
                  '--out',str(S/'runs/x2-resume-v1/native-nll'/production['manifest_sha256']/nll['tree_sha256']),
                  '--expect-tree',frozen['s2']['tree_sha256']]
            ceiling,reserve=1.5,88/60
            if task == 'x2-nll-recovery':
                amendment=json.loads((S/'runs/resume-v1/NLL_RECOVERY_AMENDMENT.json').read_text())
                rows=command(['sacct','-j','59093506,59095712','-n','-P','-o','JobID,State,ElapsedRaw'])
                previous={r.split('|')[0]:r.split('|') for r in rows.splitlines() if '.' not in r.split('|')[0]}
                if previous['59093506'][1]!='COMPLETED' or previous['59095712'][1]!='FAILED':
                    raise ValueError('generation and failed measurement must have their verified final states')
                actual=sum(int(r[2])*2/3600 for r in previous.values())
                if actual+1.5+.5>8 or abs(actual-amendment['budget']['actual_generation_and_measurement_GPUh'])>1e-8:
                    raise ValueError('complete recovery budget audit differs')
                prefixes=S/'runs/x2-resume-v1/measurement-prefixes/PREFIXES.json'
                prep=json.loads(prefixes.read_text())
                if prep['assigned_N_E_sequences']!=3354 or prep['teacher_forced_positions']>18556822:
                    raise ValueError('prepared teacher-forced workload exceeds the sealed complete cost projection')
                args[args.index('--time=00:44:00')]='--time=00:41:00'
                # The completed generation ID has left Slurm's dependency cache;
                # sacct and the complete artifact were verified above.
                args=[a for a in args if not a.startswith('--dependency=') and a!='--kill-on-invalid-dep=yes']
                args += ['--prefixes',str(prefixes)]
                reserve=(41*60+196)*2/3600
    elif task.startswith('ordered-qualify'):
        auth=json.loads((S/'runs/resume-v1/RESOURCE_AUTHORIZATION.json').read_text())
        if auth.get('routing_action_study_gpu_hours',0)<10.5:
            raise ValueError('the separate routing study requires explicit authorization')
        snapshot=frozen['s2']['snapshot']
        retry=task=='ordered-qualify-retry'
        cpu_recovery=task=='ordered-qualify-cpu-recovery'
        recovery_v2=task=='ordered-qualify-cpu-recovery-v2'
        prepared_path=S/'runs/ordered-qualification-v1'/('PREPARED.cpu-recovery-v2.json' if recovery_v2 else 'PREPARED.cpu-recovery.json' if cpu_recovery else 'PREPARED.retry.json' if retry else 'PREPARED.json')
        prepared=json.loads(prepared_path.read_text())
        if retry:
            rows=command(['sacct','-j','59096925','-n','-P','-o','JobID,State,ElapsedRaw'])
            old=next(r for r in rows.splitlines() if r.startswith('59096925|')).split('|')
            if old[1]!='FAILED' or int(old[2])*2/3600+(19*60+196)*2/3600>.75:
                raise ValueError('qualification retry including observed shutdown allowance exceeds .75 GPU-hour')
        for path,expected in prepared['files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('ordered qualifier source changed')
        env=(f'ALL,STEER_CODE={snapshot},STEER_EXPECT_TREE={frozen["s2"]["tree_sha256"]},'
            f'STEER_ORDERED_SRC={prepared["overlay"]},STEER_DEADLINE_MARGIN_S=30,STEER_NLL_DRIVER={prepared["driver"]}')
        args=['--job-name=st-gpu-13','--dependency=singleton','--time=00:20:00' if recovery_v2 else '--time=00:19:00' if retry or cpu_recovery else '--time=00:22:00',
            '--comment=resume-v1:ordered-qualification',f'--export={env}',str(launch),'measure',
            '--binding',str(prepared_path),'--overlay',prepared['overlay'],
            '--out',str(S/'runs/ordered-qualification-v1'/('results-cpu-recovery-v2' if recovery_v2 else 'results-cpu-recovery' if cpu_recovery else 'results-retry' if retry else 'results'))]
        ceiling,reserve=.75,(19*60+196)*2/3600 if retry or cpu_recovery else 44/60
        if cpu_recovery:
            args += ['--fixtures',prepared['fixtures']]
            audit=json.loads((Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')/'report/experimental-resume-v1/JOB_AUDIT.json').read_text())
            jobs=[j for j in audit['jobs'] if j['task'].startswith('ordered-qualify')]
            if not all(j['final'] for j in jobs) or sum(j['actual_resource_hours'] for j in jobs)+reserve>.9:
                raise ValueError('complete qualification recovery exceeds its proposed .9-hour stage ceiling')
            ceiling=.9
        if recovery_v2:
            args += ['--fixtures',prepared['fixtures']]
            audit=json.loads((Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')/'report/experimental-resume-v1/JOB_AUDIT.json').read_text())
            jobs=[j for j in audit['jobs'] if j['task'].startswith('ordered-qualify')]
            reserve=(20*60+196)*2/3600
            if (len(jobs)!=3 or not all(j['final'] for j in jobs)
                    or sum(j['actual_resource_hours'] for j in jobs)+reserve>1.10
                    or abs(prepared['budget']['maximum_stage_total_GPUh']-
                           (sum(j['actual_resource_hours'] for j in jobs)+reserve))>1e-8):
                raise ValueError('revised complete qualification price or prior allocation differs')
            ceiling=1.10
    elif task in ('m9-prepare','m10-prepare','paper-package','paper-package-final','paper-routing-supplement','x2-prefixes','x2-analysis','ordered-prefixes'):
        snapshot=frozen['s2']['snapshot']
        paper_final=ROOT/'paper/resume-v1/CODE_FINAL.json'
        if task=='paper-package-final':
            binding=json.loads(paper_final.read_text())
            directory=Path(binding['directory'])
            if binding['driver']!=str(directory/'paper_package.py'):
                raise ValueError('final paper driver binding differs')
            hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                for name in ('paper_package.py','uncertainty.py')}
            if hashes!=binding['files'] or hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=binding['tree_sha256']:
                raise ValueError('frozen final report code changed')
        if task=='paper-routing-supplement':
            supplement=json.loads((ROOT/'paper/resume-v1/CODE_ROUTING_SUPPLEMENT.json').read_text())
            supplement_driver=Path(supplement['driver'])
            if hashlib.sha256(supplement_driver.read_bytes()).hexdigest()!=supplement['driver_sha256']:
                raise ValueError('frozen routing supplement code changed')
        driver=(S/'runs/ordered-qualification-v1/code/ordered_prefixes.py' if task=='ordered-prefixes' else
            S/'runs/x2-resume-v1/code'/('x2_prefixes.py' if task=='x2-prefixes' else 'x2_analysis.py')
            if task.startswith('x2-') else S/'runs/m9-resume-v1/code/closure_driver.py' if task=='m9-prepare'
            else S/'runs/m10-resume-v1/code/probe_manifest.py' if task=='m10-prepare'
            else Path(binding['driver']) if task=='paper-package-final'
            else supplement_driver if task=='paper-routing-supplement'
            else ROOT/'paper/resume-v1/code/paper_package.py')
        if not driver.is_file():raise ValueError('prepare and review the stage driver before submission')
        if task.startswith('paper-') and not (ROOT/'depth-paths/results/resume-v1/results.json').is_file():
            raise ValueError('paper package requires the verified saved-cache depth summary')
        cpu_command='$VLLM_PY -B '+str(driver)
        if task=='m9-prepare':cpu_command+=' prepare --manifest '+str(S/'runs/m9-resume-v1/MANIFEST.json')
        args=[f'--job-name=st-{task}-resume-v1','--time=00:30:00' if task=='x2-analysis' else '--time=00:15:00','--cpus-per-task=1','--mem=8G',
            f'--comment=resume-v1:{task}',f'--export=ALL,STEER_CODE={snapshot},MOE_EXP_SRC={snapshot}/moe_exp_src',
            str(launch),'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '+cpu_command]
        if task in ('paper-package-final','paper-routing-supplement'):args.insert(1,'--qos=boost_qos_dbg')
        ceiling,reserve=(4. if task=='m9-prepare' else 12. if task=='m10-prepare' else 6.),(.5 if task=='x2-analysis' else .25)
        if task=='x2-analysis':
            measurement=json.loads((S/'runs/resume-v1/x2-nll-recovery-submission.json').read_text())
            args.insert(1,f'--dependency=afterany:{measurement["job_id"]}')
    elif task in ('rk1c', 'r3b', 'dp-s3', 'r3d-anticipation', 'r3d-anticipation-retry', 'r3d-readout', 'r3d-readout-retry', 'r3d-readout-final', 'r3d-readout-v3'):
        snapshot = str(SNAP)
        hours = 4 if task == 'rk1c' else 3 if task == 'r3b' else 2
        driver = (ROOT / 'reasoning-kinematics/rk1c/code/rk1c.py' if task == 'rk1c'
                  else ROOT / 'forum/tests/r3_dynamics_validity/code/r3b.py' if task == 'r3b'
                  else ROOT / 'depth-paths/results/resume-v1/code/depth_summary.py' if task == 'dp-s3'
                  else ROOT / 'forum/tests/r3_context/code/r3d_anticipation.py' if task.startswith('r3d-anticipation')
                  else ROOT / 'forum/tests/r3_context/code/r3d_readout.py')
        if task == 'r3b' and not (ROOT / 'reasoning-kinematics/rk1c/results.json').exists():
            raise ValueError('R3-B requires the completed clean descriptive stage')
        if task.startswith('r3d-') and not (ROOT / 'forum/tests/r3_dynamics_validity/estimates.json').exists():
            raise ValueError('R3-D requires completed R3-B')
        cpus, memory = (4, '64G') if task.startswith('r3d-') else (1, '8G')
        interpreter = '$VLLM_PY' if task.startswith('r3d-') or task == 'dp-s3' else '$CLIENT_PY'
        limit = ('00:15:00' if task == 'dp-s3' else '01:59:00' if task=='r3d-anticipation-retry' else '01:57:00' if task == 'r3d-readout-v3' else '01:58:00' if task == 'r3d-readout-final' else
                 '01:59:00' if task == 'r3d-readout-retry' else f'0{hours}:00:00')
        args = [f'--job-name=st-{task}-resume-v1', f'--time={limit}', f'--cpus-per-task={cpus}', f'--mem={memory}',
                f'--comment=resume-v1:{task}', f'--export=ALL,STEER_CODE={SNAP},MOE_EXP_SRC={SNAP}/moe_exp_src',
                str(launch), 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
                + interpreter + ' -B ' + str(driver)]
        ceiling, reserve = float(hours * cpus), float(hours * cpus)
        if task == 'dp-s3':
            ceiling, reserve = 6., .25
        if task == 'r3d-anticipation-retry':
            rows = command(['sacct', '-j', '59093809', '-n', '-P', '-o', 'JobID,State,ElapsedRaw,AllocCPUS'])
            row = next(r for r in rows.splitlines() if r.startswith('59093809|')).split('|')
            if row[1] != 'FAILED':
                raise ValueError('the failed anticipation initialization must be settled')
            reserve = 119 / 60 * 4
            if int(row[2]) * int(row[3]) / 3600 + reserve > 8.:
                raise ValueError('anticipation retry and failed initialization exceed eight core-hours')
        if task == 'r3d-readout-retry':
            state = command(['sacct', '-j', '59092785', '-n', '-P', '-o', 'JobID,State,ElapsedRaw,AllocCPUS'])
            first = next(row for row in state.splitlines() if row.startswith('59092785|')).split('|')
            if first[1] != 'FAILED' or int(first[2]) * int(first[3]) / 3600 > .06:
                raise ValueError('retry budget assumes the documented 25-second pre-fit environment failure')
            reserve = 119 / 60 * 4
            if reserve + int(first[2]) * int(first[3]) / 3600 > 8.:
                raise ValueError('readout retry exceeds its eight core-hour suballocation')
        if task == 'r3d-readout-final':
            total = 0.
            for previous in (59092785, 59092986):
                rows = command(['sacct', '-j', str(previous), '-n', '-P', '-o', 'JobID,State,ElapsedRaw,AllocCPUS'])
                row = next(r for r in rows.splitlines() if r.startswith(str(previous) + '|')).split('|')
                if row[1] not in ('COMPLETED', 'FAILED'):
                    raise ValueError('previous preparation attempts must be settled')
                total += int(row[2]) * int(row[3]) / 3600
            reserve = 118 / 60 * 4
            if total + reserve > 8.:
                raise ValueError('all clean-readout attempts would exceed eight core-hours')
        if task == 'r3d-readout-v3':
            total = 0.
            for previous in (59092785, 59092986, 59093150):
                rows = command(['sacct', '-j', str(previous), '-n', '-P', '-o', 'JobID,State,ElapsedRaw,AllocCPUS'])
                row = next(r for r in rows.splitlines() if r.startswith(str(previous) + '|')).split('|')
                if row[1] not in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    raise ValueError('all previous readout preparations must be settled')
                total += int(row[2]) * int(row[3]) / 3600
            reserve = 117 / 60 * 4
            if total + reserve > 8.:
                raise ValueError('clean-readout attempts exceed eight core-hours')
    else:
        raise ValueError('unimplemented or proposed workload')
    identity = json.loads(command([str(Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python')),
        '-B', str(Path(snapshot) / 'scripts/freeze_steer.py'), 'verify', snapshot]))
    return {'task': task, 'argv': ['sbatch', '--parsable', *args], 'snapshot': identity,
        'launch_wrapper': str(launch), 'launch_sha256': launch_sha,
        'unit': 'GPU-h' if task.startswith('h14') or task.startswith('ordered-qualify') or (task == 'x2' or task.startswith('x2-nll')) else 'CPU core-h', 'ceiling': ceiling,
        'maximum_reserved': reserve,
        'authorization': ('explicit user authorization: .75 additional GPU-hour for H1-H4 recovery'
                          if task == 'h14-recovery' else
                          'current user request to implement the named legacy workload and its stated ceiling')}


def run(task, submit):
    receipt_dir = S / 'runs/resume-v1'
    receipt_dir.mkdir(exist_ok=True)
    prepared = plan(task)
    (receipt_dir / f'{task}-plan.json').write_text(json.dumps(prepared, indent=1) + '\n')
    if not submit:
        print(json.dumps(prepared, indent=1))
        return
    if task == 'h14-recovery':
        approval = receipt_dir / 'RESOURCE_AUTHORIZATION.json'
        if not approval.exists() or json.loads(approval.read_text()).get('h14_recovery_gpu_hours', 0) < .75:
            raise ValueError('additional qualification recovery budget needs the user\'s explicit approval')
        prepared['authorization'] = json.loads(approval.read_text())
        (receipt_dir / f'{task}-plan.json').write_text(json.dumps(prepared, indent=1) + '\n')
    if task == 'ordered-qualify-cpu-recovery':
        approval=receipt_dir/'QUALIFICATION_RECOVERY_AUTHORIZATION.json'
        if not approval.is_file() or json.loads(approval.read_text()).get('additional_qualification_GPU_hours',0)<.15:
            raise ValueError('the sealed .75-hour qualification ceiling requires explicit approval for this .15-hour increase')
        prepared['authorization']=json.loads(approval.read_text())
        (receipt_dir/f'{task}-plan.json').write_text(json.dumps(prepared,indent=1)+'\n')
    if task == 'ordered-qualify-cpu-recovery-v2':
        approval=receipt_dir/'QUALIFICATION_RECOVERY_V2_AUTHORIZATION.json'
        if (not approval.is_file() or
                json.loads(approval.read_text()).get('additional_qualification_GPU_hours',0)<.20):
            raise ValueError('the new one-shot .20 GPU-hour stage increase requires explicit user authorization')
        prepared['authorization']=json.loads(approval.read_text())
        (receipt_dir/f'{task}-plan.json').write_text(json.dumps(prepared,indent=1)+'\n')
    receipt = receipt_dir / f'{task}-submission.json'
    if receipt.exists():
        raise ValueError('this one-shot workload was already submitted; inspect its receipt, never resubmit silently')
    queue = command(['squeue', '-h', '-u', 'lmolfett', '-o', '%i|%j|%T|%b'])
    slot = 'st-gpu-13' if task.startswith('ordered-qualify') else 'st-gpu-12' if task.startswith('x2-nll') else 'st-gpu-11' if task == 'x2' else 'st-gpu-10' if task == 'h14-recovery' else 'st-gpu-9'
    if (task.startswith('h14') or task.startswith('ordered-qualify') or (task == 'x2' or task.startswith('x2-nll'))) and any(f'|{slot}|' in row for row in queue.splitlines()):
        raise ValueError('qualification slot occupied')
    job = int(command(prepared['argv']).split(';')[0])
    record = {**prepared, 'job_id': job, 'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'verified': False}
    # Record the ID before verification: a scheduler read failure must not lose a submitted job.
    receipt.write_text(json.dumps(record, indent=1) + '\n')
    state = command(['scontrol', 'show', 'job', str(job)])
    if f'JobId={job}' not in state or 'UserId=lmolfett' not in state or 'Account=iscrc_miosr' not in state:
        raise ValueError(f'submission {job} identity verification failed; inspect it before any other launch')
    record.update(verified=True, scontrol=state)
    receipt.write_text(json.dumps(record, indent=1) + '\n')
    with (S / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'ts': record['submitted_utc'], 'task': task, 'job_id': job,
            'status': 'submitted and verified', 'source': 'resume-v1',
            'ceiling': prepared['ceiling'], 'unit': prepared['unit'], 'receipt': str(receipt)}) + '\n')
    print(json.dumps({'task': task, 'job_id': job, 'verified': True, 'ceiling': prepared['ceiling'],
                      'unit': prepared['unit']}, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('h14', 'h14-recovery', 'x2', 'x2-nll', 'x2-nll-v2','x2-nll-recovery','x2-prefixes','x2-analysis','ordered-prefixes', 'rk1c', 'r3b', 'dp-s3', 'r3d-anticipation', 'r3d-anticipation-retry', 'r3d-readout', 'r3d-readout-retry', 'r3d-readout-final', 'r3d-readout-v3','m9-prepare','m10-prepare','paper-package','paper-package-final','paper-routing-supplement','ordered-qualify','ordered-qualify-retry','ordered-qualify-cpu-recovery','ordered-qualify-cpu-recovery-v2'))
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    run(args.task, args.submit)
