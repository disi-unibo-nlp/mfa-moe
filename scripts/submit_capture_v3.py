#!/usr/bin/env python3
"""Campaign-v3 capture driver: plan -> shared verify -> cohort shards -> gate, all in Slurm.

Called by the generation-side stage script once review/T2.json is approved:
    submit_capture_v3.py start --model M --cohort B|A --bundle BUNDLE
Every later step runs inside a CPU job that this script submits, and each step submits the
next one, so the chain advances without an agent:
    plan-verify   verification-only plan (label-free, shared by A and B) -> GPU verify job
    after-verify  verify receipt passed? plan waiting cohorts; drained/failed -> bounded retry
    plan-cohort   production manifests sized from the receipt -> shard GPU jobs -> gate
    gate          capture_gate; incomplete shards are resubmitted while they make progress
Cohort A additionally needs the label gate (its annotations are bound into the manifests).
Stdlib only: it runs under the client interpreter and only shells out to the frozen bundle.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

P = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe')
D = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3')
STORAGE = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3')
CPU_JOB = D / 'sbatch/cpu_boost.sbatch'
# frozen, read-only copy of routing-study/sbatch/campaign_v3_capture.sbatch (checked before use)
CAPTURE_JOB = D / 'code/capture_sbatch-c633a2576c09bd36.sbatch'
CAPTURE_JOB_SHA256 = 'c633a2576c09bd3618d304c06fa0d4099cc46562ae95fe2ce8916628c41ae130'
CLIENT_PY = P / 'envs/correlation-client-3.11/bin/python'
VLLM_PY = P / 'envs/vllm-cu129/bin/python'
STATE = D / 'capture'
MAX_ROUNDS = 6
MAX_WALLTIME = 24 * 3600
# The longest 131k attempt runs first and is unmeasured; a walltime kill mid-attempt loses it,
# while Slurm bills only used time, so verify gets the planner's 16 h maximum.
VERIFY_SECONDS = 16 * 3600


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(model, event, **fields):
    with state(f'{model}.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(event=event, utc=now(), **fields)) + '\n')
    print(json.dumps(dict(model=model, event=event, **fields)), flush=True)


def sbatch(args, model, stage):
    if str(CAPTURE_JOB) in args and hashlib.sha256(CAPTURE_JOB.read_bytes()).hexdigest() != CAPTURE_JOB_SHA256:
        raise SystemExit('frozen capture launcher changed: ' + str(CAPTURE_JOB))
    # jobs submitted from inside a Slurm step must not inherit its SLURM_* variables
    # (an inherited SLURM_CPU_BIND makes the child's srun fail)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_', 'SRUN_', 'SALLOC_'))}
    out = subprocess.run(['sbatch', '--parsable', *args], capture_output=True, text=True, check=True, env=env)
    job = int(out.stdout.strip().split(';')[0])
    with (D / 'ledger.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(stage=f'capture-{stage}-{model}', job_id=job, submitted_utc=now(),
                                     args=args)) + '\n')
    log(model, 'submitted', stage=stage, job=job)
    return job


def cpu_step(model, step, bundle, *extra, dependency=None, name=None, mem='32G', hours=4):
    command = (f'{CLIENT_PY} {Path(__file__).resolve()} {step} --model {model} --bundle {bundle} '
               + ' '.join(extra))
    dep = [f'--dependency={dependency}', '--kill-on-invalid-dep=yes'] if dependency else []
    return sbatch([f'--job-name={name or f"v3-cap{step}-{model}"}', f'--time={hours:02d}:00:00',
                   f'--mem={mem}', *dep, str(CPU_JOB), command], model, step)


def bundle_module(bundle, module, *args):
    """Run one frozen CPU entrypoint inside the current Slurm CPU allocation."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f'{bundle}/src:{bundle}:{bundle}/scripts:{P}/envs/vllm-cu129/lib/python3.11/site-packages')
    argv = [str(VLLM_PY), '-B', str(Path(bundle) / 'scripts/run_campaign_v3_module.py'), '--bundle', str(bundle),
            '--module', module, *map(str, args)]
    result = subprocess.run(argv, cwd=bundle, env=env, capture_output=True, text=True)
    sys.stdout.write(result.stdout[-4000:])
    sys.stderr.write(result.stderr[-4000:])
    return result


def cohorts_file(model):
    return D / 'cohorts' / model / 'cohorts.json'


def verify_manifest(model):
    return STORAGE / 'manifests' / model / 'verify.json'


def verify_receipt(model):
    return STORAGE / 'capture' / model / 'verify' / 'verification.json'


def verified(model):
    path = verify_receipt(model)
    return path.exists() and json.loads(path.read_text()).get('status') == 'passed'


def walltime(seconds):
    seconds = min(max(int(seconds), 3600), MAX_WALLTIME)
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00'


def state(name):
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / name


def pending(model):
    return state(f'{model}.pending.json')


def mark_pending(model, cohort):
    path = pending(model)
    value = set(json.loads(path.read_text())) if path.exists() else set()
    value.add(cohort)
    path.write_text(json.dumps(sorted(value)))


def annotations(model):
    return sorted((D / 'labels' / model).glob('part-*/annotations.json'))


def start(a):
    if a.cohort == 'A':
        receipt = D / 'acceptance' / f'labels-{a.model}.json'
        if not (receipt.exists() and json.loads(receipt.read_text()).get('status') == 'complete'):
            raise SystemExit(f'{a.model}: cohort A needs the accepted label gate')
    if verified(a.model):
        cpu_step(a.model, 'plan-cohort', a.bundle, f'--cohort {a.cohort} --round 1')
        return
    mark_pending(a.model, a.cohort)
    if not verify_manifest(a.model).exists():
        cpu_step(a.model, 'plan-verify', a.bundle)
    elif not any(queued(f'{prefix}-{a.model}') for prefix in
                 ('v3-capplan-verify', 'v3-capverify', 'v3-capafter-verify')):
        log(a.model, 'verify-rearmed')  # planned earlier but its chain is gone
        submit_verify(a, 1)
    # otherwise a verify is queued; its after-verify step picks the pending cohort up


def queued(name):
    out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', 'lmolfett'), '-n', name, '-o', '%i'],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def plan_verify(a):
    result = bundle_module(a.bundle, 'measurement_study.production_plan', a.model,
                           '--cohort-manifest', cohorts_file(a.model), '--storage-root', STORAGE,
                           '--cohort', 'B', '--verification-only', '--verification-seconds', VERIFY_SECONDS)
    if result.returncode:
        log(a.model, 'plan-verify-failed', code=result.returncode)
        raise SystemExit(result.returncode)
    submit_verify(a, 1)


def submit_verify(a, round_):
    manifest = json.loads(verify_manifest(a.model).read_text())
    seconds = manifest.get('requested_seconds') or 5 * 3600
    job = sbatch([f'--job-name=v3-capverify-{a.model}', f'--time={walltime(seconds)}', str(CAPTURE_JOB),
                  str(a.bundle), a.model, str(verify_manifest(a.model)), 'verify'], a.model, f'verify-r{round_}')
    cpu_step(a.model, 'after-verify', a.bundle, f'--round {round_}', dependency=f'afterany:{job}', hours=1)


def after_verify(a):
    if not verified(a.model):
        drained = verify_receipt(a.model).with_name('drained.json')
        if not drained.exists():
            # a hard failure (differential, load, coverage) is deterministic: do not reload
            log(a.model, 'verify-failed', round=a.round)
            raise SystemExit(f'{a.model}: verification failed without draining (round {a.round})')
        if a.round >= MAX_ROUNDS:
            log(a.model, 'verify-failed', rounds=a.round)
            raise SystemExit(f'{a.model}: verification did not pass after {a.round} rounds')
        drained.rename(drained.with_name(f'drained-r{a.round}.json'))  # consumed by this retry
        log(a.model, 'verify-retry', round=a.round + 1)
        submit_verify(a, a.round + 1)
        return
    receipt = json.loads(verify_receipt(a.model).read_text())
    log(a.model, 'verified', measurements=receipt.get('measurements'))
    cohorts = json.loads(pending(a.model).read_text()) if pending(a.model).exists() else []
    for cohort in cohorts:
        cpu_step(a.model, 'plan-cohort', a.bundle, f'--cohort {cohort} --round 1')
    pending(a.model).unlink(missing_ok=True)


def shard_state(model, cohort):
    ready = json.loads((STORAGE / 'manifests' / model / cohort / 'ready.json').read_text())
    shards, done = [], 0
    for shard in ready['shards']:
        manifest_path = STORAGE / 'manifests' / model / cohort / f"{shard['index']}.json"
        manifest = json.loads(manifest_path.read_text())
        out = Path(manifest['capture_dir'])
        result = out / 'result.json'
        complete = result.exists() and json.loads(result.read_text()).get('status') == 'complete_native_collection'
        done += len(list(out.glob('*.complete.json'))) if out.exists() else 0
        shards.append(dict(index=shard['index'], manifest=manifest_path, complete=complete,
                           seconds=manifest.get('requested_seconds') or shard.get('requested_seconds')))
    return shards, done


def gpu_slot():
    """Shared campaign GPU slot pool (same file and counter as the generation-side stage script):
    names v3-gpu-<k> with --dependency=singleton cap label parts and capture shards together."""
    config = D / 'gpu_slots.json'
    slots = int(json.loads(config.read_text())['slots']) if config.exists() else 6
    with (D / 'gpu_slots.counter').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        n = int(handle.read().strip() or 0)
        handle.seek(0)
        handle.truncate()
        handle.write(str(n + 1))
    return [f'--job-name=v3-gpu-{n % slots}', '--dependency=singleton']


def submit_shards(a, shards, round_):
    jobs = []
    for shard in (s for s in shards if not s['complete']):
        jobs.append(sbatch([*gpu_slot(),
                            f'--time={walltime(shard["seconds"])}', str(CAPTURE_JOB), str(a.bundle), a.model,
                            str(shard['manifest']), 'capture'], a.model, f'{a.cohort}{shard["index"]}-r{round_}'))
    cpu_step(a.model, 'gate', a.bundle, f'--cohort {a.cohort} --round {round_}',
             dependency=('afterany:' + ':'.join(map(str, jobs))) if jobs else None, hours=2)


def plan_cohort(a):
    extra = ['--annotations', *annotations(a.model)] if a.cohort == 'A' else []
    if a.cohort == 'A' and len(extra) != 5:
        raise SystemExit(f'{a.model}: expected 4 label parts, found {len(extra) - 1}')
    result = bundle_module(a.bundle, 'measurement_study.production_plan', a.model,
                           '--cohort-manifest', cohorts_file(a.model), '--storage-root', STORAGE,
                           '--cohort', a.cohort, '--verification-receipt', verify_receipt(a.model), *extra)
    if result.returncode:
        log(a.model, 'plan-cohort-failed', cohort=a.cohort, code=result.returncode)
        raise SystemExit(result.returncode)
    shards, done = shard_state(a.model, a.cohort)
    log(a.model, 'planned', cohort=a.cohort, shards=len(shards),
        hours=sum(s['seconds'] for s in shards) / 3600)
    submit_shards(a, shards, 1)


def gate(a):
    manifest_dir = STORAGE / 'manifests' / a.model / a.cohort
    shards, done = shard_state(a.model, a.cohort)
    progress = state(f'{a.model}-{a.cohort}.progress.json')
    before = json.loads(progress.read_text())['done'] if progress.exists() else -1
    progress.write_text(json.dumps(dict(done=done, round=a.round)))
    if any(not s['complete'] for s in shards):
        if a.round >= MAX_ROUNDS or done <= before:
            log(a.model, 'capture-stalled', cohort=a.cohort, round=a.round, done=done)
            raise SystemExit(f'{a.model} {a.cohort}: incomplete shards and no progress (round {a.round})')
        submit_shards(a, shards, a.round + 1)
        return
    output = D / 'acceptance' / f'capture-{a.model}-{a.cohort}.json'
    result = bundle_module(a.bundle, 'measurement_study.capture_gate', '--manifest-dir', manifest_dir,
                           '--output', output)
    if result.returncode:
        log(a.model, 'gate-failed', cohort=a.cohort, code=result.returncode)
        raise SystemExit(result.returncode)
    log(a.model, 'accepted', cohort=a.cohort, receipt=str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=('start', 'plan-verify', 'after-verify', 'plan-cohort', 'gate'))
    p.add_argument('--model', required=True, choices=('gpt', 'glm', 'qwen36', 'nemotron', 'gemma', 'qwen330b', 'qwen35'))
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--cohort', choices=('A', 'B'))
    p.add_argument('--round', type=int, default=1)
    a = p.parse_args()
    if a.step != 'start' and not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('capture steps run inside Slurm CPU jobs')
    if a.step in ('start', 'plan-cohort', 'gate') and a.cohort is None:
        p.error('--cohort is required')
    if not (a.bundle / 'SHA256.json').exists():
        raise SystemExit(f'not a frozen bundle: {a.bundle}')
    {'start': start, 'plan-verify': plan_verify, 'after-verify': after_verify,
     'plan-cohort': plan_cohort, 'gate': gate}[a.step](a)


if __name__ == '__main__':
    main()
