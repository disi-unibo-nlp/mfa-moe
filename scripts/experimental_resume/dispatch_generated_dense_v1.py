"""One-shot authorized dispatch of an exactly priced dense label/analysis stage."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import dispatch_overnight_readers_v1 as shared
import generated_dense_pipeline_v1 as dense

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
PLAN = DOC / 'GENERATED_DENSE_MEASUREMENT_PLAN_v1.json'
WRAPPERS = [REPO / 'scripts/experimental_resume/label_generated_dense_v1.sbatch',
            REPO / 'scripts/experimental_resume/analyze_generated_dense_v1.sbatch']


def validate_complete_price(frame, price):
    dense.require(price['status'] == 'PASS_COMPLETE_LABEL_STAGE' and
                  price['sentences'] == len(frame['records']) == len(price['prompt_records']) and
                  price['batch_size'] == dense.BATCH and price['max_tokens'] == dense.MAX_TOKENS and
                  price['gpus_per_job'] == 2 and price['max_wall_seconds'] == 7200 and
                  price['repeat_factor'] == 1.25 and price['per_shard_overhead_seconds'] == 900 and
                  price['maximum_decode_tokens'] == price['sentences'] * dense.MAX_TOKENS,
                  'dense complete price/profile differs from frozen launch envelope')
    dense.require(all(isinstance(price[k], (int, float)) and math.isfinite(price[k]) and price[k] > 0
                      for k in ('bounded_prefill_tokens_per_second', 'bounded_decode_tokens_per_second',
                                'cold_load_seconds', 'shutdown_seconds')), 'invalid dense timing source')
    prompts = price['prompt_records']
    dense.require(all(type(r['prompt_tokens']) is int and 16 <= r['prompt_tokens'] <=
                      dense.PROFILE['max_model_len'] - dense.MAX_TOKENS for r in prompts) and
                  sum(r['prompt_tokens'] for r in prompts) == price['prompt_tokens_exact'],
                  'dense price lacks valid exact prompt counts')
    overhead = price['cold_load_seconds'] + price['shutdown_seconds'] + price['per_shard_overhead_seconds']
    last = 0
    for shard in price['shards']:
        start, stop = shard['start'], shard['stop']
        dense.require(start == last and type(start) is int and type(stop) is int and
                      start % dense.BATCH == 0 and start < stop <= len(prompts) and
                      (stop == len(prompts) or stop % dense.BATCH == 0), 'dense shards do not cover exact whole batches')
        block = prompts[start:stop]
        expected = price['repeat_factor'] * (
            sum(r['prompt_tokens'] for r in block) / price['bounded_prefill_tokens_per_second'] +
            len(block) * dense.MAX_TOKENS / price['bounded_decode_tokens_per_second'])
        dense.require(math.isclose(shard['work_seconds'], expected, rel_tol=1e-12, abs_tol=1e-8) and
                      math.isclose(shard['complete_wall_seconds'], expected + overhead, abs_tol=1e-8) and
                      shard['complete_wall_seconds'] <= price['max_wall_seconds'],
                      'dense full shard price exceeds wall or was changed')
        last = stop
    dense.require(last == len(prompts), 'dense full price omits sentence prompts')
    count = len(price['shards'])
    loads = count + (max(1, math.ceil(count * .25)) if count else 0)
    seconds = sum(s['work_seconds'] for s in price['shards']) + loads * (
        price['cold_load_seconds'] + price['shutdown_seconds']) + count * price['per_shard_overhead_seconds']
    dense.require(price['cold_loads_including_reserve'] == loads and
                  math.isclose(price['complete_stage_GPU_h'], 2 * seconds / 3600, rel_tol=1e-12, abs_tol=1e-10),
                  'dense full-stage price omits loads, reserve or overhead')


def prepared(manifest_path, generation_price_path, run_out, parent):
    manifest, generation_price = dense.sealed(Path(manifest_path)), dense.sealed(Path(generation_price_path))
    directory = dense.output_path(manifest, parent)
    frame, price = dense.inputs(directory)
    plan = dense.sealed(PLAN)
    arm_map = dense.sealed(directory / 'ARM_MAP.json')
    dense.require(plan['code_files'] == dense.measurement_code() and
                  plan['code_digest'] == frame['code_digest'] and
                  frame['generation_manifest_sha256'] == manifest['sha256'] and
                  generation_price['manifest_sha256'] == manifest['sha256'] and
                  generation_price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY' and
                  generation_price['shards'] == manifest['shards'] and
                  frame['generation_run_out'] == str(Path(run_out).resolve()) and
                  dense.sealed(Path(run_out) / 'STAGE_COMPLETION.json')['sha256'] == frame['generation_stage_sha256'] and
                  arm_map['generation_manifest_sha256'] == manifest['sha256'] and
                  arm_map['generation_stage_sha256'] == frame['generation_stage_sha256'] and
                  arm_map['assigned'] == frame['assigned'] == manifest['expected_requests'],
                  'dense source generation, measurement plan or exact prepared stage differs')
    dense.require(all(dense.file_sha(path) == sha for path, sha in manifest['code_files'].items()),
                  'frozen generation source changed before dense dispatch')
    validate_complete_price(frame, price)
    binding = {'schema': 'generated-dense-dispatch-binding-v1',
               'manifest_path': str(Path(manifest_path).resolve()), 'generation_manifest_sha256': manifest['sha256'],
               'generation_price_sha256': generation_price['sha256'],
               'dense_directory': str(directory), 'frame_sha256': frame['sha256'], 'arm_map_sha256': arm_map['sha256'],
               'price_sha256': price['sha256'], 'measurement_plan_sha256': plan['sha256'],
               'complete_stage_GPU_h': price['complete_stage_GPU_h'],
               'dispatch_driver_sha256': dense.file_sha(__file__), 'shared_dispatch_sha256': dense.file_sha(shared.__file__),
               'wrapper_hashes': {str(p): dense.file_sha(p) for p in WRAPPERS}}
    return directory, frame, price, binding


def live_account_snapshot(directory, name):
    records = {}
    for record_name, args in (
        ('gpu_partition', ['scontrol', 'show', 'partition', 'boost_usr_prod']),
        ('cpu_partition', ['scontrol', 'show', 'partition', 'lrd_all_viz']),
        ('association', ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user=lmolfett',
                         'account=iscrc_miosr', 'format=Account,Partition,QOS'])):
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        records[record_name] = result.stdout
    dense.require('iscrc_miosr' in records['association'], 'authorized project association is absent')
    shared.save(directory / name, {'schema': 'dense-live-account-snapshot-v1',
                'host': socket.gethostname(), 'user': getpass.getuser(), 'records': records})


def ensure_verified(directory, name, job_id):
    """Also reconcile the narrow receipt-written/readback-interrupted resume case."""
    path = directory / f'{name}.verified.json'
    if path.exists():
        prior = dense.sealed(path)
        dense.require(prior['job_id'] == job_id and f'JobId={job_id}' in prior['scontrol'] and
                      'UserId=lmolfett(' in prior['scontrol'], 'stored job verification differs')
        return
    result = subprocess.run(['scontrol', 'show', 'job', job_id], check=True, capture_output=True, text=True)
    dense.require(f'JobId={job_id}' in result.stdout and 'UserId=lmolfett(' in result.stdout,
                  'submitted dense job did not verify by exact ID/owner')
    shared.save(path, {'schema': 'overnight-job-verification-v1', 'job_id': job_id, 'scontrol': result.stdout})


def dispatch(directory, frame, price, binding, *, submit=shared.submit):
    receipts = directory / 'dispatch-v1'
    receipts.mkdir(exist_ok=True)
    with (receipts / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = 'LIVE_ACCOUNT-' + os.environ.get('SLURM_JOB_ID', str(os.getpid())) + '.json'
        if not (receipts / snapshot).exists():
            live_account_snapshot(receipts, snapshot)
        for wrapper in WRAPPERS:
            subprocess.run(['bash', '-n', str(wrapper)], check=True)
        env = os.environ.copy()
        label_job = None
        if price['shards']:
            label_job = submit(receipts, 'label-array', [f'--array=0-{len(price["shards"])-1}',
                               '--time=02:00:00', '--job-name=dense-label-' + frame['generation_manifest_sha256'][:8],
                               str(WRAPPERS[0]), str(directory)], env, binding)
            ensure_verified(receipts, 'label-array', label_job)
        args = ([f'--dependency=afterok:{label_job}'] if label_job else [])
        analysis_job = submit(receipts, 'trajectory-analysis', [*args,
                              '--job-name=dense-trajectory-' + frame['generation_manifest_sha256'][:8],
                              str(WRAPPERS[1]), str(directory)], env, binding)
        ensure_verified(receipts, 'trajectory-analysis', analysis_job)
        chain = shared.save(receipts / 'CHAIN.json', {'schema': 'generated-dense-measurement-chain-v1',
                            'binding': binding, 'label_array_job': label_job, 'analysis_job': analysis_job,
                            'assigned_sentences': frame['sentences'], 'label_shards': len(price['shards']),
                            'complete_stage_GPU_h': price['complete_stage_GPU_h'],
                            'result_path': str(directory / 'TRAJECTORY_RESULT.json')})
        return chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'generation_price', 'generation_out', 'dense_parent'):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    dense.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
                  getpass.getuser() == 'lmolfett' and not socket.gethostname().startswith('login'),
                  'dense dispatcher requires its authorized one-shot viz CPU job')
    directory, frame, price, binding = prepared(args.manifest, args.generation_price,
                                               args.generation_out, args.dense_parent)
    value = dispatch(directory, frame, price, binding)
    print(json.dumps({'chain': str(directory / 'dispatch-v1/CHAIN.json'), 'sha256': value['sha256'],
                      'label_array_job': value['label_array_job'], 'analysis_job': value['analysis_job']}), flush=True)


if __name__ == '__main__':
    main()
