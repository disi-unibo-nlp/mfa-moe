"""Separate measurement-only native NLL driver; execute on allocated LEONARDO GPUs only.

Qualification replays identical Q3 prompt+completion[:2048] fixtures before any X2 NLL
is measured. Generation, policies, hooks, and the frozen s2 tree are unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

# Load the measurement helper without shadowing the frozen moe_exp source package.
helper = Path(__file__).with_name('nll_helpers.py')
if not helper.is_file():
    helper = Path(__file__).resolve().parents[2] / 'src/moe_exp/routing_control/nll.py'
helper_spec = importlib.util.spec_from_file_location('native_nll_helpers', helper)
helper_module = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper_module)
actual_logprobs = helper_module.actual_logprobs
pulse_surprisal = helper_module.pulse_surprisal
validate_parity = helper_module.validate_parity
receipts_path = Path(__file__).with_name('receipts.py')
if not receipts_path.is_file():
    receipts_path = Path(__file__).resolve().parents[2] / 'src/moe_exp/routing_control/receipts.py'
receipt_spec = importlib.util.spec_from_file_location('native_nll_receipts', receipts_path)
receipt_module = importlib.util.module_from_spec(receipt_spec)
receipt_spec.loader.exec_module(receipt_module)
ReceiptStore, atomic_json = receipt_module.ReceiptStore, receipt_module.atomic_json


def require_compute():
    import socket
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native NLL inference requires an allocated compute-node Slurm step')


def measure(model, prompts, batch_size):
    from vllm import SamplingParams
    params = SamplingParams(temperature=0., max_tokens=1, presence_penalty=0.,
                            prompt_logprobs=0, detokenize=False)
    for start in range(0, len(prompts), batch_size):
        block = prompts[start:start + batch_size]
        outputs = model.generate([{'prompt_token_ids': p} for p in block], params, use_tqdm=False)
        if len(outputs) != len(block):
            raise ValueError('measurement engine returned an incomplete batch')
        for prompt, output in zip(block, outputs):
            if list(output.prompt_token_ids) != prompt:
                raise ValueError('engine changed the teacher-forced token sequence')
            yield output.prompt_logprobs


def run(args):
    require_compute()
    from moe_steer import engine, manifests as M, qualify as Q, results as RS
    started = time.monotonic()
    source_tree = engine.code_tree_sha256()
    if source_tree != args.expect_tree:
        raise ValueError('native NLL imported an unexpected frozen runtime tree')
    world = M.load_world()
    fixtures = Q.Fixtures(world).parity_traces()
    parity_prompts = [list(f.prompt_ids) + list(f.completion_ids[:Q.PARITY_TF_TOKENS]) for f in fixtures]
    manifest = M.load_manifest(args.manifest)
    records = [record for shard in range(manifest['n_shards'])
               for record in RS.read_shard_records(args.results, shard)]
    by_uid = {r['uid']: r for r in records}
    if len(by_uid) != len(records) or set(by_uid) != {r['uid'] for r in manifest['requests']}:
        raise ValueError('NLL requires complete, unique generation records including all assigned requests')
    for request in manifest['requests']:
        record = by_uid[request['uid']]
        RS.validate_result(record, request)
        if record['manifest_sha256'] != manifest['sha256'] or record['code_tree'] != source_tree:
            raise ValueError('generation result binding mismatch')
    selected = [r for r in manifest['requests'] if r['arm'] in ('N', 'E+', 'E-')]
    prefixes = None
    if args.prefixes is not None:
        prefixes = json.loads(args.prefixes.read_text())
        if prefixes['manifest_sha256'] != manifest['sha256'] or prefixes['code_tree'] != source_tree:
            raise ValueError('prepared prefix binding differs from generation')
        if set(prefixes['questions']) != {r['question'] for r in selected}:
            raise ValueError('prepared prefixes omit an assigned question')
    identity = {'uid_format': 'legacy-sha256-prefix32',
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'nll_helpers_sha256': hashlib.sha256(Path(inspect.getfile(pulse_surprisal)).read_bytes()).hexdigest(),
                'receipts_sha256': hashlib.sha256(receipts_path.read_bytes()).hexdigest(),
                'code_tree': source_tree, 'manifest_sha256': manifest['sha256'],
                'generation_tokens_sha256': hashlib.sha256(json.dumps([
                    [r['uid'], by_uid[r['uid']]['completion_token_ids'],
                     by_uid[r['uid']]['finish_reason']] for r in manifest['requests']],
                    separators=(',', ':')).encode()).hexdigest(),
                'parity_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for workload in ('tf1', 'tf8') for p in [
                        args.parity / f'plugin.{workload}.json',
                        *[args.parity / f'{name}.{workload}.npz'
                          for name in ('plugin', 'free_A', 'free_B')]]},
                'parity_prompts_sha256': hashlib.sha256(json.dumps(parity_prompts,
                    separators=(',', ':')).encode()).hexdigest(),
                'prepared_prefixes_sha256': hashlib.sha256(args.prefixes.read_bytes()).hexdigest()
                    if args.prefixes is not None else None}
    with ReceiptStore(args.out, identity, [r['uid'] for r in selected]) as store:
        done = store.read()
        final = args.out / 'native-nll.json'
        if final.exists():
            value = json.loads(final.read_text())
            if value['binding'] != identity or len(done) != len(selected):
                raise ValueError('incompatible or incomplete final NLL result')
            return
        attempt = store.attempt()
        atomic_json(attempt / 'START.json', {'slurm_job_id': os.environ['SLURM_JOB_ID'],
            'assigned': len(selected), 'already_measured': len(done), 'status': 'STARTED'})
        try:
            validation = run_measurement(args, engine, M, Q, RS, identity, fixtures, parity_prompts,
                                         by_uid, selected, store, attempt, prefixes)
        except BaseException as error:
            atomic_json(attempt / 'END.json', {'status': 'FAILED', 'error_type': type(error).__name__,
                'seconds_including_load_and_parity': time.monotonic() - started})
            raise
        values = store.read()
        if set(values) != {r['uid'] for r in selected}:
            raise ValueError('incomplete native NLL measurement')
        atomic_json(attempt / 'END.json', {'status': 'COMPLETE',
            'seconds_including_load_and_parity': time.monotonic() - started})
        atomic_json(final, {'binding': identity, 'records': [values[r['uid']] for r in selected],
            'validation': validation, 'attempts': str(args.out / 'attempts'),
            'compute_accounting': 'Use sacct for every attempt, including killed jobs without END receipts.'})


def run_measurement(args, engine, M, Q, RS, identity, fixtures, parity_prompts,
                    by_uid, selected, store, attempt, prefixes=None):
    kwargs = engine.engine_kwargs(plugin=False, max_num_seqs=48, return_routed_experts=False)
    kwargs.update(gpu_memory_utilization=.80, long_prefill_token_threshold=1024)
    native_env = engine.engine_env(None, None, plugin=False)
    for key in tuple(os.environ):
        if key.startswith('STEER_') and key not in native_env:
            os.environ.pop(key)
    engine.apply_env(native_env)
    from vllm import LLM
    model = LLM(**kwargs)
    validation = {}
    for workload, batch_size in [('tf1', 1), ('tf8', 8)]:
        metadata = json.loads((args.parity / f'plugin.{workload}.json').read_text())
        if [f['question'] for f in metadata['fixtures']] != [f.question for f in fixtures]:
            raise ValueError('parity fixture order/identity differs')
        if [f['prompt_len'] for f in metadata['fixtures']] != list(map(len, parity_prompts)):
            raise ValueError('parity fixture token positions differ')
        arrays = [actual_logprobs(p, lp) for p, lp in zip(parity_prompts, measure(model, parity_prompts, batch_size))]
        archives = [np.load(args.parity / f'{name}.{workload}.npz', allow_pickle=False)
                    for name in ('plugin', 'free_A', 'free_B')]
        try:
            reference, a, b = [[archive[f'l{i}'] for i in range(len(fixtures))] for archive in archives]
            validation[workload] = validate_parity(arrays, reference, a, b)
        finally:
            for archive in archives:
                archive.close()
        np.savez_compressed(attempt / f'native.{workload}.npz', **{f'l{i}': a for i, a in enumerate(arrays)})
    atomic_json(attempt / 'PARITY_VALIDATION.json', validation)
    if not all(row['pass'] for row in validation.values()):
        raise RuntimeError('NLL pass void: native parity validation failed; no X2 NLL read')
    manifest = M.load_manifest(args.manifest)
    trace_store = M.TraceStore(cache_dir=args.out / 'trace-offsets')
    pending = [r for r in selected if r['uid'] not in store.read()]
    for start in range(0, len(pending), 48):
        block = pending[start:start + 48]
        prepared, prompts = [], []
        for request in block:
            record = by_uid[request['uid']]
            RS.validate_result(record, request)
            if record['manifest_sha256'] != manifest['sha256'] or record['code_tree'] != identity['code_tree']:
                raise ValueError('generation result binding mismatch')
            base = manifest['questions'][request['question']]['prompt_token_ids']
            parent = request['parent']
            if prefixes is not None:
                prepared_prefix = prefixes['questions'][request['question']]
                if prepared_prefix['parent'] != parent or prepared_prefix['original_prompt_token_ids'] != base:
                    raise ValueError('prepared prefix inputs differ from the request')
                prefix = prepared_prefix['prefix_token_ids']
                if prefix[:len(base)] != base or len(prefix) != len(base) + (parent['prefix_len'] if parent else 0):
                    raise ValueError('prepared prefix length or original prompt differs')
            elif parent is None:
                prefix = list(base)
            else:
                ref = parent.get('trace_ref')
                parent_ids = (trace_store.completion_ids(ref['dataset'], ref['problem_id']) if ref
                              else by_uid[parent['parent_uid']]['completion_token_ids'])
                prefix = M.branch_prompt(base, parent_ids, parent['prefix_len'])
            pulse = record['completion_token_ids'][:256]
            if not pulse:
                prepared.append((request, None, 0, 0))
                continue
            prompt = list(prefix) + pulse
            prompts.append(prompt)
            prepared.append((request, prompt, len(prefix), len(pulse)))
        logprobs = iter(measure(model, prompts, 48))
        for request, prompt, begin, n in prepared:
            row = {'uid': request['uid'], 'question': request['question'],
                   'seed_k': request['seed_k'], 'policy_name': request['policy_name']}
            row.update({'status': 'NO_PULSE_TOKENS', 'n_tokens': 0, 'mean_nll': None}
                       if prompt is None else
                       {'status': 'MEASURED', **pulse_surprisal(prompt, next(logprobs),
                           continuation_start=begin, n_pulse_tokens=n)})
            store.put(row)
    return validation


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'results', 'parity', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--prefixes', type=Path)
    p.add_argument('--expect-tree', required=True)
    run(p.parse_args())
