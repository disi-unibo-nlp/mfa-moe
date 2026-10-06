"""Finite TP=2 generation with warmed qualification and durable per-cell receipts."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import time

import native_finalization_correction_v3 as C
C.activate()
import native_finalization_v1 as N


def route_agreement(captures, routed):
    import numpy as np
    N.require(len(captures) == 2 and {x['rank'] for x in captures} == {0, 1}, 'missing TP rank')
    arrays = []
    for r in sorted(captures, key=lambda r: r['rank']):
        N.require(N.U.file_sha(Path(r['path'])) == r['file_sha256'], 'capture bytes changed')
        with np.load(r['path']) as data:
            arrays.append((data['native_ids'].copy(), data['executed_weights'].copy()))
    exact = all(np.array_equal(x[0], routed) for x in arrays)
    weights = np.allclose(arrays[0][1], arrays[1][1], rtol=1e-5, atol=1e-6)
    N.require(exact and weights, 'native routing capture/rank agreement differs')
    return {'ids_match_returned_routes': exact, 'rank_ids_agree': True, 'rank_weights_agree': bool(weights),
            'statistical_observations': 1, 'ranks_are_not_independent_observations': True}


def run(manifest_path, root, deadline):
    N.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
              not socket.gethostname().startswith('login'), 'GPU Slurm step required')
    N.require(str(Path(os.environ.get('TMPDIR', '/tmp')).resolve()).startswith(
        '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'), 'GPU temporary directory is outside user staging')
    manifest = N.U.sealed(manifest_path); N.validate(manifest)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = N.reconcile(manifest, root)
    N.require(not index['uncommitted_attempts'], 'uncommitted attempts need explicit reconciliation; no duplicate execution')
    done = {r['assignment']['uid'] for r in index['records'] if r['status'] != 'MISSING'}
    pending = [r for r in manifest['rows'] if r['assignment']['uid'] not in done]
    if not pending:
        return
    from utility_outcomes_v3 import scoring_modules
    _, engine_module = scoring_modules()
    from vllm import LLM, SamplingParams
    import numpy as np
    from operator_panel_outcomes_v1 import boundary_contract
    import importlib.metadata
    job = os.environ['SLURM_JOB_ID']; allocation = root / 'allocations' / job
    allocation.mkdir(parents=True, exist_ok=False)
    kwargs = engine_module.engine_kwargs(plugin=False, max_num_seqs=4, enforce_eager=True, return_routed_experts=True)
    kwargs['worker_extension_cls'] = 'native_finalization_worker_v2.NativeFinalizationWorker'
    for key in list(os.environ):
        if key.startswith('STEER_'):
            del os.environ[key]
    started = time.monotonic(); model = None
    try:
        model = LLM(**kwargs)
        tokenizer = model.get_tokenizer()
        boundary_contract(tokenizer, manifest['binding']['tokenizer_sha256'])
        N.save(allocation / 'LOAD.json', {'schema': 'native-finalization-load-v1', 'job_id': job,
            'seconds': time.monotonic() - started, 'kwargs': kwargs,
            'versions': {n: importlib.metadata.version(n) for n in ('vllm', 'torch')},
            'manifest_sha256': manifest['sha256']})
        # Same warmed engine: native reference then instrumented repeat, external fixture.
        fixture_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Compute 7 + 5. Give a concise answer.'}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if hasattr(fixture_ids, 'get'):
            fixture_ids = fixture_ids['input_ids']
        fixture_ids = list(fixture_ids)
        fixture = {'uid': 'engineering-native-finalization-' + job, 'prompt_tokens': len(fixture_ids)}
        params = lambda seed, cap, uid, prompt: SamplingParams(**engine_module.card_sampling(cap, seed,
            extra_args={'native_finalization': {'uid': uid}}, routed_experts_prompt_start=len(prompt) - 1))
        from native_finalization_replay_v3 import snapshot, save_evidence
        reference_completion = model.generate([{'prompt_token_ids': fixture_ids}],
            params(0, 64, fixture['uid'], fixture_ids), use_tqdm=False)[0].outputs[0]
        reference = snapshot(reference_completion)
        repeated = snapshot(model.generate([{'prompt_token_ids': fixture_ids}],
            params(0, 64, fixture['uid'], fixture_ids), use_tqdm=False)[0].outputs[0])
        status = model.collective_rpc('finalization_install')
        model.collective_rpc('finalization_begin', args=([fixture], 64))
        captured = model.generate([{'prompt_token_ids': fixture_ids}],
            params(0, 64, fixture['uid'], fixture_ids), use_tqdm=False)[0].outputs[0]
        captures = model.collective_rpc('finalization_flush', args=(fixture['uid'], len(captured.token_ids), str(allocation / 'engineering')))
        replay = save_evidence(allocation, fixture_ids, {'native_reference': reference,
            'native_repeat': repeated, 'instrumented': snapshot(captured)}, captures)
        print('ENGINEERING_REPLAY', replay['native_repeat'], replay['instrumented_repeat'], flush=True)
        N.require(replay['native_repeat']['tokens_equal'] and replay['native_repeat']['routes_equal'],
            'native-to-native replay is unstable; attribution to capture is unresolved')
        N.require(replay['instrumented_repeat']['tokens_equal'] and replay['instrumented_repeat']['routes_equal'],
            'instrumented replay differs; see sealed ENGINEERING_REPLAY evidence')
        agreement = route_agreement(captures, np.asarray(captured.routed_experts))
        batch_fixtures = []
        for i in range(4):
            prompt = fixture_ids + tokenizer.encode(' ' * (i + 1), add_special_tokens=False)
            batch_fixtures.append({'uid': fixture['uid'] + '-batch-' + str(i),
                                   'prompt_tokens': len(prompt), 'prompt_token_ids': prompt})
        model.collective_rpc('finalization_begin', args=(batch_fixtures, 8))
        answers = model.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in batch_fixtures],
            [params(i % 2, i + 3, r['uid'], r['prompt_token_ids']) for i, r in enumerate(batch_fixtures)], use_tqdm=False)
        batch_checks = []
        for fixture_row, answer in zip(batch_fixtures, answers, strict=True):
            completion = answer.outputs[0]
            proof = model.collective_rpc('finalization_flush', args=(fixture_row['uid'],
                len(completion.token_ids), str(allocation / 'engineering')))
            batch_checks.append({'uid': fixture_row['uid'], 'tokens': len(completion.token_ids),
                'captures': proof, 'agreement': route_agreement(proof, np.asarray(completion.routed_experts))})
        qualification = N.save(allocation / 'QUALIFICATION.json', {'schema': 'native-finalization-qualification-v1',
            'status': 'PASS_ENGINEERING', 'excluded_from_outcomes': True,
            'manifest_sha256': manifest['sha256'], 'job_id': job, 'worker_status': status,
            'fixture_prompt_ids': fixture_ids, 'fixture_output_ids': list(captured.token_ids),
            'reference_output_ids': reference['tokens'], 'engineering_replay_sha256': replay['sha256'], 'captures': captures, 'agreement': agreement,
            'staggered_batch_qualification': batch_checks})
        # Submit contiguous four-cell blocks; finish each block before the next.
        for start in range(0, len(pending), 4):
            if time.time() >= deadline - 60:
                break
            block = pending[start:start + 4]
            assigned = [r['assignment'] for r in block]
            for a in assigned:
                N.save(root / 'attempts' / (N.U.digest(a['uid']) + '.json'), {
                    'schema': 'native-finalization-attempt-v1', 'manifest_sha256': manifest['sha256'],
                    'assignment': a, 'job_id': job, 'qualification_sha256': qualification['sha256']})
            model.collective_rpc('finalization_begin', args=(assigned, N.CAP))
            active = {}; began = time.monotonic()
            for r in block:
                a = r['assignment']; ids = r['prompt_token_ids']
                model.llm_engine.add_request(a['uid'], {'prompt_token_ids': ids}, params(a['seed'], N.CAP, a['uid'], ids))
                active[a['uid']] = a
            last = {}
            try:
                while model.llm_engine.has_unfinished_requests():
                    if time.time() >= deadline:
                        raise TimeoutError('allocation shutdown margin reached')
                    for output in model.llm_engine.step():
                        a = active[output.request_id]; c = output.outputs[0]
                        last[a['uid']] = list(c.token_ids)
                        if not output.finished:
                            continue
                        ids = list(c.token_ids)
                        N.require(c.finish_reason in ('stop', 'length') and
                            (c.finish_reason != 'length' or len(ids) == N.CAP), 'finish/cap contract differs')
                        routed = np.asarray(c.routed_experts)
                        N.require(routed.shape == (len(ids), 40, 8), 'prediction route coverage differs')
                        captures = model.collective_rpc('finalization_flush', args=(a['uid'], len(ids), str(root / 'routes')))
                        agreement = route_agreement(captures, routed)
                        attempt = N.U.sealed(root / 'attempts' / (N.U.digest(a['uid']) + '.json'))
                        N.save(N.receipt_path(root, a), {'schema': 'native-finalization-receipt-v1',
                            'status': 'COMMITTED_GENERATION', 'assignment': a, 'manifest_sha256': manifest['sha256'],
                            'attempt_sha256': attempt['sha256'], 'job_id': job, 'qualification_only': False,
                            'token_ids': ids, 'token_ids_sha256': N.U.digest(ids), 'finish': c.finish_reason,
                            'error': None, 'captures': captures, 'rank_agreement': agreement,
                            'batch_elapsed_seconds': time.monotonic() - began,
                            'sampler': engine_module.card_sampling(N.CAP, a['seed']),
                            'routing': 'native', 'maximum_tokens': N.CAP})
                        print('COMMITTED', a['uid'], len(ids), c.finish_reason, flush=True)
                        del active[a['uid']]
            except Exception as error:
                for a in active.values():
                    path = N.receipt_path(root, a)
                    if path.exists():
                        continue
                    attempt = N.U.sealed(root / 'attempts' / (N.U.digest(a['uid']) + '.json'))
                    N.save(path, {'schema': 'native-finalization-receipt-v1', 'status': 'GENERATION_ERROR',
                        'assignment': a, 'manifest_sha256': manifest['sha256'], 'attempt_sha256': attempt['sha256'],
                        'job_id': job, 'token_ids': last.get(a['uid'], []), 'finish': 'error', 'error': repr(error)})
                raise
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()
        N.save(allocation / 'COST.json', {'schema': 'native-finalization-driver-cost-v1', 'job_id': job,
            'driver_seconds': time.monotonic() - started, 'scope': 'sacct allocation cost remains authoritative'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=N.DOC / 'MANIFEST.json')
    parser.add_argument('--out', type=Path, default=N.ROOT)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out, args.deadline_epoch)
