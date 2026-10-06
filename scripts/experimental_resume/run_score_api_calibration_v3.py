"""Allocated-GPU comparison of score APIs at identical native prediction prefixes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


def local_module(name, filename, fallback):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / 'src/moe_exp/routing_control' / fallback
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = local_module('cf_helpers', 'counterfactual_helpers.py', 'counterfactual.py')
C = local_module('score_calibration', 'score_calibration.py', 'score_calibration.py')
R = local_module('cf_receipts', 'receipts.py', 'receipts.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(manifest):
    rows = []
    for row in manifest['rows']:
        for position in manifest['positions']:
            prompt, target = H.prediction_input(row, position)
            for mode in C.MODES:
                meta = {'prefix_uid': row['uid'], 'family': row['family'], 'position': position,
                    'mode': mode, 'prompt_sha256': H.digest(prompt), 'prompt_len': len(prompt),
                    'prefix_len': len(prompt) - len(row['prompt_ids']), 'target_token': target}
                meta['uid'] = H.digest(['score-api-calibration-v3', manifest['sha256'], meta])
                rows.append((prompt, meta))
    for row in manifest['rows']:
        for position in manifest['positions']:
            prompt, target = H.prediction_input(row, position)
            prompt = prompt + [target]
            meta = {'prefix_uid': row['uid'], 'family': row['family'], 'position': position,
                'mode': 'teacher_force_diagnostic', 'prompt_sha256': H.digest(prompt),
                'prompt_len': len(prompt), 'prefix_len': len(prompt) - len(row['prompt_ids']),
                'target_token': target}
            meta['uid'] = H.digest(['score-api-calibration-v3', manifest['sha256'], meta])
            rows.append((prompt, meta))
    if len(rows) != manifest['requests']:
        raise ValueError('calibration assignment count differs')
    return rows


def result_for(manifest, completed):
    lookup = {(r['prefix_uid'], r['position'], r['mode']): r for r in completed.values()}
    pairs, diagnostics = [], []
    for row in manifest['rows']:
        for position in manifest['positions']:
            sample = {mode: lookup[row['uid'], position, mode] for mode in C.MODES}
            if len({r['prompt_sha256'] for r in sample.values()}) != 1:
                raise ValueError('score methods do not have identical native prefixes')
            pairs.append({mode: r['raw_target_logprob'] for mode, r in sample.items()})
            diagnostics.append({'prefix_uid': row['uid'], 'position': position,
                'native_token_ids': {mode: r['sampled_token'] for mode, r in sample.items()},
                'raw_target_logprobs': pairs[-1],
                'teacher_force_abs_difference': abs(pairs[-1]['designated_a'] -
                    lookup[row['uid'], position, 'teacher_force_diagnostic']['raw_target_logprob']),
                'unordered_route_equal_cross_api_a': sample['designated_a']['unordered_route'] == sample['full_vocab_a']['unordered_route'],
                'unordered_route_equal_designated_repeat': sample['designated_a']['unordered_route'] == sample['designated_b']['unordered_route'],
                'unordered_route_equal_full_vocab_repeat': sample['full_vocab_a']['unordered_route'] == sample['full_vocab_b']['unordered_route']})
    numerical = C.calibration(pairs, manifest['calibration_ratio'], manifest['calibration_epsilon'])
    return {'schema': 'routing-score-api-calibration-result-v3', 'manifest_sha256': manifest['sha256'],
        'assigned': manifest['requests'], 'completed': len(completed), 'numerical_calibration': numerical,
        'pass': len(completed) == manifest['requests'] and numerical['pass'],
        'teacher_forcing_role': 'diagnostic only: appended input and execution layout differ; prior tight-alignment failure remains recorded',
        'case_diagnostics': diagnostics,
        'interpretation': 'engineering score extraction only; four pilot prefixes were not semantically eligible; no expert ranking, engine equivalence or semantic steering claim'}


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('score calibration requires an allocated GPU Slurm step')
    from moe_steer import engine, policies as P, qualify as Q
    manifest = H.sealed(args.manifest)
    if manifest['driver_sha256'] != sha(__file__) or manifest['base_tree_sha256'] != engine.code_tree_sha256():
        raise ValueError('score-calibration code binding changed')
    if any(sha(path) != expected for path, expected in manifest['code_files'].items()):
        raise ValueError('calibration helper or qualified worker changed')
    worker = H.sealed(manifest['worker_binding'])
    overlay = Path(manifest['worker_overlay'])
    if worker['overlay'] != str(overlay):
        raise ValueError('worker overlay differs')
    assigned = cases(manifest)
    binding = {'manifest_sha256': manifest['sha256'], 'driver_sha256': sha(__file__),
        'base_tree_sha256': manifest['base_tree_sha256'], 'qualification': 'identical-prefix raw-score API calibration'}
    with R.ReceiptStore(args.out, binding, [m['uid'] for _, m in assigned]) as store:
        done = store.read()
        if len(done) == len(assigned):
            result = result_for(manifest, done)
            R.atomic_json(args.out / 'QUALIFICATION.json', {**result, 'sha256': H.digest(result)})
            if not result['pass']:
                raise RuntimeError('preserved numerical score calibration failed')
            return
        attempt = store.attempt()
        started = time.time()
        R.atomic_json(attempt / 'START.json', {'job_id': os.environ['SLURM_JOB_ID'], 'assigned': len(assigned)})
        # Retain the exact populated pilot table; every request selects zero.
        table = P.PolicyTable.from_sealed(H.sealed(manifest['policy_table']))
        if table.digest() != manifest['policy_table_sha256'] or table.hooked_layers() != (28,):
            raise ValueError('exact pilot hook table changed')
        deadline = Q.Deadline.from_env()
        deadline.require(1100, 'cold load and first calibration batch')
        fingerprint = engine.fingerprint()
        telemetry = attempt / 'telemetry'
        telemetry.mkdir()
        env = engine.engine_env(Path(manifest['policy_table']), telemetry, expect_fingerprint=fingerprint['combined'])
        env['PYTHONPATH'] = os.pathsep.join((str(overlay), env['PYTHONPATH']))
        engine.apply_env(env)
        sys.path[:] = [str(overlay), *(p for p in sys.path if p != str(overlay))]
        import moe_exp
        spec = importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
        if Path(moe_exp.__file__).resolve().parent != overlay / 'moe_exp' or Path(spec.origin).resolve() != overlay / 'moe_exp/routing_control/ordered_vllm.py':
            raise ValueError('worker import differs from qualified overlay')
        kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=16, return_routed_experts=True)
        kwargs.update(worker_extension_cls='moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension',
            gpu_memory_utilization=.80, long_prefill_token_threshold=1024,
            logprobs_mode='raw_logprobs', max_logprobs=-1)
        model = engine.build_llm(kwargs)
        loaded = time.time()
        driver = Q.QDriver(model, plugin=True)
        harness = Q.Harness(driver, telemetry, deadline=deadline, abort_margin=90.)
        groups = [assigned[i:i + 16] for i in range(0, 64, 16)] + [assigned[64:]]
        for index, group in enumerate(groups):
            pending = [(prompt, meta) for prompt, meta in group if meta['uid'] not in done]
            if not pending:
                continue
            deadline.require(240, 'raw score calibration batch ' + str(index))
            reqs = []
            for prompt, meta in pending:
                extra = Q.steer_extra(table, meta['uid'], 'zero',
                    len(prompt) - meta['prefix_len'], prefix_len=meta['prefix_len'], restore_presence=False)
                sampling = Q.greedy_params(1, extra, presence=0., routed_start=len(prompt) - 1,
                    logprobs=-1 if meta['mode'].startswith('full_vocab') else None,
                    prompt_logprobs=0 if meta['mode'] == 'teacher_force_diagnostic' else None)
                if not meta['mode'].startswith('full_vocab'):
                    sampling['logprob_token_ids'] = [meta['target_token']]
                reqs.append(Q.QReq(meta['uid'], prompt, sampling))
            before = time.time()
            batch = harness.run(reqs, cap_in_flight=16)
            harness.flush()
            records, _, _ = harness.telemetry()
            values = []
            for _, meta in pending:
                out = batch.outcomes.get(meta['uid'])
                if out is None or out.error or len(out.tokens) != 1 or out.routed is None:
                    raise ValueError('missing native single-token assignment')
                route = np.asarray(out.routed)
                if route.shape != (1, 40, 8):
                    raise ValueError('prediction route shape differs')
                H.verify_route(route[0].tolist(), (28, [9, 189]), 'none')
                lp = H.designated_logprob(out.prompt_logprobs[-1] if meta['mode'] == 'teacher_force_diagnostic'
                    else out.logprobs[0], meta['target_token'])
                ranks = {}
                for rank in (0, 1):
                    record = records.get(rank, {}).get(meta['uid'])
                    if record is None or record['cpu_active_rows']:
                        raise ValueError('native-only calibration must have both inactive worker-rank receipts')
                    if any(c['expert_identity_mismatches'] or c['weight_mismatches']
                        for c in record['inactive_native_checks'].values()):
                        raise ValueError('inactive native identity/weight check failed')
                    ranks[str(rank)] = record
                value = {**meta, 'raw_target_logprob': lp, 'sampled_token': out.tokens[0],
                    'unordered_route': np.sort(route[0], axis=1).tolist(), 'worker_ranks': ranks,
                    'finish': out.finish, 'stop_reason': out.stop_reason,
                    'slurm_job_id': os.environ['SLURM_JOB_ID']}
                values.append(value)
            R.atomic_json(attempt / f'batch-{index:03d}.json', {'manifest_sha256': manifest['sha256'],
                'records': values, 'batch_seconds': time.time() - before})
            for value in values:
                store.put(value)
            done = store.read()
            # Full-vocabulary objects are engineering references, not durable output.
            del batch, out
        result = result_for(manifest, store.read())
        result.update(job_id=os.environ['SLURM_JOB_ID'], load_seconds=loaded - started,
            driver_seconds_including_load=time.time() - started)
        R.atomic_json(args.out / 'QUALIFICATION.json', {**result, 'sha256': H.digest(result)})
        R.atomic_json(attempt / 'END.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'case_diagnostics'}))
        if not result['pass']:
            raise RuntimeError('numerical score calibration failed; preserve all assignments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args())
