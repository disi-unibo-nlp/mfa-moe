"""Allocated-GPU qualification of single-position counterfactual routing loss."""
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


def local_module(name, filename):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / 'src/moe_exp/routing_control' / (
            'counterfactual.py' if name == 'cf_helpers' else filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = local_module('cf_helpers', 'counterfactual_helpers.py')
R = local_module('cf_receipts', 'receipts.py')


class ResetDuringPrefill:
    def __init__(self, inner, calls):
        self.inner, self.calls, self.count, self.events = inner, set(calls), 0, []
    def __getattr__(self, name):
        return getattr(self.inner, name)
    def step(self):
        self.count += 1
        if self.count in self.calls:
            flying = self.inner.in_flight()
            success = bool(self.inner.llm.reset_prefix_cache(reset_running_requests=True)) if flying else False
            self.events.append({'step': self.count, 'in_flight': flying, 'success': success})
        return self.inner.step()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(manifest):
    all_cases = []
    for i, row in enumerate(manifest['rows']):
        for position in manifest['positions']:
            prompt, target = H.prediction_input(row, position)
            for arm in manifest['arms']:
                ids = manifest['random_pairs'][i] if arm['ids'] == 'by_prefix' else arm['ids']
                policy = arm['name'] if not arm['name'].startswith('random_') else f'{arm["name"]}_{i}'
                if arm['operator'] == 'none':
                    policy = 'zero'
                meta = {'prefix_uid': row['uid'], 'family': row['family'], 'position': position,
                    'mode': 'score', 'arm': arm['name'], 'policy': policy,
                    'operator': arm['operator'], 'targets': [28, ids],
                    'prompt_sha256': H.digest(prompt), 'prompt_len': len(prompt),
                    'prefix_len': len(prompt) - len(row['prompt_ids']), 'target_token': target}
                meta['uid'] = H.digest(['counterfactual-qual-v1', manifest['sha256'], meta])
                all_cases.append((prompt, meta))
    for row in manifest['rows']:
        for position in manifest['positions']:
            prefix, target = H.prediction_input(row, position)
            prompt = prefix + [target]
            meta = {'prefix_uid': row['uid'], 'family': row['family'], 'position': position,
                'mode': 'teacher_force', 'arm': 'native_teacher_force', 'policy': 'zero',
                'operator': 'none', 'targets': [28, [9, 189]], 'prompt_len': len(prompt),
                'prompt_sha256': H.digest(prompt), 'prefix_len': len(prompt) - len(row['prompt_ids']),
                'target_token': target}
            meta['uid'] = H.digest(['counterfactual-qual-v1', manifest['sha256'], meta])
            all_cases.append((prompt, meta))
    row = manifest['rows'][0]
    prompt = row['prompt_ids'] + row['prefix_ids'] + [248069]
    for arm in ('native', 'target_bias1', 'target_force_positive', 'target_force_negative'):
        source = next(a for a in manifest['arms'] if a['name'] == arm)
        meta = {'prefix_uid': row['uid'], 'family': row['family'], 'position': None, 'mode': 'closed',
            'arm': arm, 'policy': 'zero' if arm == 'native' else arm,
            'operator': source['operator'], 'targets': [28, [9, 189]],
            'prompt_len': len(prompt), 'prompt_sha256': H.digest(prompt),
            'prefix_len': len(prompt) - len(row['prompt_ids']), 'target_token': row['native_suffix_ids'][0]}
        meta['uid'] = H.digest(['counterfactual-qual-v1', manifest['sha256'], meta])
        all_cases.append((prompt, meta))
    if len(all_cases) != manifest['requests']:
        raise ValueError('qualification assignment count differs')
    return all_cases


def table_for(manifest, P, TargetSet, Schedule):
    policies = []
    for arm in manifest['arms']:
        if arm['operator'] == 'none':
            continue
        pools = enumerate(manifest['random_pairs']) if arm['ids'] == 'by_prefix' else [(None, arm['ids'])]
        for index, ids in pools:
            name = arm['name'] if index is None else f'{arm["name"]}_{index}'
            target = TargetSet(name, 'CUSTOM', ((28, tuple(sorted(ids))),),
                'discovery engineering; single-position counterfactual loss')
            operator = (P.Operator('bias', 1, arm['magnitude']) if arm['operator'] == 'bias' else
                P.Operator('force', 1 if arm['operator'] == 'force_positive' else -1, 0.))
            policies.append(P.make_policy(target, operator, Schedule('always'), name=name))
    return P.build_table(policies)


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('counterfactual inference requires an allocated GPU Slurm step')
    from moe_steer import engine, policies as P, qualify as Q
    from moe_steer.spec import TargetSet, Schedule
    manifest = H.sealed(args.manifest)
    if manifest['driver_sha256'] != sha(__file__) or manifest['base_tree_sha256'] != engine.code_tree_sha256():
        raise ValueError('counterfactual code binding changed')
    if any(sha(path) != expected for path, expected in manifest['code_files'].items()):
        raise ValueError('counterfactual helper or qualified worker changed')
    worker = H.sealed(manifest['worker_binding'])
    overlay = Path(manifest['worker_overlay'])
    if worker['overlay'] != str(overlay):
        raise ValueError('worker overlay differs')
    assigned = cases(manifest)
    binding = {'manifest_sha256': manifest['sha256'], 'driver_sha256': sha(__file__),
        'base_tree_sha256': manifest['base_tree_sha256'], 'qualification': 'single-position discovery engineering'}
    with R.ReceiptStore(args.out, binding, [meta['uid'] for _, meta in assigned]) as store:
        if len(store.read()) == len(assigned) and (args.out / 'QUALIFICATION.json').exists():
            return
        attempt = store.attempt()
        started = time.time()
        R.atomic_json(attempt / 'START.json', {'job_id': os.environ['SLURM_JOB_ID'], 'assigned': len(assigned)})
        table = table_for(manifest, P, TargetSet, Schedule)
        table_path = args.out / 'policy-table.json'
        if table_path.exists() and json.loads(table_path.read_text()) != table.sealed():
            raise ValueError('same-bound policy table changed')
        if not table_path.exists():
            R.atomic_json(table_path, table.sealed())
        deadline = Q.Deadline.from_env()
        deadline.require(950, 'cold load and first counterfactual batch')
        fingerprint = engine.fingerprint()
        telemetry = attempt / 'telemetry'
        telemetry.mkdir()
        env = engine.engine_env(table_path, telemetry, expect_fingerprint=fingerprint['combined'])
        env['PYTHONPATH'] = os.pathsep.join((str(overlay), env['PYTHONPATH']))
        engine.apply_env(env)
        sys.path[:] = [str(overlay), *(p for p in sys.path if p != str(overlay))]
        import moe_exp
        spec = importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
        if Path(moe_exp.__file__).resolve().parent != overlay / 'moe_exp' or Path(spec.origin).resolve() != overlay / 'moe_exp/routing_control/ordered_vllm.py':
            raise ValueError('spawn worker import is outside the qualified overlay')
        kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=48, return_routed_experts=True)
        kwargs.update(worker_extension_cls='moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension',
            gpu_memory_utilization=.80, long_prefill_token_threshold=1024, logprobs_mode='raw_logprobs')
        model = engine.build_llm(kwargs)
        driver = Q.QDriver(model, plugin=True)
        harness = Q.Harness(driver, telemetry, deadline=deadline, abort_margin=90.)
        preempt = ResetDuringPrefill(driver, manifest['preemption_calls'])
        recomputed = 0
        checks = []
        # Preserve entire eight-arm position blocks; teacher-forced and closed controls are separate.
        groups = [assigned[i:i + 32] for i in range(0, 128, 32)] + [assigned[128:144], assigned[144:]]
        done = store.read()
        for batch_index, group in enumerate(groups):
            pending = [(prompt, meta) for prompt, meta in group if meta['uid'] not in done]
            if not pending:
                continue
            deadline.require(90, 'counterfactual batch ' + str(batch_index))
            reqs = []
            for prompt, meta in pending:
                extra = Q.steer_extra(table, meta['uid'], meta['policy'],
                    len(prompt) - meta['prefix_len'], prefix_len=meta['prefix_len'], restore_presence=False)
                sampling = Q.greedy_params(1, extra, presence=0., routed_start=len(prompt) - 1,
                    prompt_logprobs=0 if meta['mode'] == 'teacher_force' else None)
                sampling['logprob_token_ids'] = [meta['target_token']]
                reqs.append(Q.QReq(meta['uid'], prompt, sampling))
            harness.driver = preempt if batch_index == 0 else driver
            batch_started = time.time()
            batch = harness.run(reqs, cap_in_flight=48)
            harness.flush()
            records, _, _ = harness.telemetry()
            arrays, result_rows = {}, []
            for prompt, meta in pending:
                out = batch.outcomes.get(meta['uid'])
                if out is None or out.error or len(out.tokens) != 1 or out.routed is None:
                    raise ValueError('missing single-token assigned result')
                route = np.asarray(out.routed)
                if route.shape != (1, 40, 8):
                    raise ValueError('single prediction route shape differs')
                raw_lp = H.designated_logprob(out.logprobs[0], meta['target_token'])
                if meta['mode'] == 'teacher_force':
                    raw_lp = H.designated_logprob(out.prompt_logprobs[-1], meta['target_token'])
                operator = meta['operator'] if meta['mode'] == 'score' else 'none'
                hits = H.verify_route(route[0].tolist(), meta['targets'], operator)
                rank_rows = {}
                expected_active = int(meta['mode'] == 'score' and meta['operator'] != 'none')
                for rank in (0, 1):
                    record = records.get(rank, {}).get(meta['uid'])
                    if record is None:
                        raise ValueError('missing worker rank receipt')
                    inactive = record['inactive_native_checks']
                    if any(c['expert_identity_mismatches'] or c['weight_mismatches'] for c in inactive.values()):
                        raise ValueError('inactive routing mismatch')
                    if record['cpu_active_rows'] != expected_active:
                        raise ValueError('prediction-row action count or closure mismatch')
                    recomputed += record['recompute_rows']
                    rank_rows[str(rank)] = record
                value = {**meta, 'raw_designated_logprob': raw_lp, 'native_token_nll': -raw_lp,
                    'sampled_token': out.tokens[0], 'finish': out.finish, 'stop_reason': out.stop_reason,
                    'target_hits': hits, 'route': route[0].tolist(), 'worker_ranks': rank_rows,
                    'slurm_job_id': os.environ['SLURM_JOB_ID']}
                arrays[meta['uid']] = route
                result_rows.append(value)
            np.savez_compressed(attempt / f'batch-{batch_index:03d}.npz', **arrays)
            R.atomic_json(attempt / f'batch-{batch_index:03d}.json', {
                'manifest_sha256': manifest['sha256'], 'records': result_rows,
                'batch_seconds': time.time() - batch_started,
                'array_sha256': sha(attempt / f'batch-{batch_index:03d}.npz')})
            for value in result_rows:
                store.put(value)
            checks.append({'batch': batch_index, 'requests': len(pending), 'pass': True})
        completed = store.read()
        lookup = {(r['prefix_uid'], r['position'], r['arm']): r for r in completed.values() if r['mode'] != 'closed'}
        alignment, repeat = [], []
        for row in manifest['rows']:
            for j in manifest['positions']:
                native = lookup[row['uid'], j, 'native']['raw_designated_logprob']
                alignment.append(abs(native - lookup[row['uid'], j, 'native_teacher_force']['raw_designated_logprob']))
                repeat.append(abs(native - lookup[row['uid'], j, 'native_repeat']['raw_designated_logprob']))
        passed_alignment = np.mean(alignment) <= manifest['teacher_force_alignment_mean_tolerance'] and max(alignment) <= manifest['teacher_force_alignment_max_tolerance']
        recovery = bool(any(e['success'] for e in preempt.events) and recomputed > 0)
        result = {'schema': 'routing-counterfactual-qualification-result-v1', 'manifest_sha256': manifest['sha256'],
            'job_id': os.environ['SLURM_JOB_ID'], 'assigned': len(assigned), 'completed': len(completed),
            'pass': bool(len(completed) == len(assigned) and passed_alignment and recovery),
            'alignment_mean_abs': float(np.mean(alignment)), 'alignment_max_abs': max(alignment),
            'alignment_pass': bool(passed_alignment), 'native_repeat_mean_abs': float(np.mean(repeat)),
            'native_repeat_max_abs': max(repeat), 'reset_events': preempt.events,
            'recomputed_rank_rows': recomputed, 'recovery_pass': recovery, 'batch_checks': checks,
            'driver_seconds_including_load': time.time() - started,
            'interpretation': 'engineering measurement qualification only; no semantic expert ranking or steering claim'}
        R.atomic_json(args.out / 'QUALIFICATION.json', {**result, 'sha256': H.digest(result)})
        R.atomic_json(attempt / 'END.json', result)
        print(json.dumps(result))
        if not result['pass']:
            raise RuntimeError('counterfactual measurement qualification failed; preserve raw results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args())
