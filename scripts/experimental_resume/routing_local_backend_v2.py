"""Two-GPU, one-request generator; no model imports on login-node inspection."""
from __future__ import annotations

import os
import json
from pathlib import Path
import socket
import time
import uuid
import numpy as np

from moe_exp.routing_control.study_v1 import digest, require, file_sha
from moe_exp.routing_control.study_v2 import seal, verify, CAP

KERNEL_ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24')


def array_binding(value):
    import hashlib
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': value.dtype.str,
            'bytes': value.nbytes, 'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def source_binding():
    repo = Path(__file__).resolve().parents[2]
    files = [*sorted((repo / 'src/moe_exp/routing_control').glob('*.py')),
             *[Path(__file__).with_name(n) for n in ('routing_local_backend_v2.py', 'routing_local_worker_v2.py',
                 'routing_local_engineering_v2.py', 'routing_local_semantics_v2.py', 'routing_local_audit_v2.py',
                 'routing_local_logits_v2.py', 'utility_controller_v2.py',
                 'utility_qualification_logits_v2.py', 'overnight_routing_runner_v1.py')],
             *[repo / 'src/moe_exp' / n for n in ('correlation_pipeline/model_profiles.py', 'models/token_replay.py',
                'correlation_pipeline/scoring.py', 'correlation_pipeline/answer_equivalence.py', 'correlation_pipeline/outcomes.py')],
             *sorted(Path(__file__).parent.glob('routing_local_*v2.sbatch')),
             *sorted((KERNEL_ROOT / 'moe_steer').glob('*.py'))]
    return {str(p): file_sha(p) for p in files}


class LocalBackend:
    def __init__(self, actions, output, *, deadline_epoch, qualification=None):
        require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
                not socket.gethostname().startswith('login'), 'model loading requires GPU Slurm step')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        require(len(visible) == len(set(visible)) == 2 and all(visible), 'exactly two allocated GPUs required')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline_epoch, self.model, self.live = deadline_epoch, None, None
        self.qualification = qualification or seal({'status': 'PENDING_ENGINEERING_QUALIFICATION'})
        self.started, self.metrics = time.monotonic(), {'command_rpc_seconds': 0., 'capture_rpc_seconds': 0.,
            'engine_step_seconds': 0., 'prefill_seconds': 0., 'decode_seconds': 0., 'pause_rpc_seconds': 0.,
            'engine_steps': 0, 'requests': 0, 'processed_input_tokens': 0, 'emitted_tokens': 0}
        try:
            import sys
            sys.path.insert(0, str(KERNEL_ROOT))
            from moe_steer import engine
            from overnight_routing_runner_v1 import build_policy_table
            from moe_exp.routing_control.study_v1 import save
            table = build_policy_table(actions)
            table_path = self.output / 'POLICY_TABLE.json'
            save(table_path, {k: v for k, v in table.sealed().items() if k != 'sha256'})
            telemetry = self.output / 'telemetry'
            telemetry.mkdir()
            env = engine.engine_env(table_path, telemetry, expect_fingerprint=engine.fingerprint()['combined'],
                                    moe_exp_src=Path(__file__).resolve().parents[2] / 'src', code_root=KERNEL_ROOT)
            env['PYTHONPATH'] = os.pathsep.join((str(Path(__file__).parent), str(Path(__file__).resolve().parents[2] / 'src'), str(KERNEL_ROOT)))
            env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
            os.environ.update(env)
            self.kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=1, enforce_eager=True, return_routed_experts=True)
            self.kwargs.update(worker_extension_cls='routing_local_worker_v2.LocalWorkerExtension', gpu_memory_utilization=.80)
            self.kwargs['logits_processors'].append('utility_qualification_logits_v2:UtilityQualificationTokens')
            self.kwargs['logits_processors'].append('routing_local_logits_v2:EngineeringRepeat')
            self.engine_module = engine
            self.model = engine.build_llm(self.kwargs)
            require(type(self.model.llm_engine.engine_core).__name__ == 'InprocClient', 'engine can run ahead')
            self.tokenizer = self.model.get_tokenizer()
            self.load_seconds = time.monotonic() - self.started
            self.runtime = seal({'schema': 'routing-local-runtime-v2', 'kwargs': self.kwargs,
                'env': {k: os.environ.get(k) for k in ('VLLM_ENABLE_V1_MULTIPROCESSING', 'VLLM_USE_V2_MODEL_RUNNER',
                    'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_BATCH_INVARIANT')},
                'engine_fingerprint': engine.fingerprint(), 'code_files': source_binding(),
                'worker_status': self.model.collective_rpc('steer_status'), 'load_seconds': self.load_seconds,
                'job_id': os.environ['SLURM_JOB_ID'], 'gpus': 2, 'live_requests': 1})
            self.runtime = seal(json.loads(json.dumps({k: v for k, v in self.runtime.items() if k != 'sha256'}, allow_nan=False)))
            save(self.output / 'RUNTIME.json', {k: v for k, v in self.runtime.items() if k != 'sha256'})
        except BaseException:
            self.close()
            raise

    def _deadline(self):
        if time.time() >= self.deadline_epoch:
            raise TimeoutError('bounded allocation deadline reached')

    def start(self, assignment, prefix_ids, max_new_tokens=CAP):
        self._deadline()
        require(self.live is None and not self.model.llm_engine.has_unfinished_requests(), 'another assignment is live')
        prompt, prefix = list(assignment['original_prompt_ids']), list(prefix_ids)
        require(prompt and all(type(t) is int and t >= 0 for t in prompt + prefix), 'invalid original prompt or prefix')
        require(digest(prompt) == assignment['original_prompt_ids_sha256'] and
                digest(prefix) == assignment['prefix_ids_sha256'], 'prompt/prefix source hash mismatch')
        require(len(prompt) + len(prefix) + max_new_tokens <= self.kwargs['max_model_len'], 'input exceeds model context; truncation forbidden')
        require(1 <= max_new_tokens <= CAP and assignment['seed'] in (0, 1), 'invalid sampler cap/seed')
        engineering = assignment.get('engineering_only') is True
        if not engineering:
            q = verify(self.qualification)
            require(q.get('status') == 'PASS' and max_new_tokens == CAP and
                    q.get('execution_contract_sha256') == assignment.get('execution_contract_sha256'), 'scientific backend qualification differs')
        from vllm import SamplingParams
        steer = {'uid': assignment['uid'], 'policy_index': 0, 'prefix_len': len(prefix)}
        if prefix:
            steer['prefix_presence'] = {'penalty': 1.5, 'prefix_start': len(prompt)}
        extra = {'steer': steer}
        sequence = assignment.get('engineering_token_sequence')
        if sequence is not None:
            require(engineering and len(sequence) == max_new_tokens, 'forced sequence is engineering only')
            extra['utility_engineering_token_sequence'] = sequence
        repeat = assignment.get('engineering_repeat')
        if repeat is not None:
            require(engineering and sequence is None and repeat['cap'] == max_new_tokens, 'repeat fixture is engineering only')
            extra['routing_engineering_repeat'] = repeat
        params = SamplingParams(**self.engine_module.card_sampling(max_new_tokens, assignment['seed'],
            presence_penalty=0. if prefix else 1.5, extra_args=extra,
            routed_experts_prompt_start=len(prompt) + len(prefix) - 1))
        rid = self.model.llm_engine.add_request(assignment['uid'], {'prompt_token_ids': prompt + prefix}, params)
        self.live = {'assignment': assignment, 'rid': rid, 'prefix': prefix, 'prompt': prompt, 'tokens': [], 'routed': [],
                     'cap': max_new_tokens, 'finished': False, 'paused': None, 'input_sha256': digest(prompt + prefix)}
        self.metrics['requests'] += 1

    def step(self, policy_name, worker_action, prediction_position):
        self._deadline()
        live = self.live
        require(live is not None and not live['finished'] and live['paused'] is None, 'request absent, finished or paused')
        require(prediction_position == len(live['prefix']) + len(live['tokens']), 'skipped/duplicated prediction row')
        command = {'uid': live['assignment']['uid'], 'arm': live['assignment']['arm'], 'policy': policy_name,
            'action': worker_action, 'position': prediction_position, 'offset': len(live['tokens']),
            'original_prompt_tokens': len(live['prompt']), 'prefix_tokens': len(live['prefix']),
            'input_sha256': live['input_sha256'], 'last_token_id': live['tokens'][-1] if live['tokens'] else None}
        start = time.monotonic()
        ack = self.model.collective_rpc('routing_local_command', args=(command,))
        self.metrics['command_rpc_seconds'] += time.monotonic() - start
        require(len(ack) == 2 and {r['rank'] for r in ack} == {0, 1} and
                all((r['uid'], r['position']) == (command['uid'], prediction_position) for r in ack), 'both-rank action acknowledgement differs')
        engine = self.model.llm_engine
        completion = None
        while completion is None:
            self._deadline()
            start = time.monotonic()
            outputs = engine.step()
            elapsed = time.monotonic() - start
            self.metrics['engine_step_seconds'] += elapsed
            self.metrics['prefill_seconds' if not live['tokens'] else 'decode_seconds'] += elapsed
            self.metrics['engine_steps'] += 1
            for output in outputs:
                require(output.request_id == command['uid'] and len(output.outputs) == 1, 'neighboring/stale request output')
                item = output.outputs[0]
                if len(item.token_ids) == len(live['tokens']):
                    continue
                require(list(item.token_ids[:-1]) == live['tokens'] and len(item.token_ids) == len(live['tokens']) + 1,
                        'engine emitted hidden or reordered tokens')
                completion, finished = item, output.finished
        start = time.monotonic()
        capture = self.model.collective_rpc('routing_local_capture', args=(command['uid'], prediction_position))
        self.metrics['capture_rpc_seconds'] += time.monotonic() - start
        capture.sort(key=lambda r: r['rank'])
        require(len(capture) == 2 and [r['rank'] for r in capture] == [0, 1] and
                all((r['uid'], r['position'], r['engine_input_row']) == (command['uid'], prediction_position,
                    len(live['prompt']) + prediction_position - 1) for r in capture), 'capture coordinate/rank mismatch')
        require(np.array_equal(capture[0]['ids'], capture[1]['ids']) and
                np.array_equal(capture[0]['weights'], capture[1]['weights']), 'both-rank capture disagreement')
        token = int(completion.token_ids[-1])
        live['tokens'].append(token)
        live['routed'].append(capture[0]['ids'][1])
        live['finished'] = bool(finished)
        if finished:
            builtin = np.asarray(completion.routed_experts)
            require(builtin.shape == (len(live['tokens']), 40, 8) and np.array_equal(builtin, np.asarray(live['routed'])),
                    'raw executed capture differs from independent vLLM routed-expert output')
        processed = capture[0]['processed_input_tokens']
        require(capture[1]['processed_input_tokens'] == processed, 'rank input accounting differs')
        self.metrics['processed_input_tokens'] += processed
        self.metrics['emitted_tokens'] += 1
        return {'token_id': token, 'row_position': prediction_position, 'policy_name': policy_name,
            'native_ids': capture[0]['ids'][0], 'executed_ids': capture[0]['ids'][1],
            'native_weights': capture[0]['weights'][0], 'selected_weights': capture[0]['weights'][1],
            'rank_captures': capture, 'finished': bool(finished), 'finish_reason': completion.finish_reason,
            'processed_input_tokens': processed}

    def inspect(self):
        require(self.live is not None, 'no live assignment to inspect')
        start = time.monotonic()
        ranks = self.model.collective_rpc('routing_local_state')
        self.metrics['pause_rpc_seconds'] += time.monotonic() - start
        body = {'assignment_uid': self.live['assignment']['uid'],
            'next_prediction_index': len(self.live['prefix']) + len(self.live['tokens']),
            'input_sha256': self.live['input_sha256'], 'emitted_ids_sha256': digest(self.live['tokens']),
            'ranks': ranks, 'engine_instance': self.instance_id,
            'rng_state_sha256': digest([r['rng'] for r in ranks]),
            'engine_state_sha256': digest(ranks), 'scope': 'LIVE_ENGINE_ONLY', 'seed_only_restart': False}
        return seal(body)

    @property
    def instance_id(self):
        if not hasattr(self, '_instance_id'):
            self._instance_id = uuid.uuid4().hex
        return self._instance_id

    def pause(self):
        require(self.live is not None and self.live['paused'] is None, 'no running assignment')
        snapshot = self.inspect()
        self.live['paused'] = snapshot
        return snapshot

    def resume(self, snapshot):
        verify(snapshot)
        require(self.live is not None and self.live['paused'] == snapshot and self.inspect() == snapshot,
                'live engine/RNG state changed; seed restart is not continuation')
        self.live['paused'] = None

    def finish(self):
        if self.live is None or self.model is None:
            return
        live, engine = self.live, self.model.llm_engine
        if engine.has_unfinished_requests():
            engine.abort_request([live['rid']], internal=True)
        self.model.collective_rpc('steer_flush')
        self.model.collective_rpc('routing_local_release', args=(live['assignment']['uid'],))
        self.live = None

    def close(self):
        start = time.monotonic()
        if self.model is not None:
            try:
                self.finish()
            finally:
                self.model.llm_engine.engine_core.shutdown()
                self.model = None
        self.metrics['shutdown_seconds'] = time.monotonic() - start
        self.metrics['allocated_driver_seconds'] = time.monotonic() - self.started
        from moe_exp.routing_control.study_v1 import save
        save(self.output / 'COST.json', {'schema': 'routing-local-backend-cost-v2', 'gpus': 2,
            'load_seconds': getattr(self, 'load_seconds', None), 'metrics': self.metrics,
            'driver_gpu_hours': 2 * self.metrics['allocated_driver_seconds'] / 3600,
            'scope': 'Both GPUs, all load/prefill/generation/capture/controller waiting/partial attempts/shutdown; add Slurm boundary overhead once.'})
