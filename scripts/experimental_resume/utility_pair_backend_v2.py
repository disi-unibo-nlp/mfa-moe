"""Executable, qualification-pending four-GPU original-prompt utility backend.

Two local tensor-parallel models live only for this Slurm invocation. The
generator engine is synchronous in-process, so it cannot emit hidden lookahead
while the parent runs the two-reader screen in a separate GPU process.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

import numpy as np

import utility_controller_interface_v1 as boundary
from utility_controller_v2 import TokenController, NativeDecodeStream
import utility_scout_v1 as utility
import rate_overnight_semantics_v2 as storage
import rate_transition_v22_fullprefix_starts_v2 as start_reader


def _shutdown(model):
    if model is not None:
        model.llm_engine.engine_core.shutdown()


def _side_model(pipe, devices, output):
    """Child gets only original problem and emitted prefix, never answer gold."""
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    for key in list(os.environ):
        if key.startswith('STEER_'):
            del os.environ[key]
    model = None
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from mechanism_extension_reader_contract_v2 import exact_token_ids
        start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(start_reader.MODEL, local_files_only=True)
        kwargs = dict(model=str(start_reader.MODEL), tokenizer=str(start_reader.MODEL),
                      tensor_parallel_size=2, max_model_len=49152, max_num_seqs=2,
                      dtype='bfloat16', kv_cache_dtype='bfloat16', max_num_batched_tokens=8192,
                      language_model_only=True, attention_config={'backend': 'FLASH_ATTN'},
                      gpu_memory_utilization=.80, enable_prefix_caching=False,
                      async_scheduling=False, generation_config='vllm', enforce_eager=True)
        model = LLM(**kwargs)
        import importlib.metadata
        import sys
        storage.save(Path(output) / 'SIDE_LOAD.json', {'schema': 'utility-side-load-v2',
            'seconds': time.monotonic() - start, 'model_snapshot': str(start_reader.MODEL),
            'message_source_sha256': utility.file_sha(Path(start_reader.__file__)),
            'rubric_sha256': utility.file_sha(start_reader.RUBRIC), 'gpu_count': 2,
            'kwargs': kwargs, 'python_executable': sys.executable,
            'vllm_version': importlib.metadata.version('vllm')})
        pipe.send({'status': 'READY'})
        while True:
            value = pipe.recv()
            if value is None:
                break
            request = boundary.SideRequest(**value)
            row = {'transition': request.transition, 'reader_input': request.reader_input}
            messages = start_reader.messages(row)
            prompt = exact_token_ids(tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=True, reasoning_effort='low'))
            prompt_tokens = len(prompt)
            results, raw = [], []
            for reader in (0, 1):
                started = time.monotonic()
                storage.save(Path(output) / f'SIDE_ATTEMPT_{request.request_id}_{reader}.json', {
                    'schema': 'utility-side-reader-attempt-v2', 'request_id': request.request_id,
                    'reader': reader, 'planned_prompt_tokens': prompt_tokens, 'maximum_output_tokens': 1024,
                    'status': 'ASSIGNED_ACTUAL_OUTPUT_UNKNOWN_UNTIL_COMMITTED'})
                if prompt_tokens + 1024 > 49152:
                    finish, text, generated, parsed = 'context_overflow', '', 0, None
                else:
                    params = SamplingParams(temperature=.2, top_p=.95, max_tokens=1024,
                        seed=start_reader.rating_seed(request.request_id, reader),
                        skip_special_tokens=False)
                    answer = model.generate([{'prompt_token_ids': prompt}], params, use_tqdm=False)[0].outputs[0]
                    finish, text, generated = answer.finish_reason, answer.text, len(answer.token_ids)
                    parsed = start_reader.parse_rating(text) if finish == 'stop' else None
                result = boundary.SideResult(request.request_id, reader, finish,
                    json.dumps(parsed, separators=(',', ':')) if parsed is not None else None,
                    prompt_tokens if finish != 'context_overflow' else 0,
                    generated, time.monotonic() - started)
                results.append(asdict(result))
                raw.append({'reader': reader, 'raw_completion': text, 'result': asdict(result),
                            'planned_prompt_tokens': prompt_tokens})
                storage.save(Path(output) / f'SIDE_READER_{request.request_id}_{reader}.json', {
                    'schema': 'utility-side-reader-result-v2', 'request_id': request.request_id, **raw[-1]})
            storage.save(Path(output) / f"SIDE_{request.request_id}.json", {
                'schema': 'utility-side-pair-v2', 'request': value, 'readers': raw})
            pipe.send({'status': 'COMPLETE', 'results': results})
    except BaseException as error:
        try:
            pipe.send({'status': 'ERROR', 'error': repr(error)})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        _shutdown(model)
        pipe.close()


class FourGPUBackend:
    """Concrete backend; callers must enforce qualification and pricing gates."""
    def __init__(self, selections, output: Path, *, deadline_epoch: float):
        if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID') or
                socket.gethostname().startswith('login')):
            raise RuntimeError('utility model pair requires a GPU Slurm step')
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
            raise ValueError('exactly four allocated visible GPUs required')
        self.output, self.selections, self.deadline_epoch = Path(output), selections, deadline_epoch
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.model = None
        self.side_process = None
        context = multiprocessing.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.side_process = context.Process(target=_side_model,
            args=(child, ','.join(devices[2:]), str(self.output)), name='utility-side-readers-v2')
        self.side_process.start()
        child.close()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices[:2])
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        try:
            from moe_steer import engine
            from overnight_routing_runner_v1 import build_policy_table
            actions = {action['name']: action for selected in selections.values() for action in selected['actions']}
            table = build_policy_table(list(actions.values()))
            table_path = self.output / 'POLICY_TABLE.json'
            storage.save(table_path, {k: v for k, v in table.sealed().items() if k != 'sha256'})
            self.telemetry = self.output / 'telemetry'
            self.telemetry.mkdir()
            env = engine.engine_env(table_path, self.telemetry, expect_fingerprint=engine.fingerprint()['combined'])
            env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
            overlay = utility.STAGE / 'addenda/ordered/9727c10299b71e7a/moe_exp_src'
            env['PYTHONPATH'] = os.pathsep.join((str(overlay), str(Path(__file__).parent),
                str(utility.REPO / 'src'), env['PYTHONPATH']))
            kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=1, enforce_eager=True,
                                          return_routed_experts=True)
            kwargs['worker_extension_cls'] = 'utility_routing_worker_v2.UtilityWorkerExtension'
            kwargs['logits_processors'] = [*kwargs['logits_processors'],
                'utility_qualification_logits_v2:UtilityQualificationTokens']
            kwargs['gpu_memory_utilization'] = .80
            from run_boundary_micro_screen import prepare_worker_import_path
            prepare_worker_import_path(engine, env, overlay)
            self.model = engine.build_llm(kwargs)
            self.engine_module = engine
            self.tokenizer = self.model.get_tokenizer()
            if type(self.model.llm_engine.engine_core).__name__ != 'InprocClient':
                raise ValueError('generator may run ahead while side query is pending')
            ready = self._receive()
            if ready != {'status': 'READY'}:
                raise ValueError(f'side model did not become ready: {ready}')
            storage.save(self.output / 'PAIR_LOAD.json', {'schema': 'utility-pair-load-v2',
                'load_wall_seconds': time.monotonic() - self.started, 'allocated_gpu_count': 4,
                'generator_kwargs': kwargs, 'worker_status': self.model.collective_rpc('utility_status'),
                'qualification': 'PENDING_EXACT_GPU_CHECKS'})
        except BaseException:
            self.close()
            raise

    def _receive(self):
        while not self.pipe.poll(1):
            if time.time() >= self.deadline_epoch:
                raise TimeoutError('utility pair deadline expired')
            if not self.side_process.is_alive():
                raise RuntimeError('side process exited without response')
        value = self.pipe.recv()
        if value.get('status') == 'ERROR':
            raise RuntimeError(value['error'])
        return value

    def screen(self, request):
        storage.save(self.output / f'SIDE_ASSIGNMENT_{request.request_id}.json', {
            'schema': 'utility-side-assignment-v2', 'request': asdict(request),
            'readers': [0, 1], 'max_tokens_per_reader': 1024})
        self.pipe.send(asdict(request))
        result = self._receive()
        if result['status'] != 'COMPLETE':
            raise RuntimeError('side reader result incomplete')
        return [boundary.SideResult(**row) for row in result['results']]

    def generate_one(self, assignment, prompt_ids, problem, *, max_tokens=16384, controller=None,
                     qualification=False, qualification_preempt_at=None, qualification_token_ids=None):
        """Single original-prompt request, native or selected policy, preserving RNG."""
        from vllm import SamplingParams
        if (utility.digest(prompt_ids) != assignment['prompt_token_ids_sha256'] or
                len(prompt_ids) != assignment['prompt_tokens'] or
                (not qualification and (max_tokens != 16384 or controller is not None or
                                         qualification_preempt_at is not None or qualification_token_ids is not None)) or
                (qualification and (controller is None or not 1 <= max_tokens <= 16384))):
            raise ValueError('original prompt or 16k sampler cap differs')
        ctl = controller or TokenController(uid=assignment['uid'], arm=assignment['arm'], problem=problem,
            prompt_sha256=assignment['prompt_token_ids_sha256'], selections=self.selections,
            decode=NativeDecodeStream(self.tokenizer), screen=self.screen)
        extra = {'steer': {'uid': assignment['uid'], 'policy_index': 0, 'prefix_len': 0}}
        if qualification_token_ids is not None:
            if len(qualification_token_ids) != max_tokens:
                raise ValueError('engineering token sequence must cover its complete fixed cap')
            extra['utility_engineering_token_sequence'] = qualification_token_ids
        params = SamplingParams(**self.engine_module.card_sampling(max_tokens, assignment['seed'],
            extra_args=extra,
            routed_experts_prompt_start=len(prompt_ids) - 1))
        engine = self.model.llm_engine
        rid = engine.add_request(assignment['uid'], {'prompt_token_ids': prompt_ids}, params)
        started = time.monotonic()
        routes = None
        preemptions = []
        try:
            while engine.has_unfinished_requests():
                if time.time() >= self.deadline_epoch:
                    raise TimeoutError('utility generator reached allocation deadline')
                outputs = engine.step()
                for output in outputs:
                    if output.request_id != assignment['uid'] or len(output.outputs) != 1:
                        raise ValueError('unexpected neighboring generator request')
                    completion = output.outputs[0]
                    ids = list(completion.token_ids)
                    episode = ctl.observe_cumulative(ids, completion.finish_reason if output.finished else None)
                    if episode:
                        receipts = self.model.collective_rpc('utility_set_episode',
                            args=(assignment['uid'], episode, ids))
                        if (len(receipts) != 2 or {r['rank'] for r in receipts} != {0, 1} or
                                any(r['episode_sha256'] != utility.digest(episode) for r in receipts)):
                            raise ValueError('dynamic episode was not acknowledged on both TP ranks')
                    if output.finished:
                        routes = np.asarray(completion.routed_experts)
                    elif qualification_preempt_at is not None and len(ids) == qualification_preempt_at:
                        reset = self.model.reset_prefix_cache(reset_running_requests=True)
                        preemptions.append({'emitted_tokens': len(ids), 'reset_ok': bool(reset)})
            self.model.collective_rpc('steer_flush')
            if ctl.state.finish not in ('stop', 'length') or routes is None or routes.shape != (
                    len(ctl.state.completion_token_ids), 40, 8):
                raise ValueError('completion, route rows or natural/length stop differs')
            return {'assignment': assignment, 'controller': ctl.audit(), 'routed': routes,
                    'qualification_only': qualification, 'maximum_tokens': max_tokens,
                    'forced_preemption_receipts': preemptions,
                    'generation_wall_seconds': time.monotonic() - started}
        except BaseException as error:
            error.utility_partial = {'assignment': assignment, 'controller': ctl.audit(),
                'generation_wall_seconds': time.monotonic() - started,
                'forced_preemption_receipts': preemptions}
            try:
                engine.abort_request([rid], internal=True)
            except BaseException:
                pass
            raise

    def close(self):
        _shutdown(self.model)
        self.model = None
        if self.side_process is not None:
            if self.side_process.is_alive():
                try:
                    self.pipe.send(None)
                except (BrokenPipeError, EOFError):
                    pass
                self.side_process.join(timeout=60)
                if self.side_process.is_alive():
                    self.side_process.terminate()
                    self.side_process.join(timeout=30)
            self.pipe.close()
            self.side_process = None
        self.output.mkdir(parents=True, exist_ok=True)
        storage.save(self.output / 'PAIR_COST.json', {'schema': 'utility-pair-cost-v2',
            'allocated_gpu_count': 4, 'allocated_driver_wall_seconds': time.monotonic() - self.started,
            'driver_gpu_hours': 4 * (time.monotonic() - self.started) / 3600,
            'scope': 'All four allocated GPUs including model loading, reader/generator waiting, decoding and shutdown. Slurm allocation start/end overhead must be added once; do not add side-model GPU-hours a second time.'})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
