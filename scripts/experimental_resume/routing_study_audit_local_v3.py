"""Additive reader-serialization correction; requires separate exact v3 authorization."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from moe_exp.routing_control.study_v1 import digest, require, sealed
from routing_local_engineering_v2 import save
from moe_exp.routing_control.study_v2 import seal, CAP, incumbent_actions, execution_contract
from moe_exp.routing_control.policy_stages_v2 import PrefixPolicyController
from moe_exp.routing_control.generation_v2 import execute_engineering_request
from moe_exp.routing_control.capture_v2 import begin_attempt, commit_attempt
from routing_local_engineering_v2 import compute, validate_sources, actions
from routing_local_backend_v2 import LocalBackend, source_binding
from routing_local_semantics_v2 import messages, parse_rating, audit, CheapRecognizer
from routing_study_reader_prompt_local_v3 import reader_prompt_ids
from moe_exp.routing_control.study_v1 import file_sha


def extension_binding():
    helper = Path(__file__).with_name("routing_study_reader_prompt_local_v3.py")
    return {str(Path(__file__).resolve()): file_sha(__file__), str(helper.resolve()): file_sha(helper)}


def run(manifest_path, prerequisite_path, output, deadline, authorization_path):
    compute(True)
    manifest = sealed(manifest_path)
    validate_sources(manifest)
    authorization = sealed(authorization_path)
    require(authorization['status'] == 'AUTHORIZED_ENGINEERING_ONLY' and
            authorization['manifest_sha256'] == manifest['sha256'] and
            authorization.get('extension_code_files') == extension_binding(),
            'separate exact v3 reader correction authorization required; the old bounded allocation is exhausted')
    prerequisite = sealed(prerequisite_path)
    require(prerequisite['status'] == 'PASS' and prerequisite['manifest_sha256'] == manifest['sha256'] and
            prerequisite['code_files'] == source_binding(), 'observed passing exact backend prerequisite required')
    frame = sealed(manifest_path.with_name('AUDIT_FRAME.json'))
    require(frame['sha256'] == manifest['audit_frame_sha256'] and len(frame['records']) <= 64, 'audit enrollment changed')
    output.mkdir(parents=True, exist_ok=False)
    ratings, model, started = [], None, time.monotonic()
    failures, cells = [], []
    load_seconds, reader_seconds = None, 0.
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from rate_transition_v22_fullprefix_starts_v2 import MODEL
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        prompts = {row['uid']: reader_prompt_ids(tokenizer, messages(row)) for row in frame['records']}
        require(len(prompts) == len(frame['records']), 'duplicate audit prefix identity')
        save(output / 'PROMPT_PREFLIGHT.json', {'schema': 'routing-reader-prompt-preflight-v3',
            'frame_sha256': frame['sha256'], 'extension_code_files': extension_binding(),
            'prompts': [{'uid': row['uid'], 'input_tokens': len(prompts[row['uid']]),
                'prompt_ids_sha256': digest(prompts[row['uid']])} for row in frame['records']]})
        kwargs = dict(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2, max_model_len=49152,
            max_num_seqs=1, dtype='bfloat16', kv_cache_dtype='bfloat16', max_num_batched_tokens=8192,
            language_model_only=True, attention_config={'backend': 'FLASH_ATTN'}, gpu_memory_utilization=.80,
            enable_prefix_caching=False, async_scheduling=False, generation_config='vllm', enforce_eager=True)
        # Reader is sequential on this allocation. No live generator can emit
        # hidden tokens while the offline readers run.
        for key in list(os.environ):
            if key.startswith('STEER_'):
                del os.environ[key]
        model = LLM(**kwargs)
        load_seconds = time.monotonic() - started
        save(output / 'READER_RUNTIME.json', {'schema': 'routing-six-decision-reader-runtime-v2', 'kwargs': kwargs,
            'load_seconds': load_seconds, 'model_snapshot': str(MODEL), 'code_files': source_binding(),
            'frame_sha256': frame['sha256'], 'readers': [0, 1], 'max_tokens': 1024,
            'extension_code_files': extension_binding(), 'authorization_sha256': authorization['sha256']})
        for row in frame['records']:
            prompt = prompts[row['uid']]
            for reader in (0, 1):
                if time.time() >= deadline:
                    raise TimeoutError('audit allocation deadline reached')
                stamp = time.monotonic()
                save(output / f'ATTEMPT_{row["uid"]}_{reader}.json', {'schema': 'routing-semantic-reader-attempt-v2',
                    'uid': row['uid'], 'reader': reader, 'frame_sha256': frame['sha256'],
                    'prompt_ids_sha256': digest(prompt), 'input_tokens': len(prompt), 'maximum_output_tokens': 1024,
                    'status': 'ACTUAL_OUTPUT_UNKNOWN_UNTIL_COMMITTED'})
                if len(prompt) + 1024 > 49152:
                    finish, text, tokens = 'context_overflow', '', []
                else:
                    params = SamplingParams(temperature=.2, top_p=.95, max_tokens=1024,
                        seed=int(digest(['routing-six-reader-v2', row['uid'], reader])[:8], 16) % 2_000_000_000,
                        skip_special_tokens=False)
                    completion = model.generate([{'prompt_token_ids': prompt}], params, use_tqdm=False)[0].outputs[0]
                    finish, text, tokens = completion.finish_reason, completion.text, list(completion.token_ids)
                elapsed = time.monotonic() - stamp
                reader_seconds += elapsed
                receipt = save(output / f'READER_{row["uid"]}_{reader}.json', {'schema': 'routing-semantic-reader-result-v2',
                    'uid': row['uid'], 'reader': reader, 'frame_sha256': frame['sha256'], 'finish_reason': finish,
                    'parsed': parse_rating(text) if finish == 'stop' else None, 'raw_completion': text,
                    'input_tokens': len(prompt) if finish != 'context_overflow' else 0,
                    'output_tokens': len(tokens), 'seconds': elapsed, 'prefix_only': True})
                ratings.append(receipt)
    except BaseException as error:
        failures.append({'phase': 'semantic_audit', 'error': repr(error)})
    finally:
        stamp = time.monotonic()
        if model is not None:
            model.llm_engine.engine_core.shutdown()
            model = None
        shutdown = time.monotonic() - stamp
        reader_cost = save(output / 'READER_COST.json', {'schema': 'routing-semantic-reader-cost-v2',
            'gpus': 2, 'load_seconds': load_seconds, 'reader_seconds': reader_seconds, 'shutdown_seconds': shutdown,
            'output_tokens': sum(r['output_tokens'] for r in ratings), 'input_tokens': sum(r['input_tokens'] for r in ratings),
            'actual_readers': len(ratings), 'assigned_readers': 2 * len(frame['records']),
            'partial_attempts': [p.name for p in output.glob('ATTEMPT_*.json') if not
                (output / p.name.replace('ATTEMPT_', 'READER_')).exists()]})
    audits, recognition = audit(frame, ratings)
    for row in audits:
        save(output / ('AUDIT_' + row['decision'] + '.json'), {k: v for k, v in row.items() if k != 'sha256'})
    save(output / 'RECOGNIZER_QUALIFICATION.json', {k: v for k, v in recognition.items() if k != 'sha256'})
    if recognition['status'] == 'PASS' and not failures:
        backend = None
        try:
            selected = incumbent_actions()
            native = manifest['pricing_family']
            backend = LocalBackend([a.worker_action() for a in [*actions(), *selected]], output / 'generator', deadline_epoch=deadline)
            resume = seal({'status': 'PASS', 'engine_rng_state': True, 'seed_only_restart': False,
                'qualification_sha256': prerequisite['sha256'], 'pause_receipt_sha256': prerequisite['pause_receipt_sha256']})
            sampler = seal({'max_tokens': CAP, 'seeds': [0, 1], 'sampling': backend.engine_module.card_sampling(CAP, 0),
                'prefix_presence_penalty': 1.5, 'prefix_builtin_presence': 0.})
            profile = seal({'kwargs': backend.kwargs, 'runtime_env': backend.runtime['env'],
                'engine_fingerprint': backend.runtime['engine_fingerprint']})
            execution = execution_contract(checkpoint_revision=Path(backend.kwargs['model']).name, engine_profile=profile,
                sampler=sampler, resume=resume, recognizer=recognition, worker=prerequisite, family_boundary=manifest['family_boundary'])
            save(output / 'EXECUTION.json', {k: v for k, v in execution.items() if k != 'sha256'})
            for cell in manifest['bounded_16k_pricing_cells']:
                if time.time() >= deadline:
                    raise TimeoutError('pricing allocation deadline reached')
                assignment = {**cell, 'engineering_only': True, 'family': native['family'], 'problem': native['question'],
                    'original_prompt_ids': native['prompt_ids'], 'original_prompt_ids_sha256': digest(native['prompt_ids']),
                    'prefix_ids_sha256': digest(native['prefix_ids']), 'early_endpoint_window': 1024,
                    'execution_contract_sha256': execution['sha256'],
                    'actions': [json.loads(json.dumps(asdict(a))) for a in selected] if cell['arm'] == 'incumbent' else []}
                assignment['uid'] = digest(['routing-outcome-blind-price-v2', manifest['sha256'], assignment])
                enrolled = seal({'schema': 'routing-engineering-manifest-v2', 'assignments': [assignment],
                    'execution_contract_sha256': execution['sha256'], 'parent_manifest_sha256': manifest['sha256']})
                auth = seal({'status': 'AUTHORIZED_ENGINEERING_ONLY', 'manifest_sha256': enrolled['sha256'],
                    'parent_authorization_sha256': authorization['sha256']})
                root = output / assignment['uid']
                for retry in range(2):
                    attempt = begin_attempt(root, assignment, enrolled['sha256'])
                    controller = None if cell['arm'] == 'native' else PrefixPolicyController(selected, CheapRecognizer(), recognition,
                        uid=assignment['uid'], seed=cell['seed'], origin_position=len(native['prefix_ids']))
                    from utility_controller_v2 import NativeDecodeStream
                    try:
                        receipt, _, _ = execute_engineering_request(assignment, backend, NativeDecodeStream(backend.tokenizer),
                            manifest=enrolled, authorization=auth, controller=controller, prefix_ids=native['prefix_ids'],
                            capture_dir=attempt / 'capture')
                        commit_attempt(root, attempt, receipt)
                        cells.append({'assignment': assignment, 'receipt_path': str(attempt / 'SUCCESS.json'),
                            'receipt_sha256': receipt['sha256'], 'status': 'MEASURED', 'cap': CAP})
                        break
                    except BaseException as error:
                        save(attempt / 'FAILURE.json', {'schema': 'routing-price-attempt-failure-v2', 'error': repr(error),
                            'partial': getattr(error, 'routing_partial', None), 'continued_execution': False})
                        if retry or time.time() >= deadline:
                            raise
        except BaseException as error:
            failures.append({'phase': 'pricing', 'error': repr(error)})
        finally:
            if backend is not None:
                backend.close()
    else:
        failures.append({'phase': 'pricing', 'status': 'NOT_RUN_SEMANTIC_QUALIFICATION_FAILED'})
    save(output / 'PILOT.json', {'schema': 'routing-audit-price-pilot-v3', 'manifest_sha256': manifest['sha256'],
        'prerequisite_sha256': prerequisite['sha256'], 'code_files': source_binding(),
        'extension_code_files': extension_binding(), 'authorization_sha256': authorization['sha256'],
        'status': 'MEASURED' if len(cells) == 4 and not failures else 'EXPLICIT_INCOMPLETE_OR_FAILED',
        'recognizer': recognition, 'reader_cost_sha256': reader_cost['sha256'], 'pricing_cells': cells,
        'failures': failures, 'job_id': os.environ['SLURM_JOB_ID'],
        'allocation_driver_seconds': time.monotonic() - started, 'gpus': 2,
        'grading': 'Gold absent from all generator/reader inputs; offline receipt adapter only.',
        'replay_price': 'UNMEASURED_UNLESS_A_REPLAY_IS_ACTUALLY_EXECUTED',
        'scientific_dispatch_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    run(args.manifest, args.qualification, args.output, args.deadline_epoch, args.authorization)
