"""Frozen 96-family, original-prompt utility-scout preparation and audit.

This is an inert CPU-side contract. It does not select a policy, generate text,
grade answers, or submit Slurm jobs. A future generator must bind a validated
policy and qualified engine to this exact assignment plan before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
STAGE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
UNITS = STAGE / 'runs/routing-control-v1/dense-utility/UTILITY_UNITS_v1.json'
PROMPTS = STAGE / 'manifests/prompt-table-v1.json'
QUESTIONS = STAGE / 'manifests/question-table-v1.json'
ARMS = ('native', 'frozen_policy')
SEEDS = (0, 1)
MAX_TOKENS = 16_384


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed JSON seal: {path}')
    return value


def seal(value: dict) -> dict:
    return {**value, 'sha256': digest(value)}


def _exact_nonnegative_ids(ids: Any) -> bool:
    return isinstance(ids, list) and bool(ids) and all(type(x) is int and x >= 0 for x in ids)


def build_plan(family: dict, units: dict, prompts: dict, *, prompt_file_sha256: str) -> dict:
    """Assign every utility family twice per seed using original Qwen3.6 prompts."""
    if family.get('schema') != 'routing-control-family-freeze-v1':
        raise ValueError('unexpected family freeze')
    if (units.get('schema') != 'dense-utility-units-v1' or units.get('pool') != 'utility'
            or units.get('family_freeze_sha256') != family['sha256']):
        raise ValueError('utility units do not match the frozen family split')
    if (prompts.get('schema') != 'steer-prompts-v1' or
            prompt_file_sha256 != family['inputs']['prompts']['sha256']):
        raise ValueError('original prompt table differs from frozen family source')
    pools = family['new_parent_pools']['parent_pools']
    ordered = pools['utility']
    if (len(ordered) != 96 or len(set(ordered)) != 96 or
            set(ordered) & (set(pools['discovery']) | set(pools['mechanism']))):
        raise ValueError('utility family enrollment or disjointness differs')
    representatives = family['new_parent_pools']['representative_questions']
    records = units['records']
    if (units.get('families') != 96 or units.get('attempts') != 96 or
            units.get('sentences') != len(records) or
            {r['family'] for r in records} != set(ordered) or
            any(r['question'] != representatives[r['family']] for r in records)):
        raise ValueError('utility native-unit coverage differs from 96 frozen representatives')
    source = {'family_freeze_sha256': family['sha256'],
              'utility_units_sha256': units['sha256'],
              'prompt_table_sha256': prompts['sha256'],
              'prompt_table_file_sha256': prompt_file_sha256}
    rows = []
    for family_id in ordered:
        question = representatives[family_id]
        prompt = prompts['questions'].get(question)
        if prompt is None or not _exact_nonnegative_ids(prompt.get('prompt_token_ids')):
            raise ValueError(f'missing exact original prompt IDs: {question}')
        ids = prompt['prompt_token_ids']
        if prompt.get('prompt_tokens') != len(ids) or prompt.get('prompt_sha256') != digest(ids):
            raise ValueError(f'original prompt token count or hash differs: {question}')
        for seed in SEEDS:
            for arm in ARMS:
                uid = 'utility-v1|' + digest(['utility-scout-v1', source, family_id,
                                              question, seed, arm])[:32]
                rows.append({'uid': uid, 'family': family_id, 'question': question,
                             'seed': seed, 'arm': arm,
                             'prompt_tokens': len(ids),
                             'prompt_token_ids_sha256': digest(ids),
                             'max_emitted_plus_injected_tokens': MAX_TOKENS})
    if len(rows) != 384 or len({r['uid'] for r in rows}) != 384:
        raise ValueError('utility assignment coverage differs')
    body = {'schema': 'routing-utility-scout-plan-v1', 'sources': source,
            'model': 'Qwen3.6-35B-A3B-FP8', 'tokenizer_sha256': prompts['tokenizer_sha256'],
            'family_order': ordered, 'arms': list(ARMS), 'seeds': list(SEEDS),
            'maximum_tokens_per_assignment': MAX_TOKENS,
            'assignment_count': 384, 'family_count': 96,
            'maximum_decode_and_injection_tokens': 384 * MAX_TOKENS,
            'exact_prefill_tokens': sum(r['prompt_tokens'] for r in rows),
            'assignments': rows,
            'status': 'HOLD_POLICY_ENGINE_AND_COMPLETE_STAGE_PRICE',
            'interpretation': 'Original-prompt ITT assignments; native and future frozen policy; no observational native trace reused as an outcome.'}
    return seal(body)


def validate_plan(plan: dict) -> None:
    if plan.get('schema') != 'routing-utility-scout-plan-v1' or plan.get('sha256') != digest(
            {k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('utility plan seal or schema differs')
    rows = plan['assignments']
    if (len(plan['family_order']) != 96 or len(set(plan['family_order'])) != 96 or
            len(rows) != 384 or len({r['uid'] for r in rows}) != 384 or
            set(plan['arms']) != set(ARMS) or plan['seeds'] != list(SEEDS) or
            plan['maximum_tokens_per_assignment'] != MAX_TOKENS or
            plan['maximum_decode_and_injection_tokens'] != 384 * MAX_TOKENS or
            plan['exact_prefill_tokens'] != sum(r['prompt_tokens'] for r in rows)):
        raise ValueError('utility assignment or resource quantities differ')
    for family_id in plan['family_order']:
        family_rows = [r for r in rows if r['family'] == family_id]
        if (len(family_rows) != 4 or
                {(r['seed'], r['arm']) for r in family_rows} !=
                {(seed, arm) for seed in SEEDS for arm in ARMS} or
                len({r['question'] for r in family_rows}) != 1):
            raise ValueError('family ITT factorial incomplete')
        for row in family_rows:
            expected_uid = 'utility-v1|' + digest([
                'utility-scout-v1', plan['sources'], family_id, row['question'],
                row['seed'], row['arm']])[:32]
            if (row['uid'] != expected_uid or row['prompt_tokens'] < 1 or
                    row['max_emitted_plus_injected_tokens'] != MAX_TOKENS):
                raise ValueError('utility assignment UID or original-prompt metadata differs')


def bind_execution(plan: dict, policy: dict, qualification: dict) -> dict:
    """Bind a future, already selected policy; never infer one from utility data."""
    validate_plan(plan)
    for value, schema in ((policy, 'routing-frozen-utility-policy-v1'),
                          (qualification, 'routing-utility-engine-qualification-v1')):
        if value.get('schema') != schema or value.get('sha256') != digest(
                {k: v for k, v in value.items() if k != 'sha256'}):
            raise ValueError(f'missing sealed {schema}')
        if value.get('status') != 'PASS':
            raise ValueError(f'{schema} has not passed')
    if (not policy.get('mechanism_analysis_sha256') or
            not policy.get('controller_code_files') or
            not qualification.get('qualified_worker_sha256') or
            qualification.get('policy_sha256') != policy['sha256']):
        raise ValueError('policy, disjoint validation, or qualified engine binding absent')
    if any(file_sha(Path(path)) != expected for path, expected in
           policy['controller_code_files'].items()):
        raise ValueError('frozen controller source differs')
    return seal({'schema': 'routing-utility-scout-execution-v1',
                 'plan_sha256': plan['sha256'], 'policy_sha256': policy['sha256'],
                 'qualification_sha256': qualification['sha256'],
                 'assignment_count': 384, 'max_tokens': MAX_TOKENS,
                 'status': 'BOUND_UNPRICED'})


def shard_schedule(plan: dict, shards: int) -> list[list[dict]]:
    """Keep all four within-family ITT cells on one independent 2-GPU shard."""
    validate_plan(plan)
    if type(shards) is not int or not 1 <= shards <= 96:
        raise ValueError('shard count must be 1..96')
    by_family = {family: [] for family in plan['family_order']}
    for row in plan['assignments']:
        by_family[row['family']].append(row)
    result = [[] for _ in range(shards)]
    for index, family in enumerate(plan['family_order']):
        result[index % shards].extend(by_family[family])
    if sorted(r['uid'] for block in result for r in block) != sorted(
            r['uid'] for r in plan['assignments']):
        raise ValueError('shard assignment skips or duplicates a request')
    return result


def price_complete_stage(plan: dict, execution: dict | None, profile: dict | None) -> dict:
    """Conservative all-in GPU-hour interface; missing measurements hold launch."""
    validate_plan(plan)
    base = {'schema': 'routing-utility-scout-price-v1', 'plan_sha256': plan['sha256'],
            'assignment_count': 384,
            'exact_prefill_tokens': plan['exact_prefill_tokens'],
            'maximum_decode_and_injection_tokens': plan['maximum_decode_and_injection_tokens']}
    if execution is None or profile is None:
        return seal({**base, 'status': 'HOLD_POLICY_OR_MEASURED_PROFILE',
                     'projected_gpu_hours': None})
    if (execution.get('schema') != 'routing-utility-scout-execution-v1' or
            execution.get('plan_sha256') != plan['sha256'] or
            execution.get('status') != 'BOUND_UNPRICED' or
            execution.get('sha256') != digest({k: v for k, v in execution.items() if k != 'sha256'})):
        raise ValueError('execution is not bound to the exact plan')
    required = ('gpu_count', 'shards', 'model_loads', 'cold_load_seconds_per_load',
                'prefill_tokens_per_second', 'decode_tokens_per_second',
                'request_overhead_seconds', 'prefix_preparation_seconds',
                'recompute_prefill_tokens', 'retry_seconds', 'shutdown_seconds',
                'grading_gpu_hours', 'nll_gpu_hours', 'labeling_gpu_hours',
                'controller_gpu_hours',
                'available_gpu_hours', 'walltime_seconds')
    if (profile.get('schema') != 'routing-utility-measured-price-profile-v1' or
            not profile.get('measurement_job_ids') or
            not profile.get('measurement_evidence_sha256') or
            set(required) - set(profile)):
        raise ValueError('complete measured pricing profile required')
    if any(type(profile[k]) not in (int, float) or not math.isfinite(profile[k]) or
           profile[k] < 0 for k in required):
        raise ValueError('pricing quantities must be finite and nonnegative')
    if (type(profile['gpu_count']) is not int or type(profile['shards']) is not int or
            type(profile['model_loads']) is not int or
            profile['gpu_count'] < 1 or not 1 <= profile['shards'] <= 96 or
            profile['model_loads'] < profile['shards'] or
            profile['prefill_tokens_per_second'] <= 0 or
            profile['decode_tokens_per_second'] <= 0 or profile['walltime_seconds'] <= 0):
        raise ValueError('invalid GPU counts, throughput, or walltime')
    shards = shard_schedule(plan, profile['shards'])
    max_shard_prefill = max(sum(r['prompt_tokens'] for r in block) for block in shards)
    max_shard_decode = max(len(block) * MAX_TOKENS for block in shards)
    max_shard_requests = max(len(block) for block in shards)
    generation_seconds = (profile['model_loads'] * profile['cold_load_seconds_per_load'] +
                          (plan['exact_prefill_tokens'] + profile['recompute_prefill_tokens']) /
                          profile['prefill_tokens_per_second'] +
                          plan['maximum_decode_and_injection_tokens'] /
                          profile['decode_tokens_per_second'] +
                          384 * profile['request_overhead_seconds'] +
                          profile['shards'] * (profile['prefix_preparation_seconds'] +
                                               profile['shutdown_seconds']) +
                          profile['retry_seconds'])
    peak_shard_seconds = (
        math.ceil(profile['model_loads'] / profile['shards']) *
        profile['cold_load_seconds_per_load'] +
        (max_shard_prefill + profile['recompute_prefill_tokens']) /
        profile['prefill_tokens_per_second'] +
        max_shard_decode / profile['decode_tokens_per_second'] +
        max_shard_requests * profile['request_overhead_seconds'] +
        profile['prefix_preparation_seconds'] + profile['retry_seconds'] +
        profile['shutdown_seconds'])
    total = (generation_seconds * profile['gpu_count'] / 3600 +
             profile['grading_gpu_hours'] + profile['nll_gpu_hours'] +
             profile['labeling_gpu_hours'] + profile['controller_gpu_hours'])
    return seal({**base, 'execution_sha256': execution['sha256'],
                 'profile_sha256': digest(profile),
                 'aggregate_node_seconds': generation_seconds,
                 'peak_shard_wall_seconds': peak_shard_seconds,
                 'shards': profile['shards'],
                 'maximum_shard_decode_tokens': max_shard_decode,
                 'projected_gpu_hours': total,
                 'status': ('PASS_COMPLETE_STAGE' if total <= profile['available_gpu_hours']
                            and peak_shard_seconds <= profile['walltime_seconds'] else
                            'HOLD_REPRICE_OR_RESOURCE_CHANGE')})


def audit_receipts(plan: dict, execution: dict, receipts: list[dict]) -> dict:
    """Retain every assigned UID, including nonfires, caps, failures and missing."""
    validate_plan(plan)
    if (execution.get('schema') != 'routing-utility-scout-execution-v1' or
            execution.get('sha256') != digest({k: v for k, v in execution.items() if k != 'sha256'}) or
            execution.get('plan_sha256') != plan['sha256'] or
            execution.get('assignment_count') != 384 or
            execution.get('max_tokens') != MAX_TOKENS or
            not execution.get('policy_sha256')):
        raise ValueError('receipt execution belongs to another plan')
    assigned = {r['uid']: r for r in plan['assignments']}
    observed = {}
    for receipt in receipts:
        if (receipt.get('schema') != 'routing-utility-scout-receipt-v1' or
                receipt.get('sha256') != digest({k: v for k, v in receipt.items() if k != 'sha256'}) or
                receipt.get('execution_sha256') != execution['sha256']):
            raise ValueError('receipt seal or execution binding differs')
        uid = receipt.get('uid')
        if uid not in assigned or uid in observed:
            raise ValueError('foreign or duplicate utility receipt UID')
        assignment = assigned[uid]
        if (receipt.get('seed') != assignment['seed'] or receipt.get('arm') != assignment['arm'] or
                receipt.get('prompt_token_ids_sha256') != assignment['prompt_token_ids_sha256'] or
                receipt.get('policy_sha256') != (None if assignment['arm'] == 'native'
                                                 else execution['policy_sha256'])):
            raise ValueError('receipt assignment, original prompt, or policy differs')
        emitted, injected = receipt.get('emitted_token_ids'), receipt.get('injected_token_ids')
        completion, source = receipt.get('completion_token_ids'), receipt.get('token_sources')
        if (not isinstance(emitted, list) or not isinstance(injected, list) or
                not isinstance(completion, list) or not isinstance(source, list) or
                any(type(x) is not int or x < 0 for x in emitted + injected) or
                len(completion) != len(source) or
                any(type(x) is not int or x < 0 for x in completion) or
                any(x not in ('emitted', 'injected') for x in source) or
                [x for x, kind in zip(completion, source) if kind == 'emitted'] != emitted or
                [x for x, kind in zip(completion, source) if kind == 'injected'] != injected or
                (assignment['arm'] == 'native' and bool(injected)) or
                receipt.get('emitted_tokens') != len(emitted) or
                receipt.get('injected_tokens') != len(injected) or
                receipt.get('tokens_charged') != len(emitted) + len(injected) or
                receipt['tokens_charged'] > MAX_TOKENS):
            raise ValueError('emitted plus injected token accounting differs')
        if (receipt.get('finish') not in ('stop', 'length', 'error') or
                (receipt['finish'] == 'length' and receipt['tokens_charged'] != MAX_TOKENS) or
                (receipt['finish'] == 'error' and not receipt.get('error')) or
                (receipt['finish'] != 'error' and receipt.get('error'))):
            raise ValueError('unrecognized natural, length, or failure outcome')
        if receipt.get('question') != assigned[uid]['question']:
            raise ValueError('receipt question differs from assignment')
        observed[uid] = receipt
    rows = []
    for assignment in plan['assignments']:
        receipt = observed.get(assignment['uid'])
        rows.append({**assignment, 'receipt_status': 'MISSING' if receipt is None else receipt['finish'],
                     'tokens_charged': None if receipt is None else receipt['tokens_charged'],
                     'gradeable': bool(receipt and receipt['finish'] == 'stop'),
                     'error': None if receipt is None else receipt.get('error')})
    return seal({'schema': 'routing-utility-scout-itt-audit-v1',
                 'plan_sha256': plan['sha256'], 'execution_sha256': execution['sha256'],
                 'assigned_count': len(rows), 'receipt_count': len(receipts),
                 'missing_count': len(rows) - len(receipts),
                 'observed_tokens_charged': sum(r['tokens_charged'] or 0 for r in rows),
                 'rows': rows,
                 'status': 'COMPLETE_UNGRADED' if len(receipts) == len(rows) else 'INCOMPLETE_ITT'})


def blind_grading_contract(audit: dict, questions: dict) -> dict:
    """Specify strict math_verify, arm-blind J1 and conservative ITT defaults."""
    if (audit.get('schema') != 'routing-utility-scout-itt-audit-v1' or
            audit.get('sha256') != digest({k: v for k, v in audit.items() if k != 'sha256'}) or
            audit.get('status') != 'COMPLETE_UNGRADED' or
            audit.get('assigned_count') != 384):
        raise ValueError('all assigned generation receipts required before grading')
    if (questions.get('schema') != 'steer-questions-v1' or
            questions.get('sha256') != digest({k: v for k, v in questions.items() if k != 'sha256'})):
        raise ValueError('unexpected sealed question table')
    grade_rows, map_rows = [], []
    for row in audit['rows']:
        grade_uid = 'utility-grade-v1|' + digest([audit['sha256'], row['uid']])[:32]
        map_rows.append({'assignment_uid': row['uid'], 'blind_uid': grade_uid if row['gradeable'] else None,
                         'default_operational_correct': False if not row['gradeable'] else None,
                         'reason': row['receipt_status']})
        if row['gradeable']:
            question = questions['questions'].get(row['question'])
            if question is None:
                raise ValueError('gold table lacks assigned question')
            grade_rows.append({'uid': grade_uid, 'question': row['question'],
                               'dataset': question['dataset'], 'problem': question['problem'],
                               'gold': question['gold'],
                               'required_fields_from_decoder': ['content_text', 'finish_reason', 'natural_stop']})
    return seal({'schema': 'routing-utility-scout-blind-grade-contract-v1',
                 'audit_sha256': audit['sha256'], 'question_table_sha256': questions['sha256'],
                 'strict_method': 'moe_steer.score.strict_rows: math_verify plus registered dataset-specific rules',
                 'adjudication_method': 'arm-blind J1 for finished strict-rejected answers; no policy disclosure',
                 'decoder': 'Qwen3.6-35B-A3B-FP8 tokenizer bound by plan; no alternate decoder',
                 'assigned_count': len(map_rows), 'gradeable_count': len(grade_rows),
                 'map': map_rows, 'blind_rows': grade_rows,
                 'status': 'PENDING_DECODE_STRICT_AND_J1'})


def materialize_blind_bundle(contract: dict, audit: dict, receipts: list[dict], decode) -> list[dict]:
    """Return exact score.BUNDLE_FIELDS rows using only Qwen3.6-decoded completions.

    ``decode`` must be the tokenizer bound by the plan; its source hash and the
    resulting bundle hash must be written by the downstream CPU grading driver.
    Strict failures that naturally stop go to the existing arm-blind J1 path.
    """
    if (contract.get('sha256') != digest({k: v for k, v in contract.items() if k != 'sha256'}) or
            audit.get('sha256') != contract.get('audit_sha256') or
            len(receipts) != audit.get('receipt_count') or audit.get('status') != 'COMPLETE_UNGRADED'):
        raise ValueError('blind bundle does not match complete sealed generation audit')
    by_uid = {r['uid']: r for r in receipts}
    if len(by_uid) != len(receipts) or set(by_uid) != {r['uid'] for r in audit['rows']}:
        raise ValueError('blind bundle receipt coverage differs')
    map_by_uid = {r['assignment_uid']: r for r in contract['map']}
    source_by_blind = {r['uid']: r for r in contract['blind_rows']}
    if len(map_by_uid) != 384 or len(source_by_blind) != contract['gradeable_count']:
        raise ValueError('blind grade map coverage differs')
    bundle = []
    for assignment in audit['rows']:
        item = map_by_uid[assignment['uid']]
        if not assignment['gradeable']:
            if item['blind_uid'] is not None:
                raise ValueError('capped or failed assignment entered grading')
            continue
        receipt = by_uid[assignment['uid']]
        source = source_by_blind[item['blind_uid']]
        if receipt['finish'] != 'stop' or source['question'] != assignment['question']:
            raise ValueError('blind source differs from gradeable assignment')
        content_text = decode(receipt['completion_token_ids'])
        if not isinstance(content_text, str):
            raise ValueError('Qwen3.6 decoder did not return text')
        bundle.append({'uid': source['uid'], 'question': source['question'],
                       'dataset': source['dataset'], 'problem': source['problem'],
                       'gold': source['gold'],
                       'content_text': content_text,
                       'finish_reason': 'stop',
                       'cumulative_tokens': receipt['tokens_charged'],
                       'natural_stop': True})
    if len(bundle) != contract['gradeable_count']:
        raise ValueError('blind final-answer bundle incomplete')
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-out', type=Path, required=True)
    args = parser.parse_args()
    if args.plan_out.exists():
        raise FileExistsError(args.plan_out)
    family, units, prompts = sealed(FAMILY), sealed(UNITS), sealed(PROMPTS)
    plan = build_plan(family, units, prompts, prompt_file_sha256=file_sha(PROMPTS))
    validate_plan(plan)
    args.plan_out.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(args.plan_out), 'sha256': plan['sha256'],
                      'assigned': plan['assignment_count'], 'status': plan['status']}))


if __name__ == '__main__':
    main()
