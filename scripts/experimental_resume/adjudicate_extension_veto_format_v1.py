"""CPU-only format sensitivity on the fixed 134 enrolled native starts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import re
import socket

import mechanism_extension_reader_contract_v2 as contract
import prepare_extension_overnight_enrollment_v1 as enrollment
import rate_mechanism_extension_start_readers_v2 as rating
from freeze_mechanism_extension_220_v1 import digest, sealed

PLAN = contract.DOC / 'MECHANISM_EXTENSION_VETO_FORMAT_PLAN_v1.json'
DEFINITIONS = contract.DOC / 'MECHANISM_EXTENSION_VETO_FORMAT_DEFINITIONS_v1.md'
WRAPPER = Path(__file__).with_suffix('.sbatch')
ENROLLMENT_SHA = '7190b1dc34f8cc6e9aeac337dc66f993cd91ce139972112333f076887d4cb5ee'
STAGE_SHA = 'e893adf5bc961f1edf93ca4677c770b156e5c22984a88f2a5809451ad89b2d8d'
FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```', re.DOTALL)
require = contract.require


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def parse_final(text):
    """Accept a sole bare or exactly once fenced final boolean object."""
    if not isinstance(text, str):
        return None, 'not_text'
    final = text.rsplit('</think>', 1)[-1].strip()
    match = FENCE.fullmatch(final)
    form = 'single_fenced_json' if match else 'bare_json'
    payload = match.group('body').strip() if match else final
    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None, 'invalid_final_format'
    if not isinstance(value, dict) or set(value) != {'already_completed'} or type(value['already_completed']) is not bool:
        return None, 'invalid_boolean_schema'
    return value['already_completed'], form


def adjudicate_vote(vote):
    parsed, form = parse_final(vote['raw_completion'])
    original = vote['already_completed']
    require(original is None or type(original) is bool, 'original veto vote is malformed')
    require(original == contract.veto.parse_veto(vote['raw_completion']), 'original parser result differs from saved raw text')
    original_valid = original is not None and vote['finish_reason'] == 'stop'
    valid = parsed is not None and vote['finish_reason'] == 'stop'
    # A narrower duplicate-key rejection cannot silently remove an old valid vote.
    require(not original_valid or (valid and parsed == original), 'new formatting rule contradicts an original valid rating')
    return {'original_value': original, 'original_valid': original_valid,
            'adjudicated_value': parsed if valid else None, 'adjudicated_valid': valid,
            'format': form, 'finish_reason': vote['finish_reason'],
            'generated_tokens': vote['generated_tokens'],
            'raw_completion_sha256': digest(vote['raw_completion']),
            'format_recovered': valid and not original_valid,
            'residual_reason': None if valid else 'length' if vote['finish_reason'] == 'length' else form}


def code_files():
    paths = [Path(__file__), WRAPPER, Path(contract.__file__), Path(contract.veto.__file__),
             Path(rating.__file__), Path(enrollment.__file__), Path(rating.sealer.__file__),
             Path(contract.extension.__file__)]
    return {str(p): contract.file_sha(p) for p in paths}


def plan_body():
    frozen = sealed(enrollment.OUT)
    stage = sealed(contract.OUTPUT_ROOT / 'STAGE_COMPLETION.json')
    require(frozen['sha256'] == ENROLLMENT_SHA and stage['sha256'] == STAGE_SHA and
            frozen['reader_stage_completion_sha256'] == stage['sha256'] and
            frozen['accepted_start_count'] == len(frozen['rows']) == 134,
            'fixed native enrollment or reader completion differs')
    return {'schema': 'extension-veto-format-plan-v1', 'enrollment_path': str(enrollment.OUT),
            'enrollment_sha256': frozen['sha256'], 'reader_stage_completion_sha256': stage['sha256'],
            'code_files': code_files(), 'definitions_path': str(DEFINITIONS),
            'definitions_sha256': contract.file_sha(DEFINITIONS),
            'population': 'The same 134 chosen native starts in frozen enrollment order; both veto readers, 268 ratings. No additional starts or generation outcomes.',
            'original_flags': 'Preserve original primary enrollment and original strict-veto sensitivity flags byte-for-byte; new sensitivity columns only.',
            'prior_inspection': 'Six naturally stopped invalid raw tails exposed Markdown-fenced final objects. The narrow formatting rule is frozen before adjudicated population counts or values are calculated.',
            'residual_scope': 'All fixed enrolled veto ratings still invalid after the format rule, regardless of boolean value, family, transition, or generated outcomes.',
            'GPU_replay_gate': 'Pricing only. No GPU replay is submitted or executed by this CPU driver. Changed cap requires separate exact-profile live qualification and immutable recovery driver.',
            'resources': {'CPU_cores': 2, 'wall_seconds': 1800, 'memory_GiB': 8, 'CPU_core_hour_ceiling': 1., 'GPU_hours': 0.}}


def price_option(residuals, cap, source, previous):
    """Conservative complete proposal, with qualification and recovery overhead."""
    if not residuals:
        return {'cap': cap, 'ratings': 0, 'complete_GPU_h': 0., 'qualification_ratings': 0,
                'status': 'NO_REPLAY_NEEDED'}
    require(cap in (2048, 4096), 'unfrozen replay cap')
    require(all(r['prompt_tokens'] + cap <= contract.PROFILE['max_model_len'] for r in residuals), 'replay exceeds model context')
    prefill_rate = source['bounded_prefill_tokens_per_second']
    decode_rate = source['bounded_decode_tokens_per_second']
    retry = source['retry_factor']
    load = previous['components_seconds']['two_cold_loads'] / 2
    shutdown = previous['components_seconds']['two_shutdowns'] / 2
    reserve, wall_limit = 900, 7200
    def work(rows):
        return retry * (sum(r['prompt_tokens'] for r in rows) / prefill_rate + len(rows) * cap / decode_rate)
    shards, current, seconds = [], [], 0.
    for start in range(0, len(residuals), 16):
        block = residuals[start:start+16]
        cost = work(block)
        require(cost + load + shutdown + reserve <= wall_limit, 'one intact residual batch exceeds wall limit')
        if current and seconds + cost + load + shutdown + reserve > wall_limit:
            shards.append({'rating_keys': [r['key'] for r in current], 'work_seconds': seconds})
            current, seconds = [], 0.
        current.extend(block); seconds += cost
    shards.append({'rating_keys': [r['key'] for r in current], 'work_seconds': seconds})
    for shard in shards:
        shard['complete_wall_seconds'] = shard['work_seconds'] + load + shutdown + reserve
    qualification = []
    for transition in contract.extension.SUPPORTED:
        first = next((r for r in residuals if r['transition'] == transition), None)
        if first is not None:
            qualification.append(first)
    if residuals[-1] not in qualification:
        qualification.append(residuals[-1])
    qual_wall = load + shutdown + reserve + work(qualification)
    require(qual_wall <= 3600, 'qualification exceeds its one-hour wall limit')
    recovery_loads = max(1, math.ceil(.25 * len(shards)))
    recovery_overhead = recovery_loads * (load + shutdown + reserve)
    production_wall = sum(s['complete_wall_seconds'] for s in shards) + recovery_overhead
    return {'status': 'HOLD_NEW_GPU_DRIVER_AND_LIVE_CAP_QUALIFICATION', 'cap': cap,
            'ratings': len(residuals), 'prompt_tokens_exact': sum(r['prompt_tokens'] for r in residuals),
            'max_decode_tokens': len(residuals) * cap, 'bounded_prefill_tokens_per_second': prefill_rate,
            'bounded_decode_tokens_per_second': decode_rate, 'retry_factor': retry,
            'cold_load_seconds_each': load, 'shutdown_seconds_each': shutdown,
            'per_job_overhead_reserve_seconds': reserve, 'recovery_loads_and_shutdowns': recovery_loads,
            'max_production_wall_seconds': wall_limit, 'shards': shards,
            'qualification_rating_keys': [r['key'] for r in qualification],
            'qualification_ratings': len(qualification), 'qualification_wall_seconds': qual_wall,
            'qualification_GPU_h': 2 * qual_wall / 3600,
            'production_and_recovery_GPU_h': 2 * production_wall / 3600,
            'complete_GPU_h': 2 * (production_wall + qual_wall) / 3600,
            'price_reference_sha256': source['sha256'], 'overhead_reference_sha256': previous['sha256'],
            'accounting': 'Both GPUs, exact original prompts, cap decoding, 1.25 work factor, every planned load/shutdown, 900 seconds overhead per job, recovery loads, and a separate <=3-rating qualification counted. Qualification is not deducted from production.'}


def run():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
            getpass.getuser() == 'lmolfett' and not socket.gethostname().startswith('login'),
            'format sensitivity requires an authorized CPU Slurm step')
    plan = sealed(PLAN)
    expected = plan_body()
    require(plan['binding'] == expected, 'frozen veto format plan changed')
    frozen = sealed(enrollment.OUT)
    family, selection, frame, source, _, _, price = rating.expected_final()
    completed, stage = enrollment.completed_reader_rows(frame, price)
    require(stage['sha256'] == STAGE_SHA, 'reader stage changed during sensitivity')
    by_uid = {r['uid']: r for r in completed}
    source_rows = {r['uid']: r for r in frame['records']}
    cpu = sealed(contract.CPU_RESULT)
    prompt = {r['uid']: r for r in cpu['records'] if r['kind'] == 'veto'}
    counts = Counter(starts=len(frozen['rows']), ratings=2*len(frozen['rows']))
    records, residuals = [], []
    transition_counts = {t: Counter() for t in contract.extension.SUPPORTED}
    for chosen in frozen['rows']:
        uid = chosen['uid']
        raw = by_uid[uid]
        require(raw['transition'] == chosen['transition'] and enrollment.primary_accepted(raw), 'chosen native start binding differs')
        original_flag = enrollment.strict_sensitivity(raw)
        require(original_flag == chosen['strict_veto_sensitivity_eligible'], 'original frozen veto flag differs')
        votes = []
        for reader, vote in enumerate(raw['veto_readers']):
            value = adjudicate_vote(vote)
            value.update(reader=reader, seed=contract.rating_seed(uid, 'veto', reader))
            votes.append(value)
            kind = 'original_valid' if value['original_valid'] else 'format_recovered' if value['adjudicated_valid'] else 'residual_invalid'
            counts[kind] += 1
            transition_counts[chosen['transition']][kind] += 1
            if not value['adjudicated_valid']:
                residuals.append({'key': uid + '-veto-reader' + str(reader), 'uid': uid,
                    'reader': reader, 'family': chosen['family'], 'transition': chosen['transition'],
                    'seed': value['seed'], 'reason': value['residual_reason'],
                    'original_generated_tokens': value['generated_tokens'],
                    'raw_completion_sha256': value['raw_completion_sha256'],
                    'messages_sha256': digest(contract.messages(source_rows[uid], 'veto')),
                    'prompt_tokens': prompt[uid]['prompt_tokens'], 'prompt_ids_sha256': prompt[uid]['prompt_ids_sha256']})
        valid_pair = all(v['adjudicated_valid'] for v in votes)
        strict = valid_pair and all(v['adjudicated_value'] is False for v in votes)
        counts['original_strict_starts'] += original_flag
        counts['format_strict_starts'] += strict
        counts['format_complete_veto_pairs'] += valid_pair
        transition_counts[chosen['transition']]['starts'] += 1
        transition_counts[chosen['transition']]['format_strict_starts'] += strict
        records.append({'uid': uid, 'family': chosen['family'], 'transition': chosen['transition'],
            'original_strict_veto_sensitivity_eligible': original_flag,
            'format_adjudicated_strict_veto_sensitivity_eligible': strict,
            'format_adjudicated_veto_pair_complete': valid_pair, 'veto_readers': votes})
    counts['families'] = len({r['family'] for r in records})
    counts['format_strict_families'] = len({r['family'] for r in records if r['format_adjudicated_strict_veto_sensitivity_eligible']})
    counts['residual_starts'] = len({r['uid'] for r in residuals})
    counts['residual_families'] = len({r['family'] for r in residuals})
    require([r['uid'] for r in records] == [r['uid'] for r in frozen['rows']], 'sensitivity enrollment order changed')
    previous = sealed(contract.extension.PRIOR_PRICE)
    out = contract.ROOT / ('veto-format-v1-' + plan['sha256'][:16])
    out.mkdir(parents=True, exist_ok=True)
    result = contract.save(out / 'RESULT.json', {'schema': 'extension-veto-format-result-v1',
        'status': 'COMPLETE_CPU_FORMAT_SENSITIVITY', 'plan_sha256': plan['sha256'],
        'enrollment_sha256': frozen['sha256'], 'reader_stage_completion_sha256': stage['sha256'],
        'source_frame_sha256': frame['sha256'], 'source_reader_price_sha256': source['sha256'],
        'cpu_prompt_preflight_sha256': cpu['sha256'], 'counts': dict(counts),
        'counts_by_transition': {t: dict(c) for t, c in transition_counts.items()},
        'residual_reasons': dict(Counter(r['reason'] for r in residuals)), 'records': records,
        'residual_replay_assignments': residuals, 'replay_price_options': [price_option(residuals, cap, source, previous) for cap in (2048,4096)],
        'job_id': os.environ['SLURM_JOB_ID'],
        'scope': 'Post-format sensitivity only, preserving primary enrollment and original strict flags. No generated continuation or semantic routing outcome is used.'}, existing_ok=True)
    print(json.dumps({'result': str(out/'RESULT.json'), 'sha256': result['sha256'], 'counts': dict(counts),
                      'residual_reasons': result['residual_reasons'],
                      'price_options': [{k:p[k] for k in ('cap','ratings','complete_GPU_h')} for p in result['replay_price_options']]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze-plan', 'analyze'))
    args = parser.parse_args()
    if args.mode == 'freeze-plan':
        value = contract.save(PLAN, {'schema': 'extension-veto-format-plan-envelope-v1',
            'frozen_utc': datetime.now(timezone.utc).isoformat(), 'binding': plan_body()})
        print(value['sha256'])
    else:
        run()


if __name__ == '__main__':
    main()
