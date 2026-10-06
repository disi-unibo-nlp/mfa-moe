"""Freeze the two-reader, family-disjoint discovery feasibility enrollment.

This is a smaller exploratory amendment to the registered 48-family maximum,
not a completed discovery stage, mechanism validation, or utility result.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
POOL = ROOT / 'runs/routing-control-v1/dense-discovery/FULLPREFIX_READER_AGREED_ELIGIBLE_POOL_v0.json'
DICTIONARY = REPORT / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
GPT = REPORT / 'GPTOSS_FULLPREFIX_START_PANEL_AGREEMENT_v2.json'
OUT = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_MICRO_AMENDMENT_v1.json'
PRICE = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_SERIAL_PRICE_v1.json'


def seal_once(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('existing sealed artifact differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1) + '\n')
    return value


def main():
    pool, dictionary, gpt = (base.sealed(p) for p in (POOL, DICTIONARY, GPT))
    families = base.sealed(base.FAMILY_FREEZE)['new_parent_pools']['parent_pools']
    discovery = set(families['discovery'])
    gpt_by_uid = {r['uid']: r for r in gpt['records']}
    selected, seen = [], set()
    skipped = []
    for transition in ('candidate_to_verify', 'approach_to_commit'):
        for row in pool['records']:
            if row['transition'] != transition:
                continue
            if row['qwen_fullprefix_reader_status'] != 'both_natural_stop_start_true':
                raise ValueError('selected source row lacks frozen two-reader support')
            if row['family'] not in discovery:
                raise ValueError('validation or utility family entered discovery')
            if row['family'] in seen:
                skipped.append({'uid': row['uid'], 'transition': transition,
                                'family': row['family'], 'reason': 'cross-transition family overlap'})
                continue
            if row['uid'] not in gpt_by_uid:
                raise ValueError('selected row missing independent model diagnostic')
            seen.add(row['family'])
            selected.append({**row, 'gpt_start_diagnostic': gpt_by_uid[row['uid']]['gpt_start'],
                             'gpt_completion_sha256': gpt_by_uid[row['uid']]['gpt_raw_completion_sha256']})
    counts = dict(Counter(r['transition'] for r in selected))
    if counts != {'candidate_to_verify': 12, 'approach_to_commit': 9} or len(seen) != 21:
        raise ValueError('prospective candidate-priority global enrollment differs')
    random_schedule = {}
    for transition in ('candidate_to_verify', 'approach_to_commit'):
        rows = [r for r in selected if r['transition'] == transition]
        random_schedule[transition] = {r['family']: [i % 4, (i + 1) % 4]
                                       for i, r in enumerate(rows)}
    if any(pair[0] == pair[1] for groups in random_schedule.values()
           for pair in groups.values()):
        raise ValueError('family received same random set in both seeds')
    for groups in random_schedule.values():
        for seed in (0, 1):
            uses = [sum(pair[seed] == k for pair in groups.values()) for k in range(4)]
            if max(uses) - min(uses) > 1:
                raise ValueError('four random sets not balanced within seed')
    body = {
        'schema': 'routing-discovery-feasibility-micro-amendment-v1',
        'status': 'FROZEN_PREINTERVENTION_ENROLLMENT_PENDING_ENGINE_AND_MEASUREMENT_QUALIFICATION',
        'source_pool_sha256': pool['sha256'],
        'action_dictionary_sha256': dictionary['sha256'],
        'gpt_measurement_diagnostic_sha256': gpt['sha256'],
        'family_freeze_sha256': base.sealed(base.FAMILY_FREEZE)['sha256'],
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'registered_discovery_maximum_families': 48,
        'selected_discovery_families': 21,
        'shortfall_from_registered_maximum': 27,
        'selected_by_transition': counts,
        'selection_rule': 'Frozen v2.2 detector fire; both full-prefix Qwen readers natural-stop start=true; first fixed-hash native prefix per discovery family from sealed source pool; candidate transition priority resolves cross-transition family overlap. Native and GPT opinions are recorded strata, never enrollment filters. No intervention or future completion read.',
        'cross_transition_overlap_omissions': skipped,
        'failed_check_transition': 'Omitted: only one two-Qwen-accepted discovery family in selected pool; unsupported for causal screen',
        'rows': selected,
        'random_set_by_transition_family_seed': random_schedule,
        'maximum_arms_per_transition': 6,
        'arms': ['native', 'native_duplicate', 'target_bias0.5', 'target_bias1',
                 'matched_random_bias0.5', 'matched_random_bias1'],
        'seeds': [0, 1], 'max_tokens': 256,
        'maximum_requests': 21 * 6 * 2,
        'maximum_decode_tokens': 21 * 6 * 2 * 256,
        'prefill_tokens': sum((len(r['prompt_ids']) + len(r['prefix_ids'])) * 12
                              for r in selected),
        'maximum_context_tokens': max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256
                                      for r in selected),
        'precision_limit': 'At 12 and 9 independent families, a ±1 family-level paired difference has worst-case normal 95% half-width about 0.57 and 0.65 (before multiplicity); these are rough scale indicators, not power guarantees. Family-clustered simultaneous intervals may be wider. No narrow null or noninferiority conclusion follows from this feasibility screen.',
        'claim_limit': 'Discovery-family exploratory first-stage and semantic estimates only. This does not complete the 48-family registered discovery maximum or touch independent mechanism-validation 128-family and utility 96-family parent pools; no validation, utility or universal optimal trajectory claim.',
        'measurement_limit': 'Two Qwen readers share a model; native Qwen3.6 and GPT-OSS diagnostics disagree substantially, and GPT accepted frozen Qwen-negative controls. Report these strata and rating uncertainty, not human ground truth or independent semantic confirmation.',
        'release_gates': ['qualified native/edited neighbor isolation profile',
                          'versioned six-arm two-transition causal driver and full-stage price',
                          'arm-blind full-prefix target rubric and pre-treatment start record',
                          'held-out validation/utility kept separate'],
    }
    amendment = seal_once(OUT, body)
    # Conservative fallback using observed 8–12 aggregate tok/s for serial/eager
    # 256-token batches, with 8 tok/s stress and 1,000 prefill tok/s stress.
    cold_load, shutdown, factor = 685.0234088897705, 196., 1.25
    serial_seconds = cold_load + shutdown + factor * (
        body['prefill_tokens'] / 1000. + body['maximum_decode_tokens'] / 8.)
    price_body = {
        'schema': 'routing-discovery-feasibility-serial-price-v1',
        'amendment_sha256': amendment['sha256'],
        'source_serial_qualification_job': '59200002',
        'source_serial_qualification_elapsed_seconds': 1993,
        'source_serial_batch_decode_tokens': 1536,
        'source_serial_batch_seconds_range': [132.20585751533508, 136.98548364639282],
        'serial_fallback_max_num_seqs': 1,
        'expected_prefill_tokens': body['prefill_tokens'],
        'maximum_decode_tokens': body['maximum_decode_tokens'],
        'stress_decode_tokens_per_second': 8,
        'stress_prefill_tokens_per_second': 1000,
        'repeat_factor': factor, 'cold_load_seconds': cold_load,
        'shutdown_seconds': shutdown,
        'estimated_complete_stage_wall_seconds': serial_seconds,
        'estimated_complete_stage_gpu_hours': serial_seconds * 2 / 3600,
        'provisional_stage_wall_ceiling_seconds': 18000,
        'provisional_stage_gpu_hour_ceiling': 10,
        'status': 'Price-only fallback. Exact execution and complete-stage ceiling require final generic driver, context stress validation, and qualified neighbor profile. No GPU submission authorized by this artifact alone.',
    }
    price = seal_once(PRICE, price_body)
    print(json.dumps({'amendment': str(OUT), 'sha256': amendment['sha256'],
                      'counts': counts, 'requests': body['maximum_requests'],
                      'prefill_tokens': body['prefill_tokens'],
                      'price': str(PRICE), 'price_sha256': price['sha256'],
                      'serial_estimated_seconds': serial_seconds}))


if __name__ == '__main__':
    main()
