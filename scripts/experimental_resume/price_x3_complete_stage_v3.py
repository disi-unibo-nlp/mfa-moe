"""Seal the measured, conservative all-component legacy X3 resource amendment.

This prepares a price only. New routing-action study priority remains an explicit
launch hold. Every production stage is stopped and repriced if its measured work
exceeds the bound used here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPORT = REPO / 'report/experimental-resume-v1'
TIMING = S / 'runs/resume-v1/x3-timing-pilot-v1/results-8f7c67d6b2a06e42-v4/shard-0.status.json'
NLL = S / ('runs/x2-resume-v1/native-nll/'
           '2f10f3dbaefd0839c3ec16d0768581e0863c26ee18e70911de40f1643b4feebe/'
           'ec9661af160f42d145a09756925d2bc26a3f36360a25be772a41730a0e03c515/native-nll.json')
J1 = S / 'runs/x1/score/verdicts.jsonl'
HIST_LABEL_ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/labels/qwen330b')
PROGRAM = REPO / 'results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json'
AMEND_OUT = REPORT / 'X3_RESOURCE_AMENDMENT_v1.json'
PRICE_OUT = REPORT / 'X3_COMPLETE_STAGE_PRICE_v3.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing sealed X3 version differs: {path}')
    else:
        path.write_text(json.dumps(value, indent=1) + '\n')
    return value


def main():
    inventory = sealed(REPORT / 'X3_COST_INVENTORY_v1.json')
    eligible = sealed(REPORT / 'X3_ELIGIBILITY_v1.json')
    selection = sealed(REPORT / 'X3_G3_SELECTION_v2.json')
    density = sealed(REPORT / 'X3_LABEL_DENSITY_PROJECTION_v1.json')
    timing = json.loads(TIMING.read_text())
    if (timing['status'] != 'complete' or timing['n_requests'] != 40 or
        timing['n_done'] != 40 or timing['tokens_this_run'] != 971219 or
        timing['timing']['tokens_per_second'] != 838.61 or
        timing['engine_build_seconds'] != 707.256):
        raise ValueError('X3 completed 32k timing pilot differs')
    if (inventory['eligibility_sha256'] != eligible['sha256'] or
        inventory['registered_selection_sha256'] != selection['sha256'] or
        density['eligibility_sha256'] != eligible['sha256'] or
        density['native_sources'] != 77 or density['two_seed_five_arm_proxy_units'] != 25380):
        raise ValueError('X3 frozen cost/enrollment/label density differs')
    two = next(x for x in inventory['scenarios'] if x['signs'] == 2)
    if (two['requests'], two['maximum_decode_tokens'], two['maximum_prefill_tokens_without_cache_reuse']) != (770, 14049070, 4132590):
        raise ValueError('X3 registered maximum workload differs')
    summaries = [HIST_LABEL_ROOT / f'part-{i:02}/summary.json' for i in range(4)]
    counts = [json.loads(p.read_text())['completed'] for p in summaries]
    if counts != [28290, 28284, 28281, 28280] or sha(PROGRAM) != '467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07':
        raise ValueError('historical exact GEPA program or unit counts differ')
    if sum(1 for _ in J1.open()) != 228 or len(json.loads(NLL.read_text())['records']) != 3354:
        raise ValueError('historical J1 or native-NLL source count differs')
    # Slurm job durations and final states were verified by exact job ID before
    # sealing. Include failed GEPA service and the recovered part in gross rate.
    label_jobs = [
        {'id': 58728654, 'elapsed_seconds': 35624, 'state': 'COMPLETED'},
        {'id': 58728656, 'elapsed_seconds': 22803, 'state': 'COMPLETED'},
        {'id': 58728657, 'elapsed_seconds': 5410, 'state': 'FAILED'},
        {'id': 58728658, 'elapsed_seconds': 22324, 'state': 'COMPLETED'},
        {'id': 58811162, 'elapsed_seconds': 19495, 'state': 'COMPLETED'},
    ]
    gross_label_gpu_h = sum(2 * x['elapsed_seconds']/3600 for x in label_jobs)
    slow_label_gpu_h_per_unit = (2*35624/3600) / counts[0]
    prior_gpu_h = 2 * (972 + 2168) / 3600  # jobs 59185118 and 59194916
    amendment = write_once(AMEND_OUT, {
        'schema': 'legacy-X3-resource-amendment-v1', 'status': 'SEALED',
        'original_ceiling_GPU_h': 12.0, 'ceiling_total_GPU_h': 65.0,
        'already_spent_X3_GPU_h': prior_gpu_h,
        'prior_jobs': [{'id': 59185118, 'elapsed_seconds': 972, 'gpus': 2, 'status': 'COMPLETED'},
                       {'id': 59194916, 'elapsed_seconds': 2168, 'gpus': 2, 'status': 'COMPLETED'}],
        'authorization': 'User authorized all needed GPU hours and long jobs without a fixed cap in this session; the legacy X3 amendment is versioned separately from the new steering study.',
        'population': 'frozen 96 dev-disc questions, 77 fires, 19 retained nonfires, two seeds, N/E-/M-/E+/M+; 770 new requests',
        'exclusions': 'E4 and dev-spare escalation require their own fully priced additions and execution qualifications',
        'priority': 'new routing-action feasibility/discovery retains launch priority; builder keeps this hold',
        'source_sha256': {'inventory': inventory['sha256'], 'eligibility': eligible['sha256'],
                          'selection': selection['sha256'], 'native_label_density': density['sha256'],
                          'timing_status_file': sha(TIMING)}})
    # Max-token generation with a 25% measured-throughput derate. Prefill uses
    # 4,914 tokens/s, four times slower than historical Q10's rate. GEPA is
    # priced at the slowest complete same-program part and allows 50% more units
    # than the native-window proxy. Grading/audit uses the verified 228-item J1
    # charge plus 50% margin. NLL's 2h bound is >8x the X2 position-proportional
    # estimate and is rechecked on the actual X3 first-256-token roster.
    components = {
        'generation': two['maximum_decode_tokens']/(838.61*.75)*2/3600,
        'prefill': two['maximum_prefill_tokens_without_cache_reuse']/4914*2/3600,
        'labeling': density['two_seed_five_arm_proxy_units']*1.5*slow_label_gpu_h_per_unit,
        'grading_and_audit': (770 + 600)/228 * (2*1552/3600) * 1.5,
        'native_nll': 2.0,
        'cold_loads': 2*timing['engine_build_seconds']*2/3600,
        'retries_and_shutdown': 3.0,
    }
    if any(not math.isfinite(x) or x < 0 for x in components.values()):
        raise ValueError('invalid X3 cost component')
    all_in = sum(components.values())
    if not prior_gpu_h + all_in < amendment['ceiling_total_GPU_h']:
        raise ValueError('complete X3 cost fails amended ceiling')
    sources = {str(p): sha(p) for p in [TIMING, NLL, J1, PROGRAM, *summaries,
             REPORT / 'X3_COST_INVENTORY_v1.json', REPORT / 'X3_LABEL_DENSITY_PROJECTION_v1.json']}
    price = write_once(PRICE_OUT, {
        'schema': 'legacy-X3-complete-stage-price-v3', 'status': 'PASS_COMPLETE_STAGE',
        'launch_status': 'HOLD_NEW_STUDY_PRIORITY', 'stage_scope': 'registered-two-sign-N-E1-M1',
        'request_count': 770, 'assigned_questions': 96, 'firing_questions': 77,
        'retained_nonfire_questions': 19,
        'eligibility_sha256': eligible['sha256'],
        'registered_selection_sha256': selection['sha256'],
        'resource_amendment_sha256': amendment['sha256'],
        'approved_ceiling_total_GPU_h': amendment['ceiling_total_GPU_h'],
        'already_spent_X3_GPU_h': prior_gpu_h,
        'components_GPU_h': components, 'all_in_GPU_h': all_in,
        'remaining_margin_GPU_h': amendment['ceiling_total_GPU_h'] - prior_gpu_h - all_in,
        'maximum_decode_tokens': two['maximum_decode_tokens'],
        'maximum_prefill_tokens': two['maximum_prefill_tokens_without_cache_reuse'],
        'measured_pilot_decode_tokens_per_second': 838.61,
        'generation_throughput_derate': .75,
        'prefill_stress_tokens_per_second': 4914,
        'native_label_proxy_units': density['two_seed_five_arm_proxy_units'],
        'maximum_priced_label_units': math.ceil(density['two_seed_five_arm_proxy_units']*1.5),
        'historical_same_program_label_jobs': label_jobs,
        'historical_same_program_label_units': sum(counts),
        'historical_same_program_gross_GPU_h': gross_label_gpu_h,
        'historical_slowest_complete_GPU_h_per_unit': slow_label_gpu_h_per_unit,
        'J1_historical_items': 228, 'J1_historical_GPU_h': 2*1552/3600,
        'priced_J1_and_audit_items': 1370,
        'complete_generation_source_sha256': sha(TIMING),
        'labeling_source_sha256': digest({str(p): sha(p) for p in [PROGRAM, *summaries]}),
        'grading_source_sha256': sha(J1),
        'native_nll_source_sha256': sha(NLL),
        'native_nll_scope': 'price 770 first-256-token native teacher-forced sequences as conservative upper bound; actual registered N/E roster and long-prefix parity are checked before submission',
        'retry_rule': 'stop and reprice before any stage if actual work exceeds its bound, label units exceed 38,070, parity fails, or prior plus projected spend exceeds 65 GPU-hours; no silent retry',
        'new_study_priority_resolved': False,
        'input_file_sha256': sources,
        'scientific_limit': 'X3 remains a registered diagnostic causal screen; same-program historical GEPA throughput and native sentence density are resource proxies, not semantic outcomes.'})
    print(json.dumps({'amendment': str(AMEND_OUT), 'amendment_sha256': amendment['sha256'],
                      'price': str(PRICE_OUT), 'price_sha256': price['sha256'],
                      'all_in_GPU_h': all_in, 'prior_gpu_h': prior_gpu_h,
                      'total_ceiling_GPU_h': amendment['ceiling_total_GPU_h'],
                      'launch': price['launch_status']}))


if __name__ == '__main__':
    main()
