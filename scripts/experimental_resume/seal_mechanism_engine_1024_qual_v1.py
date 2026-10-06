"""Seal exact 12-request 1,024-token serial/eager engineering qualification and price."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import qualify_mechanism_engine_1024_v1 as qual
import run_boundary_micro_screen as base

REPO = qual.REPO
STAGE = qual.STAGE
DOC = REPO / 'report/experimental-resume-v1'
FIXTURE = STAGE / 'runs/ordered-qualification-v1/CPU_PREFIXES.json'
FAMILY = DOC / 'family-freeze.json'
DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
ORDERED = STAGE / 'runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json'
SERIAL = DOC / 'CAUSAL_MICRO_SERIAL_QUAL_AUDIT_v1.json'
H14 = STAGE / 'qualification/h14/resume-v1-recovery/H14.json'
WORKER_PREP = STAGE / 'runs/ordered-qualification-v1/PREPARED.cpu-recovery-v2.json'
BASE_DRIVER = REPO / 'scripts/experimental_resume/run_boundary_micro_screen.py'
MANIFEST = DOC / 'MECHANISM_ENGINE_1024_QUAL_MANIFEST_v1.json'
PRICE = DOC / 'MECHANISM_ENGINE_1024_QUAL_PRICE_v1.json'


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('existing sealed qualification artifact differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--price', type=Path, default=PRICE)
    args = parser.parse_args()
    fixture, family, dictionary = (base.sealed(path) for path in (FIXTURE, FAMILY, DICTIONARY))
    ordered, serial, worker_prep = (base.sealed(path) for path in (ORDERED, SERIAL, WORKER_PREP))
    h14 = json.loads(H14.read_text())
    if (not ordered['pass'] or not h14['pass'] or
            not all(h14['evidence']['holes'][x]['pass'] for x in ('H1', 'H2', 'H3', 'H4')) or
            fixture['family_freeze_sha256'] != family['sha256'] or
            len(fixture['rows']) != 4 or
            worker_prep['sha256'] != ordered['worker_code_digest'] or
            Path(worker_prep['overlay']) != Path(qual.STAGE / 'addenda/ordered/9727c10299b71e7a/moe_exp_src')):
        raise ValueError('inherited qualification or immutable fixture differs')
    files = {**worker_prep['files'], str(Path(qual.__file__)): base.file_sha(qual.__file__),
             str(BASE_DRIVER): base.file_sha(BASE_DRIVER)}
    if any(base.file_sha(path) != expected for path, expected in files.items()):
        raise ValueError('qualified source changed before sealing')
    fixtures = [{'family': fixture['rows'][0]['family'],
                 'question': fixture['rows'][0]['question'], 'prefix_tokens': 128},
                {'family': fixture['rows'][1]['family'],
                 'question': fixture['rows'][1]['question'], 'prefix_tokens': 2048}]
    expected_prefill = (4 * (len(fixture['rows'][0]['prompt_ids']) + 128) +
                        7 * (len(fixture['rows'][1]['prompt_ids']) + 2048) +
                        len(fixture['rows'][0]['prompt_ids']) + 129)
    body = {'schema': 'routing-mechanism-serial-eager-1024-qual-manifest-v1',
            'source_fixture': {'path': str(FIXTURE), 'sha256': fixture['sha256']},
            'source_family_freeze': {'path': str(FAMILY), 'sha256': family['sha256']},
            'source_action_dictionary': {'path': str(DICTIONARY), 'sha256': dictionary['sha256']},
            'source_ordered_qualification': {'path': str(ORDERED), 'sha256': ordered['sha256']},
            'source_serial_qualification': {'path': str(SERIAL), 'sha256': serial['sha256']},
            'source_h14': {'path': str(H14), 'file_sha256': base.file_sha(H14)},
            'qualification_driver_sha256': base.file_sha(qual.__file__),
            'qualified_worker_sha256': ordered['sha256'],
            'worker_code_digest': ordered['worker_code_digest'],
            'base_tree_sha256': base.REQUIRED_BASE_TREE,
            'overlay': str(worker_prep['overlay']), 'code_files': files,
            'engine_profile': qual.PROFILE, 'max_tokens': 1024,
            'pulse_slots': [0, 512], 'pulse_length': 256, 'bias': 1.0,
            'seeds': [0], 'fixtures': fixtures,
            'four_arm_order': ['native', 'target', 'random', 'native_duplicate'],
            'expected_requests': 12, 'expected_prefill_tokens': expected_prefill,
            'maximum_decode_tokens': 11 * 1024 + 128,
            'maximum_context_tokens': max(len(fixture['rows'][0]['prompt_ids']) + 129 + 128,
                                          len(fixture['rows'][1]['prompt_ids']) + 2048 + 1024),
            'mechanical_tests': ['short/long same-prefix four-arm sequential isolation',
                                 'both supported target and matched-random +1 actions',
                                 'ordered and reversed 0/512 pulses at 256 tokens',
                                 'closed-reasoning prefix',
                                 'deliberate reset_prefix_cache and recompute'],
            'interpretation': 'engineering fixtures only; no semantic effect or engine equivalence'}
    manifest = {**body, 'sha256': base.digest(body)}
    qual.validate_manifest(manifest)
    # Conservative lower-tail serial stress rate observed in discovery operations:
    # price the *maximum* 11,392 generated tokens rather than expected stops.
    cold_load = math.ceil(ordered['cold_build_seconds'])
    decode_rate = 8.0
    decode = math.ceil(manifest['maximum_decode_tokens'] / decode_rate)
    prefill_and_preparation = 120
    recovery_and_audit = 240
    shutdown = 196
    modeled = cold_load + decode + prefill_and_preparation + recovery_and_audit + shutdown
    walltime = 3600
    price_body = {'schema': 'routing-mechanism-serial-eager-1024-qual-price-v1',
                  'qualification_manifest_sha256': manifest['sha256'],
                  'reference_ordered_job_id': ordered['job_id'],
                  'reference_ordered_qualification_sha256': ordered['sha256'],
                  'reference_serial_audit_sha256': serial['sha256'],
                  'gpu_count': 2, 'requested_wall_seconds': walltime,
                  'cold_load_seconds': cold_load,
                  'decode_rate_tokens_per_second_conservative': decode_rate,
                  'maximum_decode_tokens': manifest['maximum_decode_tokens'],
                  'maximum_decode_seconds': decode,
                  'prefill_and_prefix_preparation_allowance_seconds': prefill_and_preparation,
                  'recompute_recovery_and_telemetry_audit_allowance_seconds': recovery_and_audit,
                  'shutdown_allowance_seconds': shutdown,
                  'modeled_complete_seconds': modeled,
                  'maximum_allocation_gpu_hours': walltime * 2 / 3600,
                  'modeled_complete_gpu_hours': modeled * 2 / 3600,
                  'grading_and_nll_gpu_hours': 0,
                  'automatic_retries': 0,
                  'complete_stage_ceiling_gpu_hours': 2.0,
                  'status': 'PASS_COMPLETE_STAGE' if modeled <= walltime else 'HOLD_REPRICE'}
    price = {**price_body, 'sha256': base.digest(price_body)}
    write_once(args.manifest, manifest)
    write_once(args.price, price)
    print(json.dumps({'manifest_sha256': manifest['sha256'], 'price_sha256': price['sha256'],
                      'requests': manifest['expected_requests'],
                      'prefill_tokens': expected_prefill,
                      'maximum_decode_tokens': manifest['maximum_decode_tokens'],
                      'modeled_complete_seconds': modeled, 'status': price['status']}))


if __name__ == '__main__':
    main()
