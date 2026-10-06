"""Build arm-blind full-prefix target frame for the 21-family feasibility screen.

All 480 assigned requests stay in the sealed map. The 60 prespecified technical
filler repeats are excluded from the 420-slot semantic estimand and blind frame.
Start eligibility is the prior frozen prefix-only two-Qwen decision, never a
post-intervention rating from these continuations.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket

from build_micro_blind_frame import (THINK_END_ID, TOKENIZER, UNITS,
                                     digest, file_sha, sealed, write_once)

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
AMENDMENT = REPO / 'report/experimental-resume-v1/CAUSAL_DISCOVERY_FEASIBILITY_MICRO_AMENDMENT_v2.json'


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('blind frame decoding belongs on CPU Slurm')
    manifest, amendment, units = (sealed(p) for p in (args.manifest, AMENDMENT, UNITS))
    if (manifest['schema'] != 'routing-discovery-feasibility-generation-v2' or
            manifest['stage'] != 'discovery_feasibility21' or
            manifest['amendment_sha256'] != amendment['sha256'] or
            manifest['rows'] != amendment['rows'] or
            manifest['expected_requests'] != 480 or
            manifest['expected_batches'] != 60):
        raise ValueError('blind frame requires exact qualified full-stage enrollment')
    binding, summary = (sealed(args.run_out / name) for name in
                        ('BINDING.json', 'SUMMARY.json'))
    if (binding['manifest_sha256'] != manifest['sha256'] or
            summary['manifest_sha256'] != manifest['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            summary['requests'] != 480 or summary['batches'] != 60):
        raise ValueError('full generation binding or summary differs')
    source = {r['uid']: r for r in amendment['rows']}
    units_by_key = {(r['family'], r['attempt_id'], r['sentence_index']): r
                    for r in units['records']}
    assigned, output_by_uid, receipts = [], {}, []
    for index in range(60):
        assignment = sealed(args.run_out / f'batch-{index:03d}-assignment.json')
        result = sealed(args.run_out / f'batch-{index:03d}.json')
        array_path = args.run_out / f'batch-{index:03d}.npz'
        if (assignment['manifest_sha256'] != manifest['sha256'] or
                assignment['binding_sha256'] != binding['sha256'] or
                result['manifest_sha256'] != manifest['sha256'] or
                result['binding_sha256'] != binding['sha256'] or
                len(assignment['requests']) != 8 or len(result['outputs']) != 8 or
                [r['uid'] for r in assignment['requests']] !=
                [r['uid'] for r in result['outputs']] or
                file_sha(array_path) != result['array_sha256']):
            raise ValueError('batch assignment/result/routed array differs')
        assigned.extend(assignment['requests'])
        output_by_uid.update({r['uid']: r for r in result['outputs']})
        receipts.append({'index': index, 'assignment_sha256': assignment['sha256'],
                         'result_sha256': result['sha256'],
                         'array_sha256': result['array_sha256']})
    if len(assigned) != 480 or len(output_by_uid) != 480 or \
            len({r['uid'] for r in assigned}) != 480:
        raise ValueError('assigned and observed UID sets differ')
    if {r['uid'] for r in assigned} != set(output_by_uid):
        raise ValueError('missing or extraneous completed request')
    if (sum(r['analysis_enrolled'] for r in assigned) != 420 or
            sum(r['technical_filler'] for r in assigned) != 60):
        raise ValueError('analysis/filler assignment counts changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    map_path = args.out / 'ARM_MAP.json'
    if map_path.exists():
        old = sealed(map_path)
        if (old['manifest_sha256'] != manifest['sha256'] or
                old['generation_binding_sha256'] != binding['sha256']):
            raise ValueError('existing blind map belongs to another run')
        salt = bytes.fromhex(old['blind_salt_hex'])
    else:
        salt = os.urandom(32)
    map_rows, frame_rows = [], []
    for assignment in assigned:
        output = output_by_uid[assignment['uid']]
        prefix = source.get(assignment['prefix_uid'])
        if (prefix is None or prefix['family'] != assignment['family'] or
                prefix['transition'] != assignment['transition'] or
                assignment['prompt_sha256'] != digest(prefix['prompt_ids'] + prefix['prefix_ids'])):
            raise ValueError('assigned prefix/family/transition differs from enrollment')
        blind_id = hmac.new(salt, assignment['uid'].encode(), hashlib.sha256).hexdigest()[:32]
        if assignment['technical_filler']:
            status = 'technical_filler_excluded'
        elif output['error']:
            status = 'generation_error'
        elif not output['routed_present']:
            status = 'missing_routed_array'
        elif not output['tokens']:
            status = 'zero_emitted_tokens'
        elif output['tokens'][0] == THINK_END_ID:
            status = 'reasoning_closed_without_continuation'
        else:
            status = 'gradeable'
        map_rows.append({**assignment, 'blind_id': blind_id,
                         'measurement_status': status,
                         'finish': output['finish'], 'stop_reason': output['stop_reason'],
                         'error': output['error'], 'emitted_tokens': len(output['tokens']),
                         'action_dose': output['action_dose'],
                         'inactive_native_checks': output['inactive_native_checks']})
        if status != 'gradeable':
            continue
        unit = units_by_key.get((prefix['family'], prefix['attempt_id'],
                                 prefix['sentence_index']))
        if (unit is None or unit['trace_sha256'] != prefix['trace_sha256'] or
                unit['inputs']['problem_statement'] != prefix['question']):
            raise ValueError('native trigger sentence source differs')
        emitted = output['tokens']
        reasoning = emitted[:emitted.index(THINK_END_ID)] if THINK_END_ID in emitted else emitted
        frame_rows.append({'blind_id': blind_id,
                           'reader_input': {
                               'problem': prefix['question'],
                               'previous_sentence': unit['inputs']['previous_sentence'],
                               'triggering_sentence': unit['inputs']['sentence'],
                               'full_emitted_prefix': tokenizer.decode(
                                   prefix['prefix_ids'], skip_special_tokens=False),
                               'continuation': tokenizer.decode(reasoning,
                                                                skip_special_tokens=False),
                               'target_transition': prefix['transition']}})
    frame_rows.sort(key=lambda r: r['blind_id'])
    map_body = {'schema': 'routing-discovery-feasibility-arm-map-v2',
                'manifest_sha256': manifest['sha256'],
                'generation_binding_sha256': binding['sha256'],
                'generation_summary_sha256': summary['sha256'],
                'batch_receipts': receipts,
                'source_amendment_sha256': amendment['sha256'],
                'native_units_sha256': units['sha256'],
                'tokenizer_config_sha256': file_sha(TOKENIZER / 'tokenizer_config.json'),
                'blind_salt_hex': salt.hex(), 'records': map_rows}
    mapping = write_once(map_path, map_body)
    frame_body = {'schema': 'routing-discovery-feasibility-blind-target-frame-v2',
                  'generation_manifest_sha256': manifest['sha256'],
                  'source_amendment_sha256': amendment['sha256'],
                  'opaque_id_scheme': 'HMAC-SHA256; salt only in separate arm map',
                  'reader_input_allowlist': ['problem', 'previous_sentence',
                                             'triggering_sentence', 'full_emitted_prefix',
                                             'continuation', 'target_transition'],
                  'prior_start_eligibility': 'Frozen two-reader full-prefix Qwen decision per native prefix; no post-treatment start re-selection',
                  'technical_filler_semantic_grading': 'excluded by predeclared manifest flag; retained in complete ITT assignment map',
                  'continuation_max_tokens': 256,
                  'records': frame_rows}
    frame = write_once(args.out / 'BLIND_FRAME.json', frame_body)
    print(json.dumps({'arm_map': str(map_path), 'arm_map_sha256': mapping['sha256'],
                      'blind_frame': str(args.out / 'BLIND_FRAME.json'),
                      'blind_frame_sha256': frame['sha256'],
                      'assigned': len(assigned), 'analysis_slots': 420,
                      'gradeable': len(frame_rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
