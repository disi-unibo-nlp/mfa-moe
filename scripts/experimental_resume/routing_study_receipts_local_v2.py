"""Real native-source binding, offline mechanism/grade export and closeout.

This adapter never upgrades a historical capture's row convention from its
shape alone. A separately observed alignment receipt is needed for prediction
fitting. Missing grades/labels and unresolved dense identities stay explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from moe_exp.routing_control.study_v1 import require, save, sealed, file_sha, digest
from moe_exp.routing_control.study_v2 import verify, seal
from moe_exp.routing_control.families_v2 import require_discovery
from moe_exp.routing_control.anticipation_v2 import build_prediction_frame
from moe_exp.routing_control.mechanism_v2 import early_endpoint
from routing_local_engineering_v2 import compute, FULL_FRAME, REPO

NATIVE_ROOT = FULL_FRAME.parent
ANALYSIS_ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis')
ATTEMPTS = ANALYSIS_ROOT / 'results-r2/qwen36/A/attempts.parquet'


def native_inputs(frame_path, output):
    compute()
    import pandas as pd
    from extract_discovery_routes import trace_at
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest
    frame = sealed(frame_path)
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'source_location', 'trace_sha256', 'tensor_path', 'completion_tokens'])
    by_attempt = {str(r['attempt_id']): r for r in table.to_dict('records')}
    full = sealed(FULL_FRAME)
    indexed = {r['uid']: r for r in full['records']}
    records, excluded = [], []
    for row in frame['records']:
        original = indexed[row['uid']]
        require(row['reader_input'] == original['reader_input'] and row['source_prefix_binding'] == original['analysis_meta'],
                'frozen audit/native source differs')
        source = by_attempt[original['attempt_id']]
        trace = trace_at(source['source_location'])
        require(trace_digest(TraceRecord(**trace)) == source['trace_sha256'] == row['source_prefix_binding']['native_trace_sha256'],
                'native trace source hash mismatch')
        replay = trace['metadata']['token_replay']
        n = row['prefix_tokens']
        require(len(replay['completion_token_ids']) == len(replay['completion_offsets']) == int(source['completion_tokens']) and
                digest(replay['completion_token_ids'][:n]) == row['source_prefix_binding']['prefix_ids_sha256'],
                'native prefix token alignment differs')
        right = replay['completion_offsets'][n-1][1]
        require(trace['cot_text'][:right] == row['reader_input']['emitted_prefix'], 'native prefix reaches wrong text boundary')
        tensor = Path(source['tensor_path'])
        records.append({'uid': row['uid'], 'family': row['family'], 'attempt_id': original['attempt_id'],
            'problem': row['reader_input']['problem'], 'population': 'discovery', 'arm': 'native',
            'prefix_token_ids': replay['completion_token_ids'][:n], 'prefix_ids_sha256': digest(replay['completion_token_ids'][:n]),
            'tokenizer_sha256': replay['tokenizer_sha256'], 'native_trace_sha256': source['trace_sha256'],
            'native_tensor_path': str(tensor), 'native_tensor_file_sha256': file_sha(tensor),
            'native_tensor_layers': 40, 'native_top_k': 8, 'source_location': source['source_location'],
            'coordinate_status': 'REQUIRES_OBSERVED_RAW_INPUT_ROW_TO_PREDICTION_ROW_AUDIT',
            'future_training_labels': 'SEPARATE_OFFLINE_TWO_READER_SUBSTANTIVE_MEASUREMENT_REQUIRED',
            'reader_split': row['split']})
    require_discovery([r['family'] for r in records], frame['family_boundary'])
    result = save(output, {'schema': 'routing-native-analysis-inputs-v2', 'frame_sha256': frame['sha256'],
        'source_frame_sha256': full['sha256'], 'attempts_file_sha256': file_sha(ATTEMPTS),
        'records': records, 'excluded': excluded, 'family_boundary': frame['family_boundary'],
        'status': 'BOUND_NATIVE_SOURCES_PENDING_ALIGNMENT_AND_SEMANTIC_OUTCOMES',
        'fitting': 'Only fold-local nested fits through existing context_v2/anticipation_v2 after both audits pass; unsupported multiple contexts fall back to K=1.',
        'code_files': {str(Path(__file__)): file_sha(__file__)}})
    print(json.dumps({'status': result['status'], 'records': len(records)}), flush=True)


def prediction_inputs(traces, recognizer_audit, alignment_audit, decode, boundary, output):
    """Adapt actually qualified raw native discovery captures, retaining exclusions."""
    compute()
    require_discovery([r['family'] for r in traces], boundary)
    source_bindings = []
    materialized = []
    for trace in traces:
        item = {**trace}
        npz = Path(item.pop('capture_npz'))
        require(file_sha(npz) == item.pop('capture_file_sha256'), 'native raw capture source hash mismatch')
        source_bindings.append({'path': str(npz), 'file_sha256': file_sha(npz)})
        with np.load(npz, allow_pickle=False) as archive:
            item['executed_ids'] = archive['executed_ids']
            item['selected_weights'] = archive['selected_weights'] if 'selected_weights' in archive else None
        materialized.append(item)
    frame = build_prediction_frame(materialized, recognizer_audit, alignment_audit, decode, family_boundary=boundary)
    return save(output, {**{k: v for k, v in frame.items() if k != 'sha256'}, 'native_capture_sources': source_bindings})


def mechanism_receipt(receipt_path, sentence_receipts, transition, output):
    compute()
    success = sealed(receipt_path)
    receipt = verify(success['receipt'])
    require(receipt.get('engineering_only') is False, 'deterministic engineering fixtures are not semantic outcomes')
    sentences = []
    for path in sentence_receipts:
        value = sealed(path)
        require(value['assignment_uid'] == receipt['assignment_uid'] and value['generation_receipt_sha256'] == receipt['sha256'],
                'mechanism readers belong to another generation receipt')
        sentences.append(value['sentence'])
    endpoint = early_endpoint(receipt['emitted_token_ids'], sentences, transition, finish_reason=receipt['finish_reason'])
    return save(output, {**endpoint, 'assignment_uid': receipt['assignment_uid'], 'generation_receipt_sha256': receipt['sha256'],
                         'reader_receipt_sha256': [sealed(p)['sha256'] for p in sentence_receipts]})


def blind_grade_rows(enrollment, successes, prefix_sources, questions, decode):
    """Pure receipt adapter. Gold is read solely in this offline grading function."""
    bundle, mapping, missing = [], [], []
    require(len({a['uid'] for a in enrollment}) == len(enrollment), 'duplicate grading assignment')
    for assignment in enrollment:
        uid = assignment['uid']
        success = successes.get(uid)
        if success is None:
            missing.append({'assignment_uid': uid, 'status': 'MISSING_EXECUTION'})
            continue
        verify(success)
        receipt = verify(success['receipt'])
        require(receipt['assignment_uid'] == uid and receipt['continuation_cap'] == 16384 and
                receipt.get('engineering_only') is False, 'unqualified or wrong-cap grading receipt')
        prefix = prefix_sources[assignment['prefix_ids_sha256']]
        require(digest(prefix) == assignment['prefix_ids_sha256'] and
                len(receipt['emitted_token_ids']) == receipt['generated_tokens'], 'grading prefix or output token source changed')
        blind = digest(['routing-local-offline-grade-v2', uid, receipt['sha256']])
        mapping.append({'assignment_uid': uid, 'blind_uid': blind, 'receipt_sha256': receipt['sha256'],
                        'generated_tokens': receipt['generated_tokens'], 'controller_tokens': receipt['controller']['controller_tokens']})
        if receipt['finish_reason'] == 'length':
            missing.append({'assignment_uid': uid, 'status': 'OPERATIONAL_CAP_WRONG', 'correct': False})
            continue
        require(receipt['finish_reason'] == 'stop', 'unknown generation finish reason')
        question = questions[assignment['question']]
        bundle.append({'uid': blind, 'question': assignment['question'], 'dataset': question['dataset'],
            'problem': question['problem'], 'gold': question['gold'], 'content_text': decode(prefix + receipt['emitted_token_ids']),
            'finish_reason': 'stop', 'natural_stop': True, 'cumulative_tokens': len(prefix) + receipt['generated_tokens']})
    return bundle, mapping, missing


def offline_grade(enrollment_path, results_path, prefixes_path, question_path, output):
    compute()
    # Existing strict/J1 scorer is used in a separate process with its frozen
    # package overlay; this CPU analysis process never repins its live package.
    enrollment, results, prefixes, questions = [sealed(p) for p in (enrollment_path, results_path, prefixes_path, question_path)]
    from transformers import AutoTokenizer
    from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES
    model = CARD_PROFILES['qwen36']
    cache = REPO.parent / 'cache/hf/hub'
    tokenizer = AutoTokenizer.from_pretrained(cache / ('models--' + model['model'].replace('/', '--')) / 'snapshots' / model['revision'], local_files_only=True)
    bundle, mapping, missing = blind_grade_rows(enrollment['assignments'], results['successes'], prefixes['prefixes'], questions['questions'],
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'BLIND_INPUT.json', {'schema': 'routing-offline-blind-grade-input-v2', 'bundle': bundle,
        'question_table_sha256': questions['sha256'], 'enrollment_sha256': enrollment['sha256'],
        'generation_results_sha256': results['sha256']})
    save(output / 'ARM_MAP.json', {'schema': 'routing-offline-arm-map-v2', 'mapping': mapping, 'missing': missing,
        'enrollment_sha256': enrollment['sha256'], 'gold_visible_to_generator_or_mechanism_readers': False})
    command = [sys.executable, '-B', str(Path(__file__).with_name('routing_study_grade_local_v2.py')), str(output / 'BLIND_INPUT.json'), str(output)]
    subprocess.run(command, check=True)


def accounting(jobs):
    command = ['sacct', '-nP', '-j', ','.join(jobs), '--format=JobID%80,State%40,ExitCode,ElapsedRaw,AllocTRES%150']
    observed = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = []
    for line in observed.splitlines():
        if not line.strip():
            continue
        job, state, code, seconds, tres, *_ = line.split('|')
        values = dict(x.split('=', 1) for x in tres.split(',') if '=' in x)
        rows.append({'job_id': job, 'state': state, 'exit_code': code, 'elapsed_seconds': int(seconds), 'TRES': tres,
                     'GPU_hours': int(seconds) * float(values.get('gres/gpu', 0)) / 3600,
                     'CPU_core_hours': int(seconds) * float(values.get('cpu', 0)) / 3600})
    return {'command': command, 'rows': rows}


def native_repeat_variability(qualification, output):
    if qualification.get('status') != 'PASS':
        return None
    cases = {r['case']: r for r in qualification['cases']}
    sources, arrays = [], []
    for name in ('native-before', 'native-after'):
        receipt = verify(sealed(cases[name]['receipt_path'])['receipt'])
        require(receipt['sha256'] == cases[name]['receipt_sha256'], 'native repeat result source changed')
        directory = Path(cases[name]['receipt_path']).parent / 'capture'
        binding = receipt['capture_binding']
        require(binding['rows'] == 384 and all(file_sha(directory / n) == binding['files'][n]
            for n in ('ids.npy', 'weights.npy')), 'native repeat capture changed')
        arrays.append([np.load(directory / n, mmap_mode='r')[:, 0, :384] for n in ('ids.npy', 'weights.npy')])
        sources.append({'case': name, 'generation_receipt_sha256': receipt['sha256'], 'capture_sha256': binding['sha256']})
    matches = arrays[0][0][..., :, None] == arrays[1][0][..., None, :]
    overlap = matches.any(axis=-1).sum(axis=-1)
    before_weights, after_weights = [a[1].astype(np.float64) for a in arrays]
    aligned_after = (matches * after_weights[..., None, :]).sum(axis=-1)
    outside_after = (after_weights * ~matches.any(axis=-2)).sum(axis=-1)
    weight_distance = .5 * (np.abs(before_weights - aligned_after).sum(axis=-1) + outside_after)
    return save(output, {'schema': 'routing-native-repeat-variability-v2', 'sources': sources,
        'ID_cell_mismatches_both_ranks': int(np.count_nonzero(arrays[0][0] != arrays[1][0])),
        'sorted_ID_cell_mismatches_both_ranks': int(np.count_nonzero(np.sort(arrays[0][0], axis=-1) != np.sort(arrays[1][0], axis=-1))),
        'expert_set_turnover_mean': float(np.mean(1. - overlap / 8.)),
        'expert_aligned_weight_L1_half_mean': float(np.mean(weight_distance)),
        'expert_aligned_weight_L1_half_max': float(np.max(weight_distance)),
        'selected_weight_bit_mismatches_both_ranks': int(np.count_nonzero(
            arrays[0][1].view(np.uint32) != arrays[1][1].view(np.uint32))),
        'selected_weight_max_abs_difference': float(np.max(np.abs(arrays[0][1] - arrays[1][1]))),
        'scope': 'Two isolated forced-token native engineering executions, seed 0. No cross-run equality criterion; inactive fidelity is checked within execution.',
        'qualification_sha256': qualification['sha256']})


def semantic_coverage(root, manifest_path, output):
    if not (root / 'audit/PILOT.json').exists():
        return None
    from moe_exp.routing_control.study_v2 import RECOGNIZES
    frame = sealed(Path(manifest_path).with_name('AUDIT_FRAME.json'))
    readers = {}
    for reader in (0, 1):
        for path in (root / 'audit').glob(f'READER_*_{reader}.json'):
            receipt = sealed(path)
            require(receipt['frame_sha256'] == frame['sha256'] and receipt['reader'] == reader,
                    'semantic coverage source mismatch')
            readers[receipt['uid'], reader] = receipt
    rows = []
    for decision in RECOGNIZES:
        for split in ('apparent', 'held_out'):
            selected = [r for r in frame['records'] if r['split'] == split]
            known_positive, caught_positive, agreement_count = 0, 0, 0
            for row in selected:
                pair = [readers.get((row['uid'], reader)) for reader in (0, 1)]
                agreed = all(p and p['finish_reason'] == 'stop' and p['parsed'] and
                    type(p['parsed'].get(decision)) is bool for p in pair) and pair[0]['parsed'][decision] == pair[1]['parsed'][decision]
                agreement_count += bool(agreed)
                positive = bool(agreed and pair[0]['parsed'][decision])
                known_positive += positive
                caught_positive += bool(positive and row['proposals'][decision])
            proposed = sum(bool(r['proposals'][decision]) for r in selected)
            rows.append({'decision': decision, 'split': split, 'prefixes': len(selected), 'proposed': proposed,
                'proposal_coverage': proposed / len(selected) if selected else None,
                'reader_agreement_coverage': agreement_count / len(selected) if selected else None,
                'known_reader_positives': known_positive, 'detected_known_positives': caught_positive,
                'recall_on_known_reader_positives': caught_positive / known_positive if known_positive else None,
                'unknown_reader_prefixes': len(selected) - agreement_count})
    return save(output, {'schema': 'routing-semantic-coverage-v2', 'frame_sha256': frame['sha256'], 'rows': rows,
        'reader_receipt_sha256': [r['sha256'] for _, r in sorted(readers.items())],
        'scope': 'Proposal coverage and recall on agreeing readers are distinct from reader availability. Unknown/disagreeing readers abstain; no new qualification threshold.'})


def write_report(root, output, closeout, submissions):
    states, usage = closeout['states'], closeout['usage']
    q, pilot = states['backend'], states['pilot']
    rows = ['# Routing-study v2 implementation and observed evidence', '',
        f'Observed UTC: {closeout["observed_utc"]}. Closeout receipt: `{closeout["sha256"]}`.', '',
        '| Evidence | Observed status |', '|---|---|',
        f'| CPU preparation and boundary checks | {states["cpu"]["status"]} |',
        f'| Real TP=2 generator/capture qualification | {q["status"]} |',
        f'| Six-decision audit and four outcome-blind pricing cells | {pilot["status"]} |',
        '| Scientific outcome comparisons | Not dispatched; separate measured prices and authorization required |', '',
        'Implemented: eager synchronous start/step/finish, explicit prompt/prefix/prediction coordinates, both-rank action acknowledgements, '
        'all-40-layer native/executed top-eight IDs and float32 weights, execution-local fidelity, binary captures and dose hashes, '
        'live engine/RNG pause receipts, bounded original-prefix assignment retries and duplicate-success rejection. '
        'Conservative prefix proposals and two-reader audits include symbolic candidates and ambiguity abstention. '
        'Native-source, context/anticipation, mechanism, strict/J1 grading and component-price adapters retain qualification gates.', '',
        'Short deterministic fixtures establish engineering behavior only. Historical short-horizon results establish discovery evidence; '
        'they do not establish 16k scientific utility. An interrupted GPU attempt plus whole-assignment rerun tests recovery execution; '
        'the CPU journal test covers uncommitted process-death recovery. No full 16k process-death recovery price is claimed.', '',
        '| Actual job | Role | State | Exit | Elapsed seconds |', '|---|---|---|---|---|']
    by_id = {r['job_id']: r for r in closeout['accounting']['rows'] if '.' not in r['job_id']}
    for job in submissions['jobs']:
        row = by_id.get(job['job_id'], {})
        rows.append(f'| {job["job_id"]} | {job["role"]} | {row.get("state", "UNOBSERVED")} | {row.get("exit_code", "UNKNOWN")} | {row.get("elapsed_seconds", "UNKNOWN")} |')
    rows += ['', f'Observed parent-job use: {usage["GPU_hours"]:.4f} GPU-hours and '
        f'{usage["dedicated_CPU_core_hours"]:.4f} dedicated CPU-core-hours. '
        f'GPU-job CPUs used {usage["GPU_job_CPU_core_hours"]:.4f} additional core-hours within the GPU shapes. '
        'Approved ceilings: 16 GPU-hours and 8 dedicated CPU-core-hours. Steps are retained for audit and excluded from allocation totals.', '']
    if closeout.get('native_repeat_variability'):
        rows += ['Native cross-run variability: `' + json.dumps(closeout['native_repeat_variability']) + '`. '
            'This is descriptive evidence from separate executions; inactive-path fidelity is assessed at the same hidden state within execution.', '']
    if q.get('failures') or q.get('error'):
        rows += ['Backend qualification failures: `' + json.dumps(q.get('failures', q.get('error'))) + '`.', '']
    if pilot.get('failures'):
        rows += ['Audit/pricing limitations: `' + json.dumps(pilot['failures']) + '`.', '']
    if closeout.get('audit_serialization_fix'):
        rows += ['The semantic pilot produced zero ratings: its tokenizer returned a BatchEncoding instead of a token list. '
            'The additive v3 reader adapter normalizes and validates token IDs and preflights all prompts before loading the reader. '
            'CPU verification is recorded in [../AUDIT_SERIALIZATION_FIX.json](../AUDIT_SERIALIZATION_FIX.json); '
            'this correction has no GPU qualification or new submission. Frozen v2 code and receipts remain intact.', '']
    if closeout.get('semantic_coverage'):
        rows += ['Proposal coverage and recall on agreeing readers are retained in [SEMANTIC_COVERAGE.json](SEMANTIC_COVERAGE.json); '
            'reader availability is reported separately. Missing or disagreeing readers remain unknown.', '']
    rows += ['Unresolved: native historical input-row alignment and offline substantive labels must qualify before context/anticipation fitting. '
        'A missing semantic decision prevents scientific dispatch. Replay, grading and any incomplete generation/controller prices remain unavailable. '
        'The original 32 validation and 96 final families, two seeds, first-opportunity rules and scientific 16,384-token cap remain frozen.', '',
        'Dense historical identity accounting: `' + json.dumps(closeout['dense']) + '`. '
        'The existing recovery chain is preserved; this implementation queues no duplicate historical assignments.', '',
        'DeepSeek review adjudication (configured `deepseek-flash`, reasoning `max`, as reported in the supplied plan):']
    rows += [''] + ['- ' + key + ': ' + value for key, value in closeout['deepseek_adjudication'].items()]
    rows += ['', 'The supplied review was incorporated, not rerun. The 70% apparent-precision criterion and ten eligible families are preserved; '
        'held-out uncertainty and coverage are reported without a new threshold. Eight unique expert IDs satisfy the existing turnover formula. '
        'If the cheap recognizer fails, the prepared four-GPU utility generator/reader fallback still needs its own exact-contract price and authorization.', '']
    path = output / 'REPORT.md'
    with path.open('x') as handle:
        handle.write('\n'.join(rows))
    save(output / 'REPORT_BINDING.json', {'schema': 'routing-implementation-report-binding-v2',
        'report_path': str(path), 'report_file_sha256': file_sha(path), 'closeout_sha256': closeout['sha256']})


def closeout(root, output, submissions_path=None, *, metadata_only=False):
    if not metadata_only:
        compute()
    submissions = sealed(submissions_path or root / 'SUBMISSIONS.json')
    require(all(file_sha(path) == expected for path, expected in submissions.get('closeout_code_files', {}).items()),
            'queued closeout source hash mismatch')
    acc = accounting([r['job_id'] for r in submissions['jobs']])
    parent_rows = [r for r in acc['rows'] if '.' not in r['job_id']]
    gpu_jobs = {r['job_id'] for r in submissions['jobs'] if r['gpus']}
    usage = {'GPU_hours': sum(r['GPU_hours'] for r in parent_rows),
        'dedicated_CPU_core_hours': sum(r['CPU_core_hours'] for r in parent_rows if r['job_id'] not in gpu_jobs),
        'GPU_job_CPU_core_hours': sum(r['CPU_core_hours'] for r in parent_rows if r['job_id'] in gpu_jobs),
        'parent_jobs_only': True, 'authorized_GPU_hours': 16, 'authorized_dedicated_CPU_core_hours': 8}
    status = {}
    for name, path in [('cpu', Path(submissions['manifest_path']).with_name('CPU_CHECKS.json')), ('backend', root / 'qualification/QUALIFICATION.json'),
                       ('pilot', root / 'audit/PILOT.json')]:
        status[name] = sealed(path) if path.exists() else {'status': 'UNOBSERVED'}
    output.mkdir(parents=True, exist_ok=False)
    variability = native_repeat_variability(status['backend'], output / 'NATIVE_REPEAT_VARIABILITY.json')
    coverage = semantic_coverage(root, submissions['manifest_path'], output / 'SEMANTIC_COVERAGE.json')
    prices = None
    if (root / 'audit/PILOT.json').exists():
        from routing_study_prices_local_v2 import measured_components
        prices = measured_components(root / 'audit', output / 'prices')
    chain = sealed(REPO / 'report/routing-study-v1/submissions-v5/RECOVERY_CHAIN.json')
    historical = accounting(chain['jobs'] + [chain['analysis_job'], '59495690'])
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    parents = [r for r in historical['rows'] if '.' not in r['job_id']]
    ready = len(parents) == len(set(chain['jobs'] + [chain['analysis_job'], '59495690'])) and all(r['state'].split()[0] in terminal for r in parents)
    if ready and not metadata_only:
        from routing_study_local_v2 import dense_ledger
        dense_ledger(REPO / 'report/routing-study-v1/dense-recovery-v3', output / 'DENSE_IDENTITY_LEDGER.json')
        ledger = sealed(output / 'DENSE_IDENTITY_LEDGER.json')
        dense = {'assigned': ledger['assigned'], 'status_counts': dict(Counter(r['status'] for r in ledger['records'])), 'ledger_sha256': ledger['sha256']}
        save(output / 'DENSE_LEDGER_BINDING.json', {'schema': 'routing-dense-ledger-binding-v2',
            'ledger_sha256': ledger['sha256'], 'ledger_file_sha256': file_sha(output / 'DENSE_IDENTITY_LEDGER.json'),
            'frame_sha256': ledger['frame_sha256'], 'assigned': ledger['assigned'],
            'status_counts': dense['status_counts'], 'historical_chain_sha256': chain['sha256']})
    elif ready and metadata_only and (root / 'closeout/CLOSEOUT.json').exists():
        previous = sealed(root / 'closeout/CLOSEOUT.json')
        previous_dense = previous['dense']
        require(previous.get('metadata_only') is False and previous_dense.get('assigned') == 187524 and
                sum(previous_dense.get('status_counts', {}).values()) == 187524, 'incomplete allocated reconciliation')
        require(previous_dense['ledger_sha256'] == sealed(root / 'closeout/DENSE_LEDGER_BINDING.json')['ledger_sha256'],
                'allocated dense ledger binding changed')
        dense = {**previous_dense, 'allocated_closeout_sha256': previous['sha256'],
                 'ledger_binding_sha256': sealed(root / 'closeout/DENSE_LEDGER_BINDING.json')['sha256']}
    else:
        dense = {'assigned': 187524, 'status': 'TERMINAL_CHAIN_AWAITING_ALLOCATED_RECONCILIATION' if ready else
            'WAITING_EXISTING_RECOVERY_CHAIN', 'requeued_identities': 0}
    receipt = save(output / 'CLOSEOUT.json', {'schema': 'routing-study-implementation-closeout-v2', 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'states': status, 'accounting': acc, 'usage': usage, 'submissions_sha256': submissions['sha256'],
        'historical_accounting': historical, 'dense': dense, 'component_prices': prices,
        'native_repeat_variability': variability,
        'semantic_coverage': coverage,
        'audit_serialization_fix': sealed(root / 'AUDIT_SERIALIZATION_FIX.json') if (root / 'AUDIT_SERIALIZATION_FIX.json').exists() else None,
        'metadata_only': metadata_only,
        'scientific_dispatch_authorized': False,
        'unmeasured_components': ['replay', 'grading', 'scientific_controller_and_generation_unless_four_bound_cells_complete'],
        'deepseek_adjudication': {'bootstrap': 'Separate bounded engineering manifest implemented.',
            'coordinates': 'Original prompt, frozen reasoning prefix, engine input row and continuation offset checked independently.',
            'performance': '16k engineering case measures controller/RPC/capture overhead; scientific prices require actual contract-matching cells.',
            'storage': 'Both ranks, native/executed uint16 IDs + float32 weights, binary dose/token/prefix storage and failed attempts.',
            'recovery': 'Live-engine pauses preserve RNG. One whole-assignment retry from original prefix; at most one accepted success.',
            'held_out_threshold': 'Not adopted: preserve ten eligible families and >=70% apparent precision; report held-out uncertainty/coverage.',
            'turnover': 'Existing turnover formula applies to validated sets of eight unique expert IDs.'},
        'code_files': {str(Path(__file__)): file_sha(__file__)}})
    write_report(root, output, receipt, submissions)
    print(json.dumps({'output': str(output), 'states': {k: v['status'] for k, v in status.items()}, 'dense': dense}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    native = sub.add_parser('native-inputs')
    native.add_argument('--frame', type=Path, required=True)
    native.add_argument('--output', type=Path, required=True)
    for mode in ('closeout', 'observe'):
        close = sub.add_parser(mode)
        close.add_argument('--root', type=Path, required=True)
        close.add_argument('--output', type=Path, required=True)
        close.add_argument('--submissions', type=Path)
    grade = sub.add_parser('offline-grade')
    for name in ('enrollment', 'results', 'prefixes', 'questions', 'output'):
        grade.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'native-inputs':
        native_inputs(args.frame, args.output)
    elif args.mode in ('closeout', 'observe'):
        closeout(args.root, args.output, args.submissions, metadata_only=args.mode == 'observe')
    else:
        offline_grade(args.enrollment, args.results, args.prefixes, args.questions, args.output)
