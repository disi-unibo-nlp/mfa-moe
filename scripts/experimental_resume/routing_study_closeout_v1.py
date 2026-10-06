"""Finite recovery calls and a Slurm-only, sealed routing-study closeout audit.

This reports saved measurements; it never refits an analysis or repeats a GPU
assignment. Native advancement consumes at most one readiness receipt per call.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
RUNS = ROOT / 'runs/routing-control-v1'
OUT = DOC / 'routing-study-closeout-v1'
PLAN = OUT / 'PLAN.json'
OPERATIONS = DOC / 'utility-production-root-submissions-v3-a5a6d77abd5ec32e/OPERATIONS.json'
WRAPPER = REPO / 'scripts/experimental_resume/routing_study_closeout_v1.sbatch'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'changed JSON seal: ' + str(path))
    return value


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        require(sealed(path) == value, 'existing closeout artifact differs: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=1, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    return value


def reference(path, value=None):
    path = Path(path)
    value = sealed(path) if value is None else value
    return {'path': str(path.resolve()), 'file_sha256': sha(path), 'sha256': value.get('sha256')}


def one(paths, label):
    paths = list(paths)
    require(len(paths) == 1, 'missing or ambiguous ' + label)
    return paths[0]


def native():
    # Pin the immutable scoring namespace before importing any shared helpers.
    import utility_native_orchestration_v3 as n
    return n


def accounting(jobs):
    fields = ['JobID%100', 'State%40', 'ExitCode', 'ElapsedRaw', 'AllocTRES%150']
    command = ['sacct', '-X', '-nP', '-j', ','.join(sorted(set(map(str, jobs)))),
               '--format=' + ','.join(fields)]
    raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = {}
    for line in raw.splitlines():
        parts = line.split('|')
        if not line.strip():
            continue
        require(len(parts) >= 5 and parts[0] not in rows, 'ambiguous accounting row')
        identifier, state, code, seconds, tres = parts[:5]
        counts = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
        rows[identifier] = {'job_id': identifier, 'state': state.split(' by ', 1)[0],
            'exit_code': code, 'elapsed_seconds': int(seconds), 'allocated_TRES': tres,
            'allocated_GPU_hours': int(seconds) * int(counts.get('gres/gpu', 0)) / 3600,
            'billing_unit_hours': int(seconds) * float(counts.get('billing', 0)) / 3600}
    return {'command': command, 'raw': raw, 'rows': rows}


def successful(rows, job):
    row = rows.get(str(job), {})
    return row.get('state') == 'COMPLETED' and row.get('exit_code') == '0:0'


def verify_files(value):
    for key in ('code_files', 'source_files', 'frozen_code_files', 'operational_code_files',
                'scientific_code_files', 'unchanged_dispatch_code_files', 'wrapper_hashes'):
        for path, expected in value.get(key, {}).items():
            require(sha(path) == expected, 'changed source: ' + path)


def verify_artifacts(path, value):
    for name, expected in value.get('artifacts', {}).items():
        require(sha(Path(path).parent / name) == expected, 'changed reported artifact: ' + name)


def freeze():
    n = native()
    n.login_guard()
    operations, _ = n.validate_operations(OPERATIONS)
    fresh = DOC / 'overnight-submissions-v2-6b5aa286e46cb818/MEASUREMENT_CHAIN.json'
    dense = one((RUNS / 'generated-dense-fresh-metadata-recovery-v1').glob('*/dispatch-v1/CHAIN.json'), 'fresh dense chain')
    pins = [OPERATIONS, fresh, dense, DOC / 'X2_COMPLETION_AUDIT_v1.json',
        DOC / 'OVERNIGHT_FRESH_FRAME_RECOVERY_AMENDMENT_v1.json',
        DOC / 'GENERATED_DENSE_FRESH_METADATA_RECOVERY_AMENDMENT_v1.json',
        DOC / 'INDEPENDENT_REVIEW_ADJUDICATION_2026-10-02.md',
        DOC / 'DEEPSEEK_REVIEW_DEBATE_2026-10-02_A.md',
        DOC / 'DEEPSEEK_REVIEW_DEBATE_2026-10-02_D.md',
        DOC / 'DEEPSEEK_REVIEW_DEBATE_2026-10-02_D2.md', DOC / 'JOB_AUDIT.json']
    value = save(PLAN, {'schema': 'routing-study-closeout-plan-v1',
        'operations_path': str(OPERATIONS), 'operations_sha256': operations['sha256'],
        'fresh_chain': str(fresh), 'dense_chain': str(dense),
        'pinned_files': {str(p): sha(p) for p in pins},
        'code_files': {str(p): sha(p) for p in (Path(__file__).resolve(), WRAPPER)},
        'resources': {'partition': 'lrd_all_viz', 'qos': 'normal', 'cpus': 2,
                      'memory_GiB': 32, 'wall_seconds': 3600, 'GPU_hours': 0},
        'scope': 'Read saved outcomes and exact receipts. Separate routing engagement, semantic steering and original-prompt utility. No new fit, cohort, GPU replay, sequence study, commit or push.'})
    print(json.dumps({'status': 'FROZEN', 'plan': str(PLAN), 'sha256': value['sha256']}))


def validate_plan():
    plan = sealed(PLAN)
    verify_files(plan)
    for path, expected in plan['pinned_files'].items():
        require(sha(path) == expected, 'changed closeout input: ' + path)
    return plan


def advance(submit):
    n = native()
    n.login_guard()
    operations, _ = n.validate_operations(OPERATIONS)
    root = Path(operations['native_root'])
    require((root / 'ATTACH_CHAIN.json').exists(), 'attach the sealed v3 continuation first')
    for kind in ('j1', 'final', 'recovery', 'initial'):
        ready = root / (kind.upper() + '_READY.json')
        done = root / (kind.upper() + '_NATIVE_DISPATCH.json')
        if not ready.exists():
            continue
        receipt = sealed(ready)
        if done.exists():
            require(sealed(done)['readiness_sha256'] == receipt['sha256'], 'rebound consumed readiness')
            continue
        rows = accounting([receipt['producer_job_id']])['rows']
        if not successful(rows, receipt['producer_job_id']):
            print(json.dumps({'status': 'WAIT_PRODUCER_TERMINAL_ACCOUNTING', 'kind': kind, 'accounting': rows}))
            return
        command = [sys.executable, '-B', str(Path(n.__file__)), 'dispatch',
                   '--operations', str(OPERATIONS), '--ready', str(ready)]
        if submit:
            command.append('--submit')
        subprocess.run(command, check=True)
        return
    holds = [str(p) for p in root.glob('*HOLD.json')]
    print(json.dumps({'status': 'HOLD' if holds else 'WAIT_SEALED_READINESS', 'holds': holds,
        'attachment': reference(root / 'ATTACH_CHAIN.json')}))


def reader_coverage(frame, price, ratings):
    import rate_overnight_semantics_v2 as rating
    committed, valid, summaries = set(), 0, []
    for index, shard in enumerate(price['shards']):
        directory = Path(ratings) / f'shard-{index:03d}'
        summary_path = directory / 'SUMMARY.json'
        if not summary_path.exists():
            continue
        binding, summary = sealed(directory / 'BINDING.json'), sealed(summary_path)
        require(binding['frame_sha256'] == frame['sha256'] and binding['price_sha256'] == price['sha256']
                and binding['shard_index'] == index and binding['driver_sha256'] == sha(rating.__file__)
                and summary['binding_sha256'] == binding['sha256'], 'reader shard rebound')
        hashes, count = [], 0
        for start, block in rating.shard_batches(frame, price, index):
            for reader in (0, 1):
                batch = rating.committed(directory, start, reader, block, binding['sha256'])
                require(batch is not None, 'sealed reader shard has missing batch')
                hashes.append(batch['sha256'])
                for row in batch['records']:
                    identity = (row['blind_id'], reader)
                    require(identity not in committed, 'duplicate assigned reader vote')
                    require(row['rating'] == rating.parse_rating(row['raw_completion']), 'reader parse changed')
                    committed.add(identity)
                    valid += row['finish_reason'] == 'stop' and row['rating'] is not None
                    count += 1
        require(hashes == summary['batch_sha256s'] and count == summary['ratings'] == 2 * (shard['end'] - shard['start']),
                'reader summary omits an assignment')
        summaries.append(reference(summary_path, summary))
    assigned = {(r['blind_id'], reader) for r in frame['records'] for reader in (0, 1)}
    require(committed <= assigned and len(assigned) == price['ratings'] == 6960, 'reader assignment universe changed')
    return {'assigned': len(assigned), 'accounted_in_complete_shards': len(committed),
        'valid_stopped_ratings': valid, 'malformed_or_capped': len(committed) - valid,
        'unknown_ratings': len(assigned - committed), 'unknown_assignment_ids': sorted(assigned - committed),
        'complete_shards': len(summaries), 'assigned_shards': len(price['shards']), 'summaries': summaries,
        'scope': 'Incomplete shards remain explicit unknowns until their full batch accounting verifies.'}


def dense_coverage(directory):
    import generated_dense_fresh_metadata_recovery_v1 as recovery
    recovery.install_adapter()
    dense = recovery.native
    frame, price = recovery.inputs(directory)
    seen, valid, summaries = set(), 0, []
    for index, shard in enumerate(price['shards']):
        out = directory / f'labels-shard-{index:03d}'
        if not (out / 'SUMMARY.json').exists():
            continue
        binding, summary = sealed(out / 'BINDING.json'), sealed(out / 'SUMMARY.json')
        expected = dense.label_binding(frame, price, index)
        require(binding == {**expected, 'sha256': digest(expected)} and
                summary['binding_sha256'] == binding['sha256'], 'dense shard rebound')
        hashes, count = [], 0
        for start in range(shard['start'], shard['stop'], dense.BATCH):
            batch = dense.committed_labels(out, start, min(start + dense.BATCH, shard['stop']), binding, price['prompt_records'])
            require(batch is not None, 'sealed dense shard has missing batch')
            hashes.append(batch['sha256'])
            for row in batch['records']:
                require(row['blind_id'] not in seen, 'duplicate sentence label')
                seen.add(row['blind_id']); count += 1
                valid += row['finish_reason'] == 'stop' and row['label'] in dense.qualified.CLASSES
        require(hashes == summary['batch_sha256s'] and count == summary['sentences'] == shard['stop'] - shard['start'],
                'dense summary omits a sentence')
        summaries.append(reference(out / 'SUMMARY.json', summary))
    assigned = {r['blind_id'] for r in frame['records']}
    require(seen <= assigned and len(assigned) == frame['sentences'] == 187524, 'dense assignment universe changed')
    return frame, price, {'assigned': len(assigned), 'accounted_in_complete_shards': len(seen),
        'valid_stopped_labels': valid, 'malformed_or_capped': len(seen) - valid,
        'unknown_labels': len(assigned - seen), 'unknown_assignment_ids': sorted(assigned - seen),
        'complete_shards': len(summaries), 'assigned_shards': len(price['shards']), 'summaries': summaries}


def semantic(path, expected, job, rows):
    if not path.exists():
        return {'status': 'PENDING', 'assigned': expected, 'job_id': job, 'path': str(path)}
    require(successful(rows, job), 'semantic artifact lacks successful terminal producer')
    value = sealed(path)
    assigned_path = path.parent / 'ASSIGNED_RESULTS.json'
    assigned = sealed(assigned_path)
    records = assigned['records']
    require(len(records) == len({r['uid'] for r in records}) == value['assigned'] == expected and
        assigned['sha256'] == value['assigned_results_sha256'] and
        assigned['manifest_sha256'] == value['manifest_sha256'] and
        sum(x['assigned'] for x in value['arm_summary'].values()) == expected,
        'semantic assigned universe differs')
    return {'status': 'VERIFIED', 'artifact': reference(path, value), 'assigned_results': reference(assigned_path, assigned),
        'assigned': expected, 'valid_pairs': sum(r['reader_pair_valid'] for r in records),
        'measurement_unknown': sum(r['measurement_unknown'] for r in records),
        'primary': value['primary'], 'arm_summary': value['arm_summary'],
        'missing_measurement_sensitivity': value['missing_measurement_sensitivity'],
        'job_id': job, 'interpretation': value['interpretation']}


def nll_audit(rows):
    old = sealed(DOC / 'X2_COMPLETION_AUDIT_v1.json')
    historical = json.loads((DOC / 'JOB_AUDIT.json').read_text())
    record = one((r for r in historical['jobs'] if r.get('task') == 'x2-nll-recovery'), 'X2 NLL record')
    path = Path(record['artifact'])
    require(sha(path) == old['native_nll_file_sha256'], 'X2 native NLL bytes changed')
    data = json.loads(path.read_text())
    require(len(data['records']) == 3354 and all(data['validation'][k]['pass'] for k in ('tf1', 'tf8'))
            and successful(rows, '59101870'), 'X2 NLL incomplete or parity failed')
    failures = []
    for r in historical['jobs']:
        if r.get('task') in ('counterfactual-routing-loss-qualification-v2', 'score-api-calibration-v3', 'score-api-calibration-v4-serial'):
            p = Path(r['artifact']); value = sealed(p)
            require(sha(p) == r['artifact_sha256'] and value['pass'] is False and
                    rows[r['job_id']]['state'] == 'FAILED', 'preserved calibration failure changed')
            failures.append({'task': r['task'], 'artifact': reference(p, value), 'job_id': r['job_id'],
                'assigned': value['assigned'], 'completed': value['completed'],
                'numerical_checks': value.get('numerical_calibration', {}).get('checks'),
                'allocated_GPU_hours': rows[r['job_id']]['allocated_GPU_hours']})
    return {'schema': 'routing-study-NLL-closeout-audit-v1', 'X2_status': 'COMPLETE_NO_RERUN',
        'native_NLL': reference(path, data), 'assigned_sequences': 3354,
        'validation': data['validation'], 'prior_completion': reference(DOC / 'X2_COMPLETION_AUDIT_v1.json', old),
        'counterfactual_score_API_status': 'FAILED_QUALIFICATION_DEFERRED', 'preserved_failures': failures,
        'claim_limit': 'The Q3 repeat-rule pass is scoped to the legacy native-NLL measurement. It does not qualify a different counterfactual score API or establish engine equivalence.'}


def utility_result(n, operations, rows):
    chain_dir = Path(operations['production_chain_dir'])
    if not (chain_dir / 'FINAL_CHAIN.json').exists():
        return []
    final, initial = sealed(chain_dir / 'FINAL_CHAIN.json'), sealed(chain_dir / 'INITIAL_CHAIN.json')
    require(final['native_operations_sha256'] == initial['native_operations_sha256'] == operations['sha256']
            and final['further_automatic_recovery'] is False, 'utility terminal chain rebound')
    index_path = Path(final['reconciled_index'])
    index, grade_root = n.J1.stage_paths(index_path, Path(initial['output']))
    require(index['sha256'] == final['reconciled_index_sha256'] and
            index['manifest_sha256'] == final['manifest_sha256'], 'utility final index rebound')
    path = grade_root / 'analysis/ANALYSIS.json'
    if not path.exists():
        return []
    value = sealed(path)
    expected = {r['assignment']['uid'] for r in index['records']}
    observed = {r['uid'] for r in value['assigned_rows']}
    require(len(value['assigned_rows']) == len(observed) == len(expected) == 384 and observed == expected and
            value['index_sha256'] == index['sha256'] and successful(rows, value['job_id']),
            'utility final assignment or terminal producer accounting incomplete')
    n.J1.priced(grade_root)
    completion = sealed(grade_root / 'J1_COMPLETION.json')
    require(completion['verdicts_file_sha256'] == value['verdicts_file_sha256'] == sha(grade_root / 'verdicts.jsonl'),
            'utility J1 completion changed')
    manifest = sealed(initial['manifest_path']); n.P.validate_manifest(manifest)
    require(len(manifest['imported_receipts']) == 8, 'utility pilot import is not exactly eight assignments')
    return [{'artifact': reference(path, value), 'effect': value['effect'],
        'execution_counts': value['execution_counts'], 'terminal_index': reference(index_path, index),
        'remaining_missing_executions': final['missing'], 'further_automatic_recovery': False,
        'J1_completion': reference(grade_root / 'J1_COMPLETION.json', completion)}]


def review_reconciliation():
    # These are source-audit dispositions, not additional independent reviews.
    entries = [
        ('Fixed serial arm order and pooled random sets', 'RESOLVED_IN_CANONICAL_ASSIGNMENTS',
         'The October 2 adjudication records counterbalanced positive-v3 and per-transition/per-seed random sets; held duplicates consumed zero GPU time.'),
        ('Reader schema, price and failure-cost accounting', 'RESOLVED_FOR_CURRENT_MEASUREMENT',
         'The sealed fresh metadata amendment preserves science and binds 6,960 exact ratings, cold loads, allowances and immutable attempted/committed batches.'),
        ('CPU placement and native balance transport', 'RESOLVED_OPERATIONALLY',
         'Sanitized child environments and finite v3 native-login dispatch preserve failed jobs and query fresh balance and queue commitments.'),
        ('Batched-neighbor differences and serial stochasticity', 'REMAINING_SCOPE_LIMIT',
         'Engineering qualification does not establish global engine equivalence or exact hidden-state pairing. Interpret assigned-policy effects and same-forward inactive telemetry within their frozen scope.'),
        ('Detector and contiguous dense windows', 'VERSIONED_REPAIRS_AND_SCOPE_LIMIT',
         'Precomputed starts are used for continuations. The frozen dense analyzer breaks adjacency at invalid labels and incomplete tails; the separately qualified utility controller retains its own source bindings.'),
        ('Seven-class trajectories and action ordering', 'CLASS_MEASUREMENTS_AVAILABLE_ORDERING_DEFERRED',
         'A/B/C dense outputs are complete; fresh class labeling is assigned. Per-sentence substantive votes and distinct-action ordering remain a separate unperformed study.'),
        ('Approach expert proxy', 'REMAINING_CONSTRUCT_LIMIT',
         'Preserve the frozen commitment proxy and semantic rubric; class transitions alone do not establish semantic commitment.'),
        ('Utility price and accuracy accounting', 'PROSPECTIVE_RESERVATIONS_RESULT_PENDING',
         'The 4.75-GPU-hour line was superseded by the measured pilot and 16,000 billing-hour generation/recovery plus 216-GPU-hour grading allowances. Utility requires canonical strict/J1 outcomes and paired token inference.'),
        ('Small-family intervals and prior confirm exposure', 'REMAINING_INFERENCE_LIMIT',
         'Keep separate multiplicity families, bootstrap approximations, exposure disclosures, K1 scope, K2 uncertainty, G2 non-rejection and G3 dose-selection roles.'),
    ]
    sources = [DOC / name for name in ('INDEPENDENT_REVIEW_ADJUDICATION_2026-10-02.md',
        'DEEPSEEK_REVIEW_DEBATE_2026-10-02_A.md', 'DEEPSEEK_REVIEW_DEBATE_2026-10-02_D.md',
        'DEEPSEEK_REVIEW_DEBATE_2026-10-02_D2.md')]
    return {'schema': 'routing-study-review-reconciliation-v1',
        'sources': [{'path': str(p), 'file_sha256': sha(p), 'bytes': p.stat().st_size} for p in sources],
        'dispositions': [{'finding': f, 'disposition': d, 'evidence_and_limit': e} for f, d, e in entries],
        'scope': 'Current reconciliation of existing advisory reviews. No new external review call or claim of completed independent moderator review.'}


def audit():
    n = native(); n.cpu_guard()
    plan = validate_plan()
    operations, config = n.validate_operations(OPERATIONS)
    import fresh_frame_recovery_v1 as fresh
    amendment = sealed(fresh.AMENDMENT); manifest = sealed(fresh.MANIFEST)
    paths = amendment['paths']; measurement = Path(paths['measurement'])
    amap, frame, price = [sealed(measurement / p) for p in ('ARM_MAP.json', 'BLIND_FRAME.json', 'READER_PRICE.json')]
    fresh.validate_measurement(manifest, amap, frame, price, amendment)
    fresh_chain = sealed(plan['fresh_chain']); dense_chain = sealed(plan['dense_chain'])
    jobs = [fresh_chain['reader_job'], fresh_chain['analysis_job'], dense_chain['label_array_job'], dense_chain['analysis_job'],
        '59380077', '59380073', '59380092', '59350069', '59349213', '59356205',
        '59350111', '59349261', '59378922', '59101870', '59093506', '59095712',
        '59189255', '59196354', '59198795']
    import audit_parallel_jobs_v2 as allocation
    jobs += list(allocation.ROOT_JOBS) + [os.environ['SLURM_JOB_ID']]
    receipt_paths = set(allocation.receipt_paths())
    for parent in ('generated-dense-fresh-metadata-recovery-v1', 'utility-production-v1-*'):
        receipt_paths.update(RUNS.glob(parent + '/**/dispatch*/*.json'))
    for directory in DOC.glob('fresh-veto-sensitivity-chain-*'):
        receipt_paths.update(directory.glob('*.json'))
    for path in receipt_paths:
        jobs += list(allocation.identifiers(sealed(path)))
    acct = accounting(jobs); rows = acct['rows']
    out = OUT / ('audit-' + os.environ['SLURM_JOB_ID'])
    nll = save(out / 'NLL_AUDIT.json', nll_audit(rows))
    readers = reader_coverage(frame, price, Path(paths['ratings']))
    directory = Path(dense_chain['binding']['dense_directory'])
    dense_frame, dense_price, dense_labels = dense_coverage(directory)
    stages = {}
    for lane, count, job, path in (
        ('A', 128, '59350069', DOC / 'OVERNIGHT_DISCOVERY_A_ANALYSIS_v1/ANALYSIS.json'),
        ('B', 156, '59349213', DOC / 'OVERNIGHT_DISCOVERY_B_ANALYSIS_v1/ANALYSIS.json'),
        ('C', 208, '59356205', DOC / 'OVERNIGHT_DISCOVERY_C_ANALYSIS_v2/ANALYSIS.json'),
        ('FRESH', 3480, fresh_chain['analysis_job'], Path(paths['analysis']) / 'ANALYSIS.json')):
        stages[lane] = semantic(path, count, job, rows)
    papers = []
    for path in sorted(DOC.glob('routing-paper-v3-*/SUMMARY.json')):
        value = sealed(path); verify_artifacts(path, value)
        require(sealed(value['source'])['sha256'] == value['source_sha256'], 'paper source changed')
        require(successful(rows, value['report_slurm_job']), 'paper lacks terminal accounting')
        papers.append(reference(path, value))
    engagement = []
    for path in sorted((RUNS / 'routing-first-stage').glob('*/RESULT.json')):
        value = sealed(path); verify_artifacts(path, value)
        engagement.append({'artifact': reference(path, value), 'assigned': value['assigned_requests'],
            'manifest_sha256': value['manifest_sha256'], 'inference': value['inference'], 'claim_limit': value['claim_limit']})
    dense_results = []
    for path in sorted(RUNS.glob('generated-dense*/dense-generated-*/TRAJECTORY_RESULT.json')):
        value = sealed(path); verify_files(value)
        require(sha(value['expert_selection_counts_path']) == value['expert_selection_counts_sha256'], 'expert counts changed')
        completion = sealed(path.parent / 'LABEL_COMPLETION.json')
        require(completion['sha256'] == value['label_completion_sha256'] and completion['sentences'] == value['assigned_sentences'],
                'dense completion changed')
        chain = sealed(path.parent / 'dispatch-v1/CHAIN.json')
        require(successful(rows, chain['analysis_job']), 'dense result lacks successful producer')
        dense_results.append({'artifact': reference(path, value), 'manifest_sha256': sealed(path.parent / 'BLIND_FRAME.json')['generation_manifest_sha256'],
            'assigned_continuations': value['assigned_continuations'], 'assigned_sentences': value['assigned_sentences'],
            'parsed_stop_labels': value['parsed_stop_labels'], 'metrics': value['available_metrics'], 'limitations': value['limitations']})
    utilities = utility_result(n, operations, rows)
    review = save(out / 'REVIEW_RECONCILIATION.json', review_reconciliation())
    sensitivities = [reference(p) for p in sorted(DOC.glob('*SENSITIVITY*.json'))
                     if 'PLAN' not in p.name and 'PRICE' not in p.name]
    required_jobs = ['59380008', '59380077', '59380073', '59380092', '59385736']
    complete = (all(successful(rows, j) for j in required_jobs) and readers['unknown_ratings'] == 0 and
                dense_labels['unknown_labels'] == 0 and stages['FRESH']['status'] == 'VERIFIED' and bool(utilities))
    value = save(out / 'STUDY_INDEX.json', {'schema': 'routing-study-closeout-index-v1',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan['sha256'],
        'status': 'COMPLETE_VERIFIED' if complete else 'IN_PROGRESS',
        'source_bindings': {'operations': reference(OPERATIONS), 'fresh': reference(fresh.AMENDMENT),
                           'dense': reference(DOC / 'GENERATED_DENSE_FRESH_METADATA_RECOVERY_AMENDMENT_v1.json')},
        'semantic': stages, 'fresh_reader_accounting': readers, 'fresh_dense_accounting': dense_labels,
        'dense_results': dense_results, 'routing_engagement': engagement, 'paper_addenda': papers,
        'sensitivities': sensitivities, 'utility': utilities, 'utility_attachment': reference(Path(operations['native_root']) / 'ATTACH_CHAIN.json'),
        'NLL_audit': reference(out / 'NLL_AUDIT.json', nll), 'accounting': acct,
        'review_reconciliation': reference(out / 'REVIEW_RECONCILIATION.json', review),
        'allocated_GPU_hours_so_far': sum(r['allocated_GPU_hours'] for r in rows.values()),
        'allocation_scope': 'Closeout job set: parallel graph, legacy X2 and preserved score-calibration failures, including provisional running allocations. Not total project spending. Project saldo and resource projections are separate.',
        'projected_resources': {'fresh_semantic_GPU_hours': price['estimated_complete_gpu_hours'],
            'fresh_dense_GPU_hours': dense_price['complete_stage_GPU_h'],
            'utility_generation_recovery_billing_core_hours': config['available_generation_billing_core_hours'],
            'utility_grading_GPU_hour_reserve': config['offline_grading_reserve_gpu_hours']},
        'deferred': ['distinct-substantive-action ordering study', 'X3', 'counterfactual score-API calibration'],
        'acceptance': 'Successful terminal accounting and validated saved outputs for every required stage. Malformed and capped measurements retain frozen outcomes; missing executions remain unknown with identification bounds and terminal disposition. No positive effect is required.'})
    write_tables(out, value)
    print(json.dumps({'status': value['status'], 'index': str(out / 'STUDY_INDEX.json'),
        'sha256': value['sha256'], 'fresh_ratings_accounted': readers['accounted_in_complete_shards'],
        'dense_labels_accounted': dense_labels['accounted_in_complete_shards']}))


def write_tables(out, index):
    lines = ['stage\tscope\tarm\treference\testimate\tci_low\tci_high\tfamilies']
    for lane, stage in index['semantic'].items():
        if stage['status'] != 'VERIFIED':
            continue
        for row in stage['primary']['contrasts']:
            ci = row.get('simultaneous_ci95') or [None, None]
            lines.append('\t'.join(map(str, [lane, row['scope'], row['arm'], row['reference'], row['estimate'], *ci, row['families']])))
    (out / 'semantic_contrasts.tsv').write_text('\n'.join(lines) + '\n')
    text = ['# Routing study evidence index', '', f"Observed UTC: {index['observed_utc']}. Status: **{index['status']}**.", '',
        'The saved studies report routing engagement, blinded semantic outcomes and original-prompt utility as separate endpoints.', '',
        '| Study | Assigned continuations | Valid reader pairs | Status |', '|---|---:|---:|---|']
    for lane, value in index['semantic'].items():
        text.append(f"| {lane} | {value['assigned']} | {value.get('valid_pairs', 'pending')} | {value['status']} |")
    text += ['', 'Small A/B/C intervals use separate Bonferroni family-bootstrap approximations. A null interval is compatible with uncertainty and does not establish equivalence.', '',
        f"Fresh ratings accounted in complete shards: {index['fresh_reader_accounting']['accounted_in_complete_shards']}/6,960; dense sentence labels: {index['fresh_dense_accounting']['accounted_in_complete_shards']}/187,524.", '',
        'Dense transitions, dwell, re-entry and loops preserve gaps and incomplete tails. Expert-ID motion measures selection frequencies; substantive-action ordering and full gate-weight trajectories remain outside these measurements.', '',
        'Utility: ' + ('verified saved strict/J1 paired inference is linked in STUDY_INDEX.json.' if index['utility'] else 'pending the original-prompt pilot, production, strict/J1 grading and paired inference.'), '',
        f"Tracked allocated GPU-hours so far: {index['allocated_GPU_hours_so_far']:.4f}. Running allocations are provisional; resource projections are separate.", '',
        'Legacy X2 native NLL is complete. Counterfactual score-API calibration remains a failed qualification; its failed attempts are preserved in NLL_AUDIT.json.', '',
        'Distinct-action steering and X3 are deferred. The immutable index links all source seals, saved figures, sensitivities, unknown assignment identities and exact job accounting.', '']
    (out / 'SYNTHESIS.md').write_text('\n'.join(text))
    save(out / 'ARTIFACTS.json', {'schema': 'routing-study-closeout-artifacts-v1', 'index_sha256': index['sha256'],
        'artifacts': {name: sha(out / name) for name in ('semantic_contrasts.tsv', 'SYNTHESIS.md')}})


def submit_checkpoint(label, dependencies):
    n = native(); n.login_guard()
    plan = validate_plan(); n.validate_operations(OPERATIONS)
    require(label in ('initial', 'semantic', 'dense', 'final'), 'unsupported checkpoint')
    for job in dependencies:
        require(job.isdigit(), 'invalid dependency job')
    # Conservative one-hour CPU/memory allocation allowance; no GPU reservation.
    n.live_preflight(OUT, 'checkpoint-' + label, 8)
    import dispatch_overnight_readers_v1 as shared
    args = [*(['--dependency=afterany:' + ':'.join(dependencies)] if dependencies else []),
            '--job-name=routing-closeout-' + label, str(WRAPPER)]
    job = shared.submit(OUT, 'checkpoint-' + label, args, os.environ.copy(), {'plan_sha256': plan['sha256'], 'label': label})
    print(json.dumps({'status': 'SUBMITTED_OR_REUSED', 'job_id': job, 'label': label}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'audit', 'advance', 'checkpoint'))
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--label', default='initial')
    parser.add_argument('--after', nargs='*', default=[])
    args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'audit': audit()
    elif args.mode == 'advance': advance(args.submit)
    else:
        require(args.submit, 'checkpoint submission requires --submit')
        submit_checkpoint(args.label, args.after)


if __name__ == '__main__':
    main()
