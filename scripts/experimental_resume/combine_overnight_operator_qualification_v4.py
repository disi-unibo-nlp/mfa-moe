"""Combine preserved qualification with an additive missing-pulse coverage test.

The original qualification and CPU adjudication remain FAIL. Their sole remaining
coverage gap is met by separate actual GPU traces; no existing request is rerun.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import adjudicate_overnight_operator_qualification_v3 as adjudication

RESULT = adjudication.DOC / 'OVERNIGHT_OPERATOR_COMBINED_QUALIFICATION_v4.json'


def evidence():
    import numpy as np
    original, runner, corrected = adjudication.modules()
    import overnight_routing_pulse_supplement_v4 as supplement
    base, common = runner.base, runner.common
    prior_manifest, prior, _ = supplement.prior_evidence()
    manifest = supplement.validate_manifest(base.sealed(supplement.MANIFEST))
    result = supplement.validate_result(base.sealed(supplement.RESULT))
    common.require(all(base.file_sha(p) == s for p, s in result['code_files'].items()),
                   'supplemental source code changed')
    root = supplement.RUN_ROOT / ('overnight-pulse-supplement-v4-' + manifest['sha256'][:16])
    raw_path, array_path = root / 'RAW_RESULTS.json', root / 'ROUTED.npz'
    common.require(result['artifacts'] == {str(raw_path): base.file_sha(raw_path),
                   str(array_path): base.file_sha(array_path)}, 'supplemental raw evidence differs')
    raw = json.loads(raw_path.read_text())
    common.require(len(raw) == 3 and len(result['checks']) == 3, 'supplement needs all three requests')
    checks, uids = [], []
    with np.load(array_path, allow_pickle=False) as arrays:
        for index, (record, plan, old_check) in enumerate(zip(raw, manifest['plan'], result['checks'], strict=True)):
            uid = 'overnight-pulse-v4|' + base.digest([manifest['sha256'], index])[:24]
            uids.append(uid); row = manifest['fixtures'][0]; native = plan['role'].startswith('native')
            policy = 'zero' if native else row['transition'] + '_bias'
            common.require(record['uid'] == old_check['uid'] == uid and uid in arrays and
                all(record[k] == v for k, v in plan.items() if k != 'role') and
                record['operator'] == plan['role'] and record['role'] == ('native' if native else 'target') and
                record['policy'] == policy and record['policies'] == ([] if native else [policy, policy]) and
                record['prompt_sha256'] == base.digest(row['prompt_ids'] + row['prefix_ids']) and
                record['transition'] == row['transition'] and record['error'] is False,
                'supplemental raw request assignment differs')
            check = adjudication.check_one(record, old_check, manifest, arrays[uid], runner, corrected)
            if not native:
                for rank in ('0', '1'):
                    common.require(record['action_dose'][rank]['rows'] == {policy: 512} and
                        record['action_dose'][rank]['segments'] == {policy: [[0, 256], [512, 768]]},
                        'supplement did not execute two complete distinct pulses')
            common.require(check['pass'], 'supplemental independent raw audit failed')
            checks.append(check)
        common.require(set(arrays.files) == set(uids), 'extra or missing supplemental routed UID')
    return runner, prior_manifest, prior, manifest, result, checks


def prepared():
    runner, original_manifest, prior, manifest, supplement, checks = evidence()
    base = runner.base
    return {'schema': 'overnight-operator-combined-qualification-v4',
            'status': 'PASS_COMBINED_OPERATOR_QUALIFICATION', 'pass': True,
            'qualification_manifest_sha256': original_manifest['sha256'],
            'engine_profile': runner.common.PROFILE, 'requests': 31,
            'original_requests': 28, 'supplemental_requests': 3,
            'original_pass': False, 'original_adjudication_pass': False,
            'original_result_path': prior['original_result_path'],
            'original_result_sha256': prior['original_result_sha256'],
            'original_adjudication_path': str(adjudication.RESULT),
            'original_adjudication_sha256': prior['sha256'],
            'supplemental_manifest_path': str(adjudication.DOC / 'OVERNIGHT_PULSE_SUPPLEMENT_MANIFEST_v4.json'),
            'supplemental_manifest_sha256': manifest['sha256'],
            'supplemental_result_path': str(adjudication.DOC / 'OVERNIGHT_PULSE_SUPPLEMENT_RESULT_v4.json'),
            'supplemental_result_sha256': supplement['sha256'],
            'supplemental_job_id': supplement['job_id'],
            'supplemental_checks': checks, 'kernel_checks_pass': True,
            'preserved_accepted_original_case_indices': [i for i, c in enumerate(prior['checks']) if c['pass']],
            'original_coverage_gap': {'case': 20, 'reason': 'second complete pulse not exercised',
                'status': 'PRESERVED_FAIL_COVERAGE_SUPPLEMENTED_BY_SEPARATE_GPU_FIXTURE'},
            'code_files': {**supplement['code_files'], str(Path(__file__).resolve()): base.file_sha(__file__)},
            'artifacts': {**prior['artifacts'], **supplement['artifacts']},
            'original_gpu_requests_rerun': 0, 'worker_changed': False,
            'interpretation': 'Combined engineering qualification only. Original FAIL receipts preserved. '
                'Only missing second-pulse coverage is supplemented by three new fixed-fixture requests; '
                'all 27 other original cases and checks remain required. No semantic result is selected or inferred.'}


def qualified_result():
    value = prepared()
    _, runner, _ = adjudication.modules()
    stored = runner.base.sealed(RESULT)
    runner.common.require(stored == {**value, 'sha256': runner.base.digest(value)},
                          'combined qualification is not exact saved PASS')
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('combined qualification seal requires CPU Slurm')
    value = prepared()
    _, runner, _ = adjudication.modules()
    result = runner.common.write_once(RESULT, value)
    print(json.dumps({'status': result['status'], 'sha256': result['sha256'],
                      'supplemental_requests': result['supplemental_requests']}), flush=True)


if __name__ == '__main__': main()
