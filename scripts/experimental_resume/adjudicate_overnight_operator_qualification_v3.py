"""CPU-only re-audit of immutable operator qualification with a corrected slot bound.

Only the demonstrated combined membership/L1 verifier error may be superseded,
and only after every replacement check passes. All other original failure and
coverage reasons survive. No qualification GPU request is rerun by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
RUNS = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')
RESULT = DOC / 'OVERNIGHT_OPERATOR_QUAL_ADJUDICATION_v3.json'
CHECKER_REASON = 'invalid membership or gate displacement dose'


def modules():
    import sys
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    overlay = RUNS.parents[1] / 'addenda/ordered/9727c10299b71e7a/moe_exp_src'
    if 'moe_exp' not in sys.modules:
        pin_qualified_worker(overlay)
    elif Path(sys.modules['moe_exp'].__file__).resolve().parent != overlay / 'moe_exp':
        raise RuntimeError('adjudication requires the immutable worker namespace')
    import overnight_routing_qualify_v2 as original
    import overnight_routing_runner_v2 as runner
    import overnight_routing_dose_audit_v3 as corrected
    return original, runner, corrected


def paths(original, manifest):
    out = RUNS / ('overnight-operator-qual-v2-' + manifest['sha256'][:16])
    return out, out / 'QUALIFICATION.json', out / 'RAW_RESULTS.json', out / 'ROUTED.npz'


def check_one(record, old_check, manifest, route, runner, corrected):
    """Preserve non-checker failures, then independently recheck all saved evidence."""
    import numpy as np
    from moe_steer.engine import THINK_END_ID
    reasons = [r for r in old_check['reasons'] if r != CHECKER_REASON]
    if route.shape != (len(record['tokens']), 40, 8) or not ((route >= 0) & (route < 256)).all() or \
            not (np.diff(np.sort(route, axis=-1), axis=-1) > 0).all():
        reasons.append('raw executed top8 shape, IDs or uniqueness differs')
    try:
        if record['closed']:
            for entry in record['action_dose'].values():
                runner.common.require(not any(entry['rows'].values()) and not any(entry['segments'].values()) and
                    all(all(v == 0 for v in layer.values()) for layers in entry['dose'].values() for layer in layers.values()),
                    'closed prefix has nonzero executed intervention')
            audit = corrected.audit_output_dose({**record, 'role': 'native', 'policies': [], 'slots': []},
                {'horizon': record['horizon'], 'actions': manifest['actions']})
        else:
            audit = runner.audit_operator_dose(record,
                {'horizon': record['horizon'], 'actions': manifest['actions']}, corrected.audit_output_dose)
        if record['slots'] == [0, 512] and not record['closed']:
            stop = next((i + 1 for i, t in enumerate(record['tokens']) if t == THINK_END_ID), len(record['tokens']))
            runner.common.require(stop >= 768, 'second complete pulse not exercised')
        if record['operator'] == 'force' and not record['closed']:
            action = next(a for a in manifest['actions'] if a['name'] == record['policy'])
            for layer, targets in action['experts']:
                n = int(record['action_dose']['0']['dose'][record['policy']][str(layer)]['active_rows'])
                runner.common.require(all(np.any(route[:n, layer, :] == expert, axis=-1).all() for expert in targets),
                                      'raw executed force IDs omit a target during the pulse')
    except Exception as exc:
        audit = None; reasons.append(str(exc))
    return {'uid': record['uid'], 'pass': not reasons, 'original_pass': old_check['pass'],
            'original_reasons': old_check['reasons'], 'reasons': list(dict.fromkeys(reasons)),
            'superseded_checker_reason': CHECKER_REASON if CHECKER_REASON in old_check['reasons'] and audit is not None else None,
            'corrected_audit': audit}


def adjudicate():
    import numpy as np
    original, runner, corrected = modules(); base = runner.base; require = runner.common.require
    manifest = original.validate_manifest(base.sealed(original.MANIFEST))
    out, raw_result_path, raw_path, arrays_path = paths(original, manifest)
    result = base.sealed(raw_result_path)
    require(result['qualification_manifest_sha256'] == manifest['sha256'] and
            result['code_files'] == manifest['code_files'] and result['engine_profile'] == runner.common.PROFILE and
            result.get('kernel_checks_pass') is True and result['requests'] == len(manifest['plan']) == 28 and
            len(result['checks']) == 28, 'original qualification is incomplete or kernel/source checks failed')
    require(all(base.file_sha(p) == s for p, s in manifest['code_files'].items()), 'qualified engine source changed')
    require(result['artifacts'] == {str(raw_path): base.file_sha(raw_path), str(arrays_path): base.file_sha(arrays_path)},
            'original qualification raw artifacts changed')
    raw = json.loads(raw_path.read_text()); require(len(raw) == 28, 'not all raw qualification cases are present')
    checks = []; expected_uids = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for index, (record, plan, old_check) in enumerate(zip(raw, manifest['plan'], result['checks'], strict=True)):
            uid = 'overnight-qual-v2|' + base.digest([manifest['sha256'], index])[:24]
            expected_uids.append(uid); fixture = manifest['fixtures'][plan['fixture']]
            native = plan['role'].startswith('native')
            policy = 'zero' if native else fixture['transition'] + '_' + plan['role']
            from moe_steer.engine import THINK_END_ID
            prompt = fixture['prompt_ids'] + fixture['prefix_ids'] + ([THINK_END_ID] if plan['closed'] else [])
            require(record['uid'] == old_check['uid'] == uid and
                    all(record[k] == v for k, v in plan.items() if k != 'role') and
                    record['operator'] == plan['role'] and record['role'] == ('native' if native else 'target') and
                    record['policy'] == policy and record['policies'] == ([] if native else [policy]*len(plan['slots'])) and
                    record['prompt_sha256'] == base.digest(prompt) and record['transition'] == fixture['transition'] and
                    record['error'] is False and uid in arrays, 'raw qualification assignment differs')
            checks.append(check_one(record, old_check, manifest, arrays[uid], runner, corrected))
        require(set(arrays.files) == set(expected_uids), 'routed archive has missing or extra UID')
    helper_sources = {str(Path(p).resolve()): base.file_sha(p) for p in (__file__, corrected.__file__)}
    body = {'schema': 'overnight-operator-qualification-adjudication-v3',
            'status': 'PASS_CPU_ADJUDICATED_OPERATOR_QUALIFICATION' if all(c['pass'] for c in checks)
                      else 'FAIL_REMAINING_QUALIFICATION_CHECKS',
            'pass': all(c['pass'] for c in checks), 'qualification_manifest_sha256': manifest['sha256'],
            'engine_profile': runner.common.PROFILE, 'requests': len(raw), 'kernel_checks_pass': True,
            'original_result_path': str(raw_result_path), 'original_result_sha256': result['sha256'],
            'original_pass': result['pass'], 'original_job_id': result['job_id'],
            'code_files': {**manifest['code_files'], **helper_sources}, 'artifacts': result['artifacts'],
            'checks': checks, 'gpu_reexecution': False,
            'interpretation': 'CPU checker adjudication only. Original failure is preserved. Inserted expert slots are bounded by8R; mR is diagnostic without stable tie-order assumptions. All other original failures, engine coverage and checks remain required.'}
    return runner.common.write_once(RESULT, body)


def qualified_result():
    """Runtime gate for the new adapter; never writes or disguises the v2 failure."""
    original, runner, corrected = modules(); base = runner.base; require = runner.common.require
    value = base.sealed(RESULT); manifest = original.validate_manifest(base.sealed(original.MANIFEST))
    require(value['schema'] == 'overnight-operator-qualification-adjudication-v3' and value['pass'] is True and
            value['status'] == 'PASS_CPU_ADJUDICATED_OPERATOR_QUALIFICATION' and
            value['qualification_manifest_sha256'] == manifest['sha256'] and value['requests'] == 28 and
            len(value['checks']) == 28 and all(c['pass'] for c in value['checks']) and
            value['kernel_checks_pass'] is True and value['gpu_reexecution'] is False and
            all(base.file_sha(p) == sha for p, sha in value['code_files'].items()) and
            value['code_files'].get(str(Path(__file__).resolve())) == base.file_sha(__file__) and
            value['code_files'].get(str(Path(corrected.__file__).resolve())) == base.file_sha(corrected.__file__),
            'corrected qualification is not exact PASS')
    require(base.sealed(Path(value['original_result_path']))['sha256'] == value['original_result_sha256'] and
            all(base.file_sha(p) == sha for p, sha in value['artifacts'].items()), 'original qualification evidence changed')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('complete raw qualification adjudication requires CPU Slurm')
    value = adjudicate()
    print(json.dumps({'status': value['status'], 'sha256': value['sha256'],
                      'failed': [c for c in value['checks'] if not c['pass']]}), flush=True)
    if not value['pass']: raise SystemExit(3)


if __name__ == '__main__': main()
