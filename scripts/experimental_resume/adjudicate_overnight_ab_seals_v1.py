"""CPU-only A/B completion adjudication for an expert-slot/row checker error.

Original generation files, passing completion seals and frozen checker code
remain intact. New completion seals explicitly reference independent corrected
audits; they never turn an original failed check into an unqualified old PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'report/experimental-resume-v1/OVERNIGHT_AB_DOSE_ADJUDICATION_PLAN_v1.json'
CORRECTED_SHA = '25954a18784403d7600c4a16f78366871759675f5043bf8d71b1baae2da3bf87'
ORIGINAL_DIMENSION_FAILURE = 'invalid membership or gate displacement dose'


def code_files():
    import hashlib
    paths = [Path(__file__).with_name(name) for name in
             ('adjudicate_overnight_ab_seals_v1.py', 'build_price_overnight_semantics_adjudicated_v1.sbatch',
              'overnight_routing_dose_audit_v3.py', 'overnight_routing_runner_v1.py',
              'diagnose_mechanism_validation_v3.py', 'build_overnight_blind_frame_v1.py',
              'price_overnight_semantics_v1.py')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def preserved_or_new(common, path, core, provenance):
    """Existing original successful seals remain byte-for-byte unchanged."""
    if path.exists():
        old = common.base.sealed(path)
        common.require(all(old.get(k) == v for k, v in core.items()), 'existing completion core differs')
        if 'dose_adjudication' in old:
            common.require(old['dose_adjudication'] == provenance, 'existing adjudication binding differs')
        return old
    return common.write_once(path, {**core, 'dose_adjudication': provenance})


def seal_one(common, corrected, out, manifest, index, binding):
    directory = Path(out) / f'shard-{index:03d}'
    summary = common.base.sealed(directory / 'SUMMARY.json')
    source_receipts = []
    for batch_index in range(summary['batches']):
        assignment_path = directory / f'batch-{batch_index:03d}-assignment.json'
        batch_path = directory / f'batch-{batch_index:03d}.json'
        assignment, batch = common.base.sealed(assignment_path), common.base.sealed(batch_path)
        common.require(assignment['manifest_sha256'] == batch['manifest_sha256'] == manifest['sha256'] and
            assignment['binding_sha256'] == batch['binding_sha256'] == summary['binding_sha256'] and
            [r['uid'] for r in assignment['requests']] == [r['uid'] for r in batch['outputs']],
            'raw assignment coverage or manifest binding differs')
        source_receipts.append({'assignment_path': str(assignment_path), 'assignment_sha256': assignment['sha256'],
            'batch_path': str(batch_path), 'batch_sha256': batch['sha256'], 'array_sha256': batch['array_sha256']})
    original_audit, original_write = common.audit_output_dose, common.write_once
    checks, captures = [], []

    def audit(result, frozen_manifest):
        # Every old check except its dimensional error must still pass. The
        # new check runs first, including stricter target-hit upper bounds.
        accepted = corrected.audit_output_dose(result, frozen_manifest)
        old_failure = None
        try:
            original_audit(result, frozen_manifest)
        except ValueError as error:
            old_failure = str(error)
            common.require(old_failure == ORIGINAL_DIMENSION_FAILURE,
                           'original checker failed for an unadjudicated reason: ' + old_failure)
        checks.append({'uid': result['uid'], 'original_check': 'PASS' if old_failure is None else 'REJECTED_DIMENSION_BOUND',
                       'original_failure': old_failure, 'corrected_check': accepted})
        return accepted

    def capture(path, body):
        common.require(Path(path) == directory / 'OVERNIGHT_COMPLETION.json', 'unexpected seal write')
        captures.append(body)
        return {**body, 'sha256': common.base.digest(body)}

    common.audit_output_dose, common.write_once = audit, capture
    try:
        common.seal_shard(Path(out), manifest, index)
    finally:
        common.audit_output_dose, common.write_once = original_audit, original_write
    common.require(len(captures) == 1 and len(checks) == captures[0]['counts']['assigned'],
                   'corrected audit omitted an assigned output')
    proof_path = directory / 'DOSE_ADJUDICATION_v1.json'
    proof = original_write(proof_path, {'schema': 'overnight-ab-shard-dose-adjudication-v1',
        'binding': binding, 'manifest_sha256': manifest['sha256'], 'shard': index,
        'source_summary_sha256': summary['sha256'], 'raw_receipts': source_receipts, 'checks': checks,
        'original_dimension_rejections': sum(c['original_failure'] is not None for c in checks),
        'status': 'PASS_CORRECTED_CPU_DOSE_AUDIT',
        'change': 'Inserted expert slots bounded by 8 × active rows; target-count × rows is diagnostic. All other old checks retained.',
        'generation_repeated': False})
    link = {'path': str(proof_path), 'sha256': proof['sha256'], 'corrected_checker_sha256': CORRECTED_SHA,
            'status': proof['status']}
    receipt = preserved_or_new(common, directory / 'OVERNIGHT_COMPLETION.json', captures[0], link)
    return receipt, proof


def run(args):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID') or
            os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz' or socket.gethostname().startswith('login')):
        raise RuntimeError('A/B adjudication requires a CPU Slurm step')
    # Pin before importing either the frozen runner or the corrected helper.
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    overlay = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
                   'steering-v1/addenda/ordered/9727c10299b71e7a/moe_exp_src')
    pin_qualified_worker(overlay)
    import overnight_routing_runner_v1 as common
    import overnight_routing_dose_audit_v3 as corrected
    plan = common.base.sealed(PLAN)
    common.require(plan['schema'] == 'overnight-ab-dose-adjudication-plan-v1' and
        plan['code_files'] == code_files() and common.base.file_sha(corrected.__file__) == CORRECTED_SHA,
        'adjudication plan or independently reviewed checker changed')
    manifest, price = common.base.sealed(args.manifest), common.base.sealed(args.price)
    common.require(str(args.manifest.resolve()) in plan['authorized_manifest_paths'] and
        price['manifest_sha256'] == manifest['sha256'] and price['shards'] == manifest['shards'] and
        price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY', 'not the authorized complete A/B generation')
    common.validate(manifest, common.base.__file__)
    common.require(all('bias' in action for action in manifest['actions']), 'A/B adjudication is bias-only')
    binding = {'schema': 'overnight-ab-dose-adjudication-binding-v1', 'plan_sha256': plan['sha256'],
        'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
        'original_checker_path': str(Path(common.__file__).resolve()),
        'original_checker_sha256': common.base.file_sha(common.__file__),
        'corrected_checker_path': str(Path(corrected.__file__).resolve()),
        'corrected_checker_sha256': CORRECTED_SHA, 'entry_sha256': common.base.file_sha(__file__)}
    with (args.out / 'DOSE_ADJUDICATION_WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts, proofs = [], []
        for index in range(len(manifest['shards'])):
            receipt, proof = seal_one(common, corrected, args.out, manifest, index, binding)
            receipts.append(receipt); proofs.append(proof)
        counts = Counter()
        for receipt in receipts:
            counts.update(receipt['counts'])
        common.require(counts['assigned'] == manifest['expected_requests'], 'incomplete generation stage')
        proof_path = args.out / 'DOSE_STAGE_ADJUDICATION_v1.json'
        proof = common.write_once(proof_path, {'schema': 'overnight-ab-stage-dose-adjudication-v1',
            'binding': binding, 'manifest_sha256': manifest['sha256'], 'counts': dict(counts),
            'shard_adjudication_sha256s': [p['sha256'] for p in proofs],
            'shard_completion_sha256s': [r['sha256'] for r in receipts],
            'original_dimension_rejections': sum(p['original_dimension_rejections'] for p in proofs),
            'status': 'PASS_CORRECTED_CPU_DOSE_AUDIT', 'generation_repeated': False})
        link = {'path': str(proof_path), 'sha256': proof['sha256'], 'corrected_checker_sha256': CORRECTED_SHA,
                'status': proof['status']}
        result = preserved_or_new(common, args.out / 'STAGE_COMPLETION.json',
            {'schema': 'overnight-routing-stage-completion-v1', 'manifest_sha256': manifest['sha256'],
             'counts': dict(counts), 'shard_completion_sha256s': [r['sha256'] for r in receipts],
             'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION'}, link)
        print(json.dumps({'status': proof['status'], 'adjudication_sha256': proof['sha256'],
            'stage_completion_sha256': result['sha256'], 'original_dimension_rejections': proof['original_dimension_rejections'],
            'assigned': counts['assigned'], 'generation_repeated': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'price', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
