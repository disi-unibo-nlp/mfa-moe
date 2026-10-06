"""Reseal the pilot with artifact-backed retirement of completed prerequisites."""
import argparse
from pathlib import Path

import utility_scout_v1 as utility
import dispatch_overnight_readers_v1 as shared
from attach_utility_price_pilot_v4 import first_two_assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dispatcher-sha256', required=True,
                        help='Independently reviewed hash of the corrected shared dispatcher')
    args = parser.parse_args()
    doc = utility.REPO / 'report/experimental-resume-v1'
    plan_path = doc / 'UTILITY_SCOUT_PLAN_v1.json'
    qual_path = doc / 'utility-engineering-submissions-v2/CHAIN.json'
    design_path = doc / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'
    plan, qual, design = map(utility.sealed, (plan_path, qual_path, design_path))
    engineering = utility.sealed(doc / 'UTILITY_PAIR_ENGINEERING_PLAN_v2.json')
    family = utility.sealed(doc / 'family-freeze.json')
    families, assignments = first_two_assignments(plan)
    fixture = engineering['fixture_family']
    membership = {pool: fixture in members for pool, members in family['new_parent_pools']['parent_pools'].items()}
    if membership != {'discovery': True, 'mechanism': False, 'utility': False} or fixture in plan['family_order']:
        raise ValueError('engineering exposure differs: review disclosure before preserving enrollment')
    scripts = utility.REPO / 'scripts/experimental_resume'
    dispatcher_sha = utility.file_sha(scripts / 'dispatch_overnight_readers_v1.py')
    if dispatcher_sha != args.dispatcher_sha256:
        raise ValueError('dispatcher is not the reviewed placement-environment fix')
    previous_path = doc / 'UTILITY_RUNTIME_PILOT_PROPOSAL_v3.json'
    previous = utility.sealed(previous_path)
    names = ('prepare_utility_pilot_attachment_v4.py', 'attach_utility_price_pilot_v4.py',
             'attach_utility_price_pilot_v4.sbatch', 'run_utility_price_pilot_v4.sbatch',
             'dispatch_overnight_readers_v1.py')
    body = {'schema': 'utility-runtime-pilot-proposal-v4', 'status': 'READY_BOUNDED_RUNTIME_PILOT_ATTACHMENTS',
        'supersedes_proposal_path': str(previous_path),
        'supersedes_proposal_sha256': previous['sha256'],
        'amendment': 'Scheduling-only correction: preserve active afterok dependencies, but retire completed or purged prerequisites only after sacct COMPLETED 0:0 plus matching sealed artifact validation. Preserve the sealed v3 proposal, corrected placement environment, all 208 engineering sources, enrollment, controller, model profile and statistical policy.',
        'corrected_dispatcher_sha256': dispatcher_sha,
        'utility_plan_path': str(plan_path), 'utility_plan_sha256': plan['sha256'],
        'pilot_families': 2, 'family_order': families, 'assigned_cells': len(assignments),
        'canonical_assignment_uids': [row['uid'] for row in assignments],
        'assignment_selection': 'First two families in frozen utility order; all native/frozen-policy × two-seed cells.',
        'qualification_submission_path': str(qual_path), 'qualification_submission_sha256': qual['sha256'],
        'qualification_job': qual['qualification_job'], 'qualification_result': str(Path(qual['binding']['output']) / 'QUALIFICATION.json'),
        'engineering_plan_sha256': engineering['sha256'],
        'fresh_dispatch_dependency': 'Root supplies UTILITY_FRESH_DISPATCH_JOB and matching afterok dependency when submitting the first attachment. The exact value is recorded in attachment/submission receipts, not fixed in this proposal.',
        'fresh_design_sha256': design['sha256'],
        'gpus': 4, 'wall_seconds': 28800, 'allocation_gpu_hour_ceiling': 32,
        'shutdown_deadline_margin_seconds': 600,
        'maximum_output_tokens_per_assignment': 16384, 'maximum_generator_output_tokens': 8 * 16384,
        'maximum_injected_tokens': 0, 'policy_assignments': 4, 'native_assignments': 4,
        'side_query_count_cap': None, 'side_reader_output_cap_each': 1024,
        'completion_guarantee': False,
        'allocation_rule': 'Single bounded four-GPU allocation up to eight hours. Charge all allocated wall time including both loads, waits, semantic queries, failures and shutdown; no automatic retry. Stop with incomplete artifacts if eight assignments cannot finish.',
        'cpu_attachment_allocations': 'Three one-core 5-minute jobs plus one two-core 30-minute selector/preparation job; 1.25 CPUh total requested ceiling.',
        'dependency_rule': 'After fresh generation dispatcher, attach after its reader dispatcher; then attach selector after complete semantic analysis and successful engineering job. Validate sealed qualification PASS independently before GPU submission and again before load.',
        'engineering_exposure': {'fixture_family': fixture, 'pool_membership': membership,
            'utility_96_overlap': False, 'family_freeze_sha256': family['sha256'],
            'action': 'Original frozen split preserved; no utility-family replacement or fixture-exclusion sensitivity needed.'},
        'reuse_rule': 'Preserve canonical full-study UIDs. Clean pilot receipts can be reused only with identical policy, sampler, controller/code, original prompts and qualified execution profile; retain pilot provenance and all errors/nonfires/caps in ITT.',
        'production_status': 'FULL_96_FAMILY_PRODUCTION_REQUIRES_MEASURED_COMPLETE_STAGE_PROPOSAL',
        'attachment_files': {str(scripts / name): utility.file_sha(scripts / name) for name in names}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = shared.save(args.out, body)
    print(result['status'], result['sha256'], '8 cells; 32 GPUh maximum allocation', flush=True)


if __name__ == '__main__':
    main()
