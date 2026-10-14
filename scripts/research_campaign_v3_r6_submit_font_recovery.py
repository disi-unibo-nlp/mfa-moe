#!/usr/bin/env python3
"""Submit the explicitly authorized, one-time font-bound r8 completion through its journal."""
from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]

from moe_exp.research_campaign_v3_r6.contracts import (
    check_files, digest, file_sha, load, publish, require, seal,
)
from moe_exp.research_campaign_v3_r6.scheduler import Slurm, card, serial_capacity, submit_card
from moe_exp.research_campaign_v3_r6.controller import _check, jobs
from moe_exp.research_campaign_v3_r6_ops_r1.policy import check_binding

ROOT = REPO / 'report/research-campaign-v3-20261014-r6'
WRAPPER = REPO / 'scripts/research_campaign_v3_r6_validation_candidate_r8.sbatch'


def authorize():
    pending = load(ROOT / 'OPS_R1_FONT_RECOVERY_BINDING.pending.json')
    check_files(pending['files']); check_files(pending['font_files'])
    authorization = publish(ROOT / 'OPS_R1_FONT_RECOVERY_AUTHORIZATION.json', {
        'authorized': True,
        'authorization': "User: Approve the 10-minute completion and launch",
        'date': '2026-10-14', 'cpus': 4, 'memory_gib': 24, 'minutes': 10,
        'reservation_local_hours': 4/6, 'reservation_with_margin': 5/6,
        'automatic_retries': 0, 'original_ceiling_local_hours': 42050,
        'pending_binding_sha256': pending['sha256'], 'previous_failed_job': '60560738',
        'preserve_report_only_owners': ['60509618', '60509619', '60509620', '60509621'],
        'canonical_manuscript_writes': False, 'remote_writes': False,
        'submission_source_sha256': file_sha(Path(__file__)),
        'scope': 'One font-bound r8 completion reusing actual contracts, reader and export; after full validation release the protected scientific chain and CPU amendment, with complete measured admission before production',
    })
    binding = publish(ROOT / 'OPS_R1_FONT_RECOVERY_BINDING.json', {
        **{k: v for k, v in pending.items() if k != 'sha256'},
        'explicit_extension_authorized': True,
        'authorization_sha256': authorization['sha256'],
        'pending_binding_sha256': pending['sha256'],
    })
    return authorization, binding


class CompletionSlurm(Slurm):
    def command(self, args):
        args = list(args)
        if args[0] == 'sbatch':
            original = str(REPO / 'scripts/research_campaign_v3_r6/job.sbatch')
            require(original in args, 'Unexpected submission entrypoint')
            args[args.index(original)] = str(WRAPPER)
            args[-1] = str(ROOT)
        return super().command(args)

    def verify(self, jid, value):
        result = super().verify(jid, value)
        fields = result['fields']
        require(fields['Command'] == str(WRAPPER) and fields['TimeLimit'] == '00:10:00'
                and fields['Requeue'] == '0' and 'gres/gpu=' not in fields['ReqTRES'],
                'Accepted completion differs from the approved allocation')
        return result


def submit():
    campaign = load(ROOT / 'CAMPAIGN.json')
    authorization, binding = authorize()
    operations = check_binding(campaign)
    check_files(binding['files']); check_files(binding['font_files'])
    scheduler = CompletionSlurm()
    path = ROOT / 'OPS_R1_FONT_RECOVERY_LAUNCH.json'
    if path.exists():
        return load(path)
    require(not (ROOT / 'OPS_R1_FONT_RECOVERY_TERMINAL.json').exists(),
            'The single completion attempt is exhausted')
    value = {k: v for k, v in card(campaign, 'validate', minutes=10, attempt=8, hold=True).items()
             if k not in ('sha256', 'comment')}
    value.update(ops_binding_sha256=operations['sha256'],
                 validation_runner_binding_sha256=binding['sha256'])
    value['claim_cell'].update(
        paper_artifact='Isolated active-source PDF and complete seven-model graph readiness',
        missing_measurement='Remaining PDF, graph traversal and active-source/raw Git integrity proofs',
        cost_bound_local_hours=4/6, scientific_draws=0,
        reused_actual_validation_job='60520129')
    value['comment'] = 'rc3:' + digest(value)
    value = seal(value)
    snapshot = scheduler.snapshot()
    require(all(serial_capacity(snapshot, [value], handoff_slots=0).values()),
            'Completion exceeds live serial limits')
    admission = _check(campaign, scheduler, jobs(campaign, scheduler),
                       hours=4/6, bytes_=512 * 1024**2)
    require(admission['admitted'], 'Authorized completion cannot be admitted')
    proof = publish(ROOT / 'OPS_R1_FONT_RECOVERY_ADMISSION.json', admission)
    accepted = submit_card(campaign, value, scheduler)
    result = publish(path, {
        'allocation': accepted, 'authorization_sha256': authorization['sha256'],
        'binding_sha256': binding['sha256'], 'card_sha256': value['sha256'],
        'admission_sha256': proof['sha256'], 'reservation_local_hours': 4/6,
        'reservation_with_margin': 5/6, 'automatic_retries': 0,
        'scientific_release_requires_complete_validation': True,
        'original_actual_validation_evidence_reused': True,
    })
    release_path = ROOT / 'submissions' / value['sha256'] / 'RELEASED.json'
    if not release_path.exists():
        scheduler.release(accepted['job_id'])
        readback = scheduler.verify(accepted['job_id'], value)
        require(readback['fields'].get('Reason') != 'JobHeldUser', 'Completion remains held')
        publish(release_path, {'job_id': accepted['job_id'], 'proof': readback})
    return result


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('authorize', 'submit'), 'Use authorize or submit')
    result = authorize()[1] if sys.argv[1] == 'authorize' else submit()
    print(json.dumps({'sha256': result['sha256'],
                      'job_id': result.get('allocation', {}).get('job_id')}))
