"""Reconcile saved paper artifacts, frozen inputs and final Slurm charges."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO/'report/experimental-resume-v1'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    audit_path = DOC/'JOB_AUDIT.json'
    gate_path = DOC/'STUDY_GATES.json'
    audit, gate = load(audit_path), load(gate_path)
    if not audit['all_submitted_jobs_final']:
        raise ValueError('a submitted experiment has no final Slurm state')
    jobs = {j['task']: j for j in audit['jobs']}
    for task in ('paper-package-final','paper-routing-supplement','x2','x2-nll-recovery','x2-analysis'):
        if not jobs[task]['verified_completion']:
            raise ValueError('required completed artifact or job is unverified: ' + task)
    pointer = load(DOC/'PAPER_SNAPSHOT.json')
    paper = Path(pointer['path'])
    frozen = load(paper/'FROZEN.json')
    archive = load(paper/'INPUT_ARCHIVES.json')
    if (str(frozen['job_id']) != str(pointer['job_id']) or
            archive['paper_frozen_sha256'] != sha(paper/'FROZEN.json') or
            set(archive['inputs']) != set(frozen['inputs'])):
        raise ValueError('paper pointer or frozen input archive differs')
    for source, expected in frozen['inputs'].items():
        saved = archive['inputs'][source]
        if saved['sha256'] != expected or sha(Path(saved['archive'])) != expected:
            raise ValueError('archived report input mismatch: ' + source)
    supplement_pointer = load(DOC/'ROUTING_PROFILE_SUPPLEMENT.json')
    supplement = Path(supplement_pointer['path'])
    supplement_frozen = load(supplement/'FROZEN.json')
    if str(supplement_frozen['job_id']) != str(supplement_pointer['job_id']):
        raise ValueError('routing supplement job binding differs')
    for path, expected in supplement_frozen['sources'].items():
        if sha(Path(path)) != expected:
            raise ValueError('routing supplement saved source changed: '+path)
    if load(DOC/'CLAIM_LEDGER.json') != load(paper/'claim-ledger.json'):
        raise ValueError('public claim ledger differs from frozen paper snapshot')
    if load(DOC/'EXPERIMENT_INVENTORY.json') != load(paper/'experiment-inventory.json'):
        raise ValueError('public experiment inventory differs from frozen paper snapshot')
    g3 = load(DOC/'G3_DOSE_SUPPORT_AMENDMENT.json')
    g3_seal = g3.pop('sha256')
    if hashlib.sha256(json.dumps(g3,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() != g3_seal:
        raise ValueError('registered G3 amendment seal changed')
    expected_figures = ('clean-dynamics.pdf','composition-control.pdf','clean-class-transitions.pdf',
                        'X2-dose-diagnostics.pdf','compute-expenditure.pdf')
    expected_supplements = ('routing-velocity-acceleration.pdf','expert-use-profiles.pdf',
                            'routing-profiles.json','expert-use-profiles.csv')
    if not all((paper/name).is_file() for name in expected_figures):
        raise ValueError('paper figure missing')
    if not all((supplement/name).is_file() for name in expected_supplements):
        raise ValueError('routing supplement missing')
    def sumhours(predicate):
        return sum(j['actual_resource_hours'] for j in audit['jobs'] if predicate(j))
    budgets = {
        'H14_GPUh': sumhours(lambda j:j['unit']=='GPU-h' and j['task'].startswith('h14')),
        'X2_GPUh': sumhours(lambda j:j['unit']=='GPU-h' and
                           (j['task']=='x2' or j['task'].startswith('x2-nll'))),
        'new_qualification_GPUh': sumhours(lambda j:j['unit']=='GPU-h' and j['task'].startswith('ordered-qualify')),
        'R3D_CPU_coreh': sumhours(lambda j:j['unit']=='CPU core-h' and j['task'].startswith('r3d-')),
        'paper_CPU_coreh': sumhours(lambda j:j['unit']=='CPU core-h' and j['task'].startswith('paper-')),
    }
    if budgets['H14_GPUh']>1.25 or budgets['X2_GPUh']>8 or budgets['new_qualification_GPUh']>.9 or budgets['R3D_CPU_coreh']>16:
        raise ValueError('verified expenditure exceeds an authorized line')
    if abs(budgets['new_qualification_GPUh']-gate['qualification_stage']['GPU_hours_spent'])>1e-8:
        raise ValueError('routing gate and Slurm audit disagree on cost')
    body = {'schema':'resume-final-status-v1','observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'job_audit':str(audit_path),'job_audit_sha256':sha(audit_path),
            'study_gates':str(gate_path),'study_gates_sha256':sha(gate_path),
            'all_submitted_jobs_final':True,
            'paper_snapshot':str(paper),'paper_job_id':pointer['job_id'],
            'paper_frozen_sha256':sha(paper/'FROZEN.json'),
            'paper_input_archive':str(paper/'INPUT_ARCHIVES.json'),
            'paper_input_archive_sha256':sha(paper/'INPUT_ARCHIVES.json'),
            'routing_supplement':str(supplement),'routing_supplement_job_id':supplement_pointer['job_id'],
            'routing_supplement_frozen_sha256':sha(supplement/'FROZEN.json'),
            'claim_count':len(load(paper/'claim-ledger.json')),
            'G3_registered_passing_cells':g3['registered_passing_cells'],
            'G3_registered_selected':g3['registered_selected'],
            'actual_allocation_hours':budgets,
            'new_study_status':gate['qualification_stage']['status'],
            'new_study_downstream':'HOLD unqualified worker, semantic detector, behavioral eligibility and complete stage pricing',
            'next_qualification_proposal':str(DOC/'QUALIFICATION_RECOVERY_PROPOSAL_v2.json'),
            'next_qualification_authorization':'NOT_RECEIVED_AT_THIS_SNAPSHOT',
            'paper_note':'primary compute chart uses the pre-publication job audit; this receipt includes the two completed report jobs'}
    (DOC/'FINAL_STATUS.json').write_text(json.dumps(body,indent=1)+'\n')
    print(json.dumps({'path':str(DOC/'FINAL_STATUS.json'),'jobs':len(audit['jobs']),
                      'claim_count':body['claim_count'],'actual_allocation_hours':budgets}))


if __name__ == '__main__':
    run()
