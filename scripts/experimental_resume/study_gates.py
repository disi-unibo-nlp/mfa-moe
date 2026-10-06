"""Prospective complete-stage pricing and eligibility gates for the authorized study.

Historical throughput is an explicit scenario, never adapter qualification.
Unknown prefill, detector, labeling or retry costs hold launch.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')


def run():
    auth_path=R/'steering-v1/runs/resume-v1/RESOURCE_AUTHORIZATION.json'
    auth=json.loads(auth_path.read_text())
    cost_path=R/'steering-v1/runs/x2prep/cost_model.json'
    cost=json.loads(cost_path.read_text())
    families=json.loads((REPO/'report/experimental-resume-v1/family-freeze.json').read_text())
    audit_path=REPO/'report/experimental-resume-v1/JOB_AUDIT.json'
    audit=json.loads(audit_path.read_text()) if audit_path.is_file() else {'jobs':[]}
    qualification_jobs=[j for j in audit['jobs'] if j['task'].startswith('ordered-qualify')]
    spent=sum(j['actual_resource_hours'] for j in qualification_jobs)
    judge_jobs=[j for j in audit['jobs'] if j['task'] in ('dense-judge-parity','dense-discovery-labels')]
    judge_spent=sum(j['actual_resource_hours'] for j in judge_jobs)
    dense_price_path=REPO/'report/experimental-resume-v1/DENSE_DISCOVERY_LABEL_PRICE.json'
    dense_price=json.loads(dense_price_path.read_text()) if dense_price_path.is_file() else None
    recovery_auth_path=R/'steering-v1/runs/resume-v1/QUALIFICATION_RECOVERY_V2_AUTHORIZATION.json'
    recovery_auth=json.loads(recovery_auth_path.read_text()) if recovery_auth_path.is_file() else None
    qual_ceiling=1.10 if recovery_auth is not None else .75
    pass_path=R/'steering-v1/runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json'
    if not pass_path.is_file():pass_path=R/'steering-v1/runs/ordered-qualification-v1/results-cpu-recovery/QUALIFICATION.json'
    if not pass_path.is_file():pass_path=R/'steering-v1/runs/ordered-qualification-v1/results-retry/QUALIFICATION.json'
    mechanical=json.loads(pass_path.read_text()) if pass_path.is_file() else None
    qualification_status=('PASS_ENGINEERING_ONLY' if mechanical and mechanical['pass'] else
        'SEE_VERIFIED_QUALIFICATION_JOBS')
    throughput=cost['inputs']['x1']['steady_tok_per_s']
    derate=cost['constants']['pessimistic_derate']
    # Full successful allocation elapsed, not merely the timed test batches.
    cold_and_tests_gpu_h=1010*2/3600
    cells={}
    for stage,requests,horizon,ceiling in [('discovery',48*7*2,1024,2.),
        ('mechanism',128*4*2,1024,2.25),('utility',96*2*2,16384,4.75)]:
        decode=requests*horizon
        central=decode/throughput*2/3600
        pessimistic=decode/(throughput*derate)*2/3600
        cells[stage]={'maximum_requests':requests,'maximum_decode_tokens':decode,
            'GPU_hour_ceiling':ceiling,'scenario_source':'historical X1, not qualified controller timing',
            'historical_decode_only_central_GPU_h':central,
            'historical_decode_only_pessimistic_GPU_h':pessimistic,
            'one_observed_load_and_qualification_allocation_GPU_h':cold_and_tests_gpu_h,
            'pessimistic_decode_plus_one_observed_allocation_GPU_h':pessimistic+cold_and_tests_gpu_h,
            'missing_costs':['qualified adapter decode timing at relevant contexts','prefix preparation',
                'complete prefill token count','native NLL if required','independent blind ratings / GPU labeling',
                'retries and per-stage overhead'],
            'stage_price_status':'HOLD_UNQUALIFIED_AND_INCOMPLETE',
            'over_ceiling_in_historical_pessimistic_scenario':pessimistic+cold_and_tests_gpu_h>ceiling}
    value={'schema':'routing-control-resource-gates-v1','authorization':auth,
        'cost_input_sha256':hashlib.sha256(cost_path.read_bytes()).hexdigest(),
        'family_freeze_sha256':families['sha256'],'parent_family_capacity':families['new_parent_pools']['feasibility'],
        'behavioral_eligibility':'UNKNOWN: 48-family discovery windows are prepared; dense class labels and qualified semantic transition ratings remain pending',
        'prefix_candidate_parser':'CPU edge-case qualification exists; semantic transition detector unqualified',
        'ordered_worker':{'per_action_dose_implemented':True,'mechanical_GPU_pass':None if mechanical is None else mechanical['pass'],
            'semantic_detector_qualified':False,'original_prompt_controller_qualified':False},
        'qualification_stage':{'GPU_hour_ceiling':qual_ceiling,'status':qualification_status,
            'recovery_authorization':recovery_auth,
            'jobs':qualification_jobs,'audit_observed_utc':audit.get('observed_utc'),
            'GPU_hours_spent':spent,'GPU_hours_remaining':qual_ceiling-spent,
            'successful_artifact':str(pass_path) if qualification_status=='PASS_ENGINEERING_ONLY' else None,
            'prior_price_receipt_failed_before_model_load':str(R/'steering-v1/runs/ordered-qualification-v1/PREPARED.retry.json'),
            'current_driver_minimum_load_test_allowance_seconds':950,
            'observed_scheduler_shutdown_seconds':196,
            'minimum_allowance_and_shutdown_GPUh':(950+196)*2/3600,
            'fits_remaining_even_with_zero_CPU_preparation':(950+196)*2/3600<=qual_ceiling-spent,
            'CPU_prefix_preparation':str(R/'steering-v1/runs/ordered-qualification-v1/CPU_PREFIXES.json'),
            'recovery_proposal':'completed as job 59108983; 17/17 worker checks passed; semantic and throughput gates remain'},
        'dense_measurement':{'parity_audit':dense_price['status'] if dense_price else 'PENDING',
            'operational_coverage':dense_price['operational_coverage'] if dense_price else None,
            'historical_class_agreement':dense_price['all_assigned_agreement'] if dense_price else None,
            'label_stage_price':dense_price['complete_price'] if dense_price else None,
            'jobs':judge_jobs,
            'status':'LABELS_SUBMITTED_CLASS_AUDIT_ONLY; semantic and online detector gates remain'},
        'stages':cells,'new_study_GPU_hours_spent':spent+judge_spent,
        'decision':'HOLD action discovery/mechanism/utility until dense labels, semantic detector, eligible families and complete intervention price pass; direct-judge class consistency passed',
        'revised_proposal':{'status':'HISTORICAL_SCENARIO_ONLY_NOT_COMPLETE_PRICE',
            'utility_historical_pessimistic_scenario_GPU_h':cells['utility']['pessimistic_decode_plus_one_observed_allocation_GPU_h'],
            'other_missing_components':'must be measured and added to a versioned complete price before submission',
            'expanded_authorization':'RESOURCE_AUTHORIZATION_EXPANDED.json'}}
    path=REPO/'report/experimental-resume-v1/STUDY_GATES.json'
    path.write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps({'path':str(path),'action_discovery_and_later_submissions':'HOLD','study_GPU_hours_spent':spent+judge_spent,
        'utility_historical_pessimistic_scenario_GPU_h':value['revised_proposal']['utility_historical_pessimistic_scenario_GPU_h']}))


if __name__=='__main__':run()
