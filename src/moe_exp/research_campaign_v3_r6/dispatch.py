"""One public worker dispatch, immutable cards and terminal stage receipts."""
from .lifecycle import exception_record, failure_scope
import os
from pathlib import Path
import time

from .contracts import MODELS,Units,allocated,check_files,digest,load,locked,publish,require
from .controller import stage_key


def reader_price(campaign,deadline):
    from .reader_pricing import price
    return price(campaign,deadline)


def finish_reader_price(campaign,chosen,selected):
    from .reader_pricing import finish
    return finish(campaign,[r for m,r in chosen],selected)


def worker(campaign,value):
    root=Path(campaign["root"])
    stage,model,shard=value["stage"],value["model"],value["shard_index"]
    require(value["campaign_sha256"]==campaign["sha256"],"card belongs to another campaign")
    check_files(campaign["code_files"])
    check_files(campaign["source_files"])
    from .storage import verify_layout
    verify_layout(campaign)
    allocated(gpu=bool(value["gpus"]))
    if stage=="controller":
        from .controller import control,terminal
        try:
            return control(campaign,value)
        except Exception as exc:
            if shard<2:
                raise  # The prearmed all-ancestor failure owner takes over.
            terminal(campaign,"unresolved_submission",{"error_type":type(exc).__name__,"error":str(exc)[:4000],
                "all_prearmed_owners_exhausted":True,"started_units_never_redrawn":True})
            from .export import closeout
            try:return closeout(campaign)
            except Exception as export_error:
                publish(root/'OPERATIONAL_FAILURE.json',{'status':'failure_owners_exhausted',
                    'failure':exception_record(export_error,'emergency_export'),
                    'controller_handoffs_terminated':True,'REPORT_COMPLETE':(root/'REPORT_COMPLETE.json').exists()})
                raise
    directory=root/"stages"/stage_key(stage,model,shard,value["threshold_index"])
    deadline=time.time()+value["minutes"]*60-120
    with locked(directory,"worker"):
        if (directory/"RESULT.json").exists():
            return load(directory/"RESULT.json")
        if not (directory/"START.json").exists():
            publish(directory/"START.json",{"job_id":os.environ["SLURM_JOB_ID"],"card_sha256":value["sha256"],"started_epoch":time.time()})
        started=time.monotonic()
        try:
            shared=root/"SHARED_IMPLEMENTATION_FAILURE.json"
            if shared.exists() and value["gpus"]:
                return publish(directory/"RESULT.json",{"stage":stage,"model":model,"shard_index":shard,
                    "threshold_index":value["threshold_index"],"status":"upstream_unavailable",
                    "reason":"shared implementation qualification failed before this allocation started work",
                    "shared_failure_receipt":str(shared),"no_scientific_draw_started":True})
            if stage=='validate':
                from .validation import validate
                receipt=validate(campaign)
                detail={'status':'complete','readiness_sha256':receipt['sha256'],'actual_import_graph_validated':True}
            elif stage=="resolve":
                from .inputs import resolve
                if campaign.get("resume_from"):
                    from .resume import import_evidence
                    detail=import_evidence(campaign)
                else:
                    detail=resolve(campaign)
                detail={"status":"complete","reuse_sha256":detail["sha256"]}
            elif stage=="mapping":
                from .populations import mapping
                detail=mapping(campaign,model)
            elif stage=="derive":
                from .derive import derive
                detail=derive(campaign,model)
            elif stage=='analysis_pilot':
                from .analysis import pilot
                detail=pilot(campaign,model)
            elif stage=="supplementary":
                from .observations import supplementary
                detail=supplementary(campaign,model)
            elif stage=="coordination":
                if campaign.get('compact_extraction'):
                    from .analysis import shard as coordination_shard
                    detail=coordination_shard(campaign,model,shard,deadline)
                else:
                    from .observations import coordination
                    detail=coordination(campaign,model,shard)
            elif stage=='analysis_finalize':
                from .analysis import finalize as retire
                detail=retire(campaign,model)
            elif stage=="measure":
                from .workers import measure
                detail=measure(campaign,model,shard,deadline)
            elif stage=="native":
                from .workers import reuse_native
                detail=reuse_native(campaign,model,deadline)
            elif stage=='native_contexts':
                from .annotation import prepare_contexts
                manifest=prepare_contexts(campaign,model,phase='native')
                detail={'status':'complete','contexts_sha256':manifest['sha256']}
            elif stage=='memory_prepare':
                from .early_memory import freeze
                manifest=freeze(campaign,model)
                detail={'status':'complete','tasks_sha256':manifest['sha256']}
            elif stage=='available_finalize':
                from .available import finalize
                detail=finalize(campaign,model)
            elif stage=='memory_qualify':
                from .workers import qualify
                detail=qualify(campaign,model,deadline,shard,component='non_reap')
            elif stage=='memory_price':
                from .early_memory import price
                detail=price(campaign,model,deadline,shard)
            elif stage=="freeze":
                from .workers import freeze
                detail=freeze(campaign,model,deadline,shard)
            elif stage=="qualify":
                from .workers import qualify
                detail=qualify(campaign,model,deadline,shard)
            elif stage=="price":
                from .workers import price
                detail=price(campaign,model,deadline,shard)
            elif stage=="reader_price":
                detail=reader_price(campaign,deadline)
            elif stage=="execute":
                from .workers import execute_rows
                slices=load(root/model/"EXECUTION_SHARDS.json")["shards"]
                table=load(root/model/"TASKS.json")["records"]
                indices=slices[shard] if shard<len(slices) else load(root/model/"execution_fragments"/f"{shard:06d}.json")["indices"]
                detail=execute_rows(campaign,model,[table[i] for i in indices],deadline)
            elif stage=="contexts":
                from .annotation import prepare_contexts
                from .phases import shard_phase
                detail=prepare_contexts(campaign,model,phase=shard_phase(stage,shard) if shard else "production")
                detail={"status":"complete","task_sha256":detail["sha256"]}
            elif stage=="annotate":
                from .annotation import annotate
                from .phases import shard_phase
                phase=shard_phase(stage,shard)
                suffix="-"+phase if shard>=100000 else ""
                table=load(root/model/("ANNOTATION_TASKS"+suffix+".json"))["records"]
                slices=load(root/model/("ANNOTATION_SHARDS"+suffix+".json"))["shards"]
                index=shard%100000-1 if suffix else shard
                detail=annotate(campaign,model,[table[i] for i in slices[index]],deadline)
            elif stage=="score":
                from .scoring import score
                table=load(root/model/"TASKS.json")["records"]
                if shard:
                    from .phases import shard_phase
                    phase=shard_phase(stage,shard)
                    suffix="-"+phase if shard>=100000 else ""
                    index=shard%100000-1 if suffix else shard-1
                    indices=load(root/model/("SCORE_SHARDS"+suffix+".json"))["shards"][index]
                    detail=score(campaign,model,[table[i] for i in indices])
                else:
                    scores=Units(root/model/"scores")
                    detail={"status":"complete","units":len(table),"missing":sum((scores.state(r["uid"]) or {}).get("status")!="complete" for r in table)}
            elif stage == "phase_export":
                from .export import closeout
                from .phases import PHASES
                detail=closeout(campaign,final=False,phase=PHASES[shard-1])
            elif stage=='publish':
                from .publication import deliver
                directories=[root/'output',root/'deliveries/memory',root/'deliveries/central',root/'deliveries/sensitivity',root/'snapshots/2026-10-15']
                detail=deliver(campaign,load(directories[shard]/'REPORT.json'))
            elif stage in ("closeout","snapshot"):
                from .export import closeout
                detail=closeout(campaign,final=stage=="closeout")
            else:
                raise ValueError("unknown registered stage: "+stage)
            result={"stage":stage,"model":model,"shard_index":shard,"threshold_index":value["threshold_index"],
                "status":detail.get("status","complete"),"detail":detail,"seconds":time.monotonic()-started,
                "job_id":os.environ["SLURM_JOB_ID"],"failure_scope":detail.get("failure_scope")}
        except Exception as exc:
            result={"stage":stage,"model":model,"shard_index":shard,"threshold_index":value["threshold_index"],
                "status":"execution_failed",**exception_record(exc,stage,{"card_sha256":value["sha256"],"model":model,"shard":shard}),
                "seconds":time.monotonic()-started,"job_id":os.environ["SLURM_JOB_ID"],"redraw":False,
                "failure_scope":failure_scope(exc)}
        from .recovery import recoverable,attempt_receipt
        if result['status']=='execution_failed' and recoverable(stage,result.get('failure_scope'),value['attempt']):
            attempt_receipt(directory,result,value['attempt'])
            return {**result,'bounded_recovery_pending':True}
        from .failure_domains import record as record_disposition
        record_disposition(campaign,result)
        return publish(directory/"RESULT.json",result)
