"""One public worker dispatch, immutable cards and terminal stage receipts."""
import os
from pathlib import Path
import time

from .contracts import MODELS,Units,allocated,check_files,digest,load,locked,publish,require
from .controller import stage_key


def reader_price(campaign,deadline):
    from .annotation import prepare_contexts,annotate,Renderer
    root=Path(campaign["root"])
    candidates=[]
    for model in MODELS:
        if not (root/model/"TASKS.json").exists() and not (root/model/"NATIVE_TASKS.json").exists():
            continue
        table=prepare_contexts(campaign,model,phase="price")
        units=Units(root/model/"annotations")
        candidates.extend((model,r) for r in table["records"] if units.state(r["uid"]) is None)
    if not candidates:
        return publish(root/"ANNOTATOR_PRICE.json",{"status":"upstream_unavailable","reason":"no genuinely missing compatible original annotator context"})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(campaign["annotator"]["path"],local_files_only=True)
    renderer=Renderer(campaign)
    sizes={}
    try:
        config=campaign["annotator"]["config"]
        for model,row in candidates:
            require(time.time()<deadline-120,"deadline before exact original reader input sizing")
            conversation=renderer.request("render",inputs=row["inputs"])["conversation"]
            tokens=tokenizer.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True,
                enable_thinking=config["enable_thinking"],preserve_thinking=config["preserve_thinking"],reasoning_effort=config["reasoning_effort"])
            sizes[model,row["uid"]]=len(tokens)
            if len(tokens)>=campaign["annotator"]["native_context"]:
                units=Units(root/model/"annotations")
                if units.begin(row,os.environ["SLURM_JOB_ID"]):
                    units.commit(row,{"status":"upstream_unavailable","label":None,"unknown":True,
                        "reason":"complete original context exceeds the preserved reader context","input_tokens":len(tokens),"draw_started":False})
    finally:renderer.close()
    candidates=[(m,r) for m,r in candidates if Units(root/m/"annotations").state(r["uid"]) is None]
    if not candidates:return publish(root/"ANNOTATOR_PRICE.json",{"status":"upstream_unavailable","reason":"no missing context fits the preserved original reader"})
    candidates.sort(key=lambda mr:(-sizes[mr[0],mr[1]["uid"]],mr[1]["uid"]))
    source_model=candidates[0][0]
    chosen=[item for item in candidates if item[0]==source_model][:campaign["annotator"]["config"]["batch_size"]]
    # Unique original contexts, retained labels, no fitted recognizer or J1.
    selected=publish(root/"ANNOTATOR_PRICE_TASKS.json",{"records":[{"model":m,**r} for m,r in chosen],
        "original_batch_size":campaign["annotator"]["config"]["batch_size"],"outcomes_used":False})
    for model in dict.fromkeys(m for m,r in chosen):
        annotate(campaign,model,[r for m,r in chosen if m==model],deadline)
    measured=[]
    for model,row in chosen:
        result=Units(root/model/"annotations").state(row["uid"])
        require(result and "seconds" in result,"original annotator execution price unavailable")
        measured.append({"model":model,"uid":row["uid"],"seconds":result["seconds"],
            "generated_tokens":result["generated_tokens"],"input_tokens":result["input_tokens"],
            "status":result["status"],"bytes":len(str(result).encode()),"load_seconds":result["load_seconds"]})
    # Native output allowance rather than an assumed short valid-label draw.
    upper=max(m["seconds"]/max(1,m["generated_tokens"])*campaign["annotator"]["config"]["max_tokens"] for m in measured)
    return publish(root/"ANNOTATOR_PRICE.json",{"status":"complete","records":measured,
        "seconds_per_context_upper":upper,"load_seconds":max(m["load_seconds"] for m in measured),
        "bytes_per_context_upper":max(m["bytes"] for m in measured)*4,
        "original_assignments_retained":True,"table_sha256":selected["sha256"]})


def worker(campaign,value):
    root=Path(campaign["root"])
    stage,model,shard=value["stage"],value["model"],value["shard_index"]
    require(value["campaign_sha256"]==campaign["sha256"],"card belongs to another campaign")
    check_files(campaign["code_files"])
    check_files(campaign["source_files"])
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
            return closeout(campaign)
    directory=root/"stages"/stage_key(stage,model,shard,value["threshold_index"])
    deadline=time.time()+value["minutes"]*60-120
    with locked(directory,"worker"):
        if (directory/"RESULT.json").exists():
            return load(directory/"RESULT.json")
        if not (directory/"START.json").exists():
            publish(directory/"START.json",{"job_id":os.environ["SLURM_JOB_ID"],"card_sha256":value["sha256"],"started_epoch":time.time()})
        started=time.monotonic()
        try:
            if stage=="resolve":
                from .inputs import resolve
                detail=resolve(campaign)
                detail={"status":"complete","reuse_sha256":detail["sha256"]}
            elif stage=="derive":
                from .derive import derive
                detail=derive(campaign,model)
            elif stage=="supplementary":
                from .observations import supplementary
                detail=supplementary(campaign,model)
            elif stage=="coordination":
                from .observations import coordination
                detail=coordination(campaign,model,shard)
            elif stage=="measure":
                from .workers import measure
                detail=measure(campaign,model,shard,deadline)
            elif stage=="native":
                from .workers import reuse_native
                detail=reuse_native(campaign,model,deadline)
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
                detail=prepare_contexts(campaign,model)
                detail={"status":"complete","task_sha256":detail["sha256"]}
            elif stage=="annotate":
                from .annotation import annotate
                table=load(root/model/"ANNOTATION_TASKS.json")["records"]
                slices=load(root/model/"ANNOTATION_SHARDS.json")["shards"]
                detail=annotate(campaign,model,[table[i] for i in slices[shard]],deadline)
            elif stage=="score":
                from .scoring import score
                table=load(root/model/"TASKS.json")["records"]
                if shard:
                    indices=load(root/model/"SCORE_SHARDS.json")["shards"][shard-1]
                    detail=score(campaign,model,[table[i] for i in indices])
                else:
                    scores=Units(root/model/"scores")
                    detail={"status":"complete","units":len(table),"missing":sum((scores.state(r["uid"]) or {}).get("status")!="complete" for r in table)}
            elif stage in ("closeout","snapshot"):
                from .export import closeout
                detail=closeout(campaign,final=stage=="closeout")
            else:
                raise ValueError("unknown registered stage: "+stage)
            result={"stage":stage,"model":model,"shard_index":shard,"threshold_index":value["threshold_index"],
                "status":detail.get("status","complete"),"detail":detail,"seconds":time.monotonic()-started,
                "job_id":os.environ["SLURM_JOB_ID"]}
        except Exception as exc:
            result={"stage":stage,"model":model,"shard_index":shard,"threshold_index":value["threshold_index"],
                "status":"execution_failed","error_type":type(exc).__name__,"error":str(exc)[:4000],
                "seconds":time.monotonic()-started,"job_id":os.environ["SLURM_JOB_ID"],"redraw":False}
        return publish(directory/"RESULT.json",result)
